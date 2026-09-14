"""Decision-analytic and uncertainty analyses for the JCCE revision.

Reads the exported predictions and sample keys, so it performs no model
fitting and runs in seconds.

Tasks
  clustered_ci   cluster and block bootstrap intervals for the screening metrics
  decision_curve net benefit against review-all and review-none, plus a
                 cost-sensitive sweep over the cost of a missed high-carbon mix
  heuristics     ranking baselines representing current triage practice
  within_class   screening performance inside each strength class
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

import numpy as np
import pandas as pd

import pipeline_common as R

PRED = R.RESULTS_DIR / "acrm_predictions_enriched.csv"
KEYS = R.RESULTS_DIR / "sample_keys.csv"
XGB = R.RESULTS_DIR / "validation_predictions.csv"
N_BOOT = 1000
REVIEW_FRACTIONS = [0.10, 0.20]
CLUSTER_FOR_SPLIT = {
    "group_company": "company",
    "group_epd_source": "source",
    "temporal_latest20": "issue_block",
}


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(PRED, parse_dates=["issue_date"])
    pred["issue_block"] = pred["issue_date"].dt.to_period("Q").astype(str)
    # The whole-sample key table is needed only by the heuristic baselines, so
    # the other tasks can run before it has been exported.
    keys = (
        pd.read_csv(KEYS, parse_dates=["issue_date"])
        if KEYS.exists()
        else pd.DataFrame(columns=["row_index"])
    )
    return pred, keys


# --------------------------------------------------------------------------
# screening statistics
# --------------------------------------------------------------------------

def screening_stats(frame: pd.DataFrame, score_col: str = "proba") -> dict[str, float]:
    y = frame["y_true"].to_numpy()
    s = frame[score_col].to_numpy()
    n = len(y)
    pos = float(y.sum())
    prevalence = pos / n if n else float("nan")
    out: dict[str, float] = {"n_test": n, "prevalence": prevalence}
    if pos == 0 or n == 0:
        return out
    order = np.argsort(-s, kind="stable")
    for frac in REVIEW_FRACTIONS:
        k = max(int(round(n * frac)), 1)
        top = order[:k]
        captured = float(y[top].sum())
        tag = f"top_{int(frac * 100)}pct"
        out[f"{tag}_capture"] = captured / pos
        out[f"{tag}_precision"] = captured / k
        out[f"{tag}_lift"] = (captured / k) / prevalence if prevalence else float("nan")
        random_reviews = captured / prevalence if prevalence else float("nan")
        out[f"{tag}_review_saving"] = 1 - (k / random_reviews) if random_reviews else float("nan")
        if "opportunity" in frame.columns:
            opp = frame["opportunity"].to_numpy()[top]
            opp = opp[(y[top] == 1) & np.isfinite(opp)]
            out[f"{tag}_median_opportunity"] = float(np.median(opp)) if len(opp) else float("nan")
            out[f"{tag}_share_opportunity_ge_10"] = float(np.mean(opp >= 0.10)) if len(opp) else float("nan")
    return out


# --------------------------------------------------------------------------
# comparable-group opportunity  (Eq. 8 of the manuscript)
# --------------------------------------------------------------------------

def attach_opportunity(frame: pd.DataFrame, min_alternatives: int = 5) -> pd.DataFrame:
    """Relative shortfall of each declaration against the 25th percentile of
    its comparable group, using the strict-to-relaxed fallback hierarchy."""
    frame = frame.copy()
    hierarchy = [
        ["region", "application_family", "strength_bin_500", "curing_family"],
        ["application_family", "strength_bin_500", "curing_family"],
        ["region", "strength_bin_500"],
    ]
    opportunity = pd.Series(np.nan, index=frame.index, dtype=float)
    group_used = pd.Series("none", index=frame.index, dtype=object)
    for level, keys in enumerate(hierarchy, start=1):
        pending = opportunity.isna()
        if not pending.any():
            break
        grouped = frame.groupby(keys, dropna=False)["gwp_per_ksi"]
        p25 = grouped.transform(lambda s: s.quantile(0.25))
        size = grouped.transform("size")
        eligible = pending & (size >= min_alternatives) & p25.notna() & (frame["gwp_per_ksi"] > 0)
        rel = (frame["gwp_per_ksi"] - p25) / frame["gwp_per_ksi"]
        opportunity[eligible] = rel[eligible]
        group_used[eligible] = f"level_{level}"
    frame["opportunity"] = opportunity
    frame["opportunity_group_level"] = group_used
    return frame


# --------------------------------------------------------------------------
# clustered bootstrap  (Reviewer 1, comment 3)
# --------------------------------------------------------------------------

def _bootstrap(
    frame: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], dict[str, float]],
    unit: str | None,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    draws: list[dict[str, float]] = []
    if unit is None:
        idx_all = np.arange(len(frame))
        for _ in range(n_boot):
            take = rng.integers(0, len(idx_all), len(idx_all))
            draws.append(stat_fn(frame.iloc[take]))
    else:
        groups = frame.groupby(unit).indices
        names = np.array(list(groups.keys()), dtype=object)
        for _ in range(n_boot):
            picked = rng.integers(0, len(names), len(names))
            take = np.concatenate([groups[names[i]] for i in picked])
            draws.append(stat_fn(frame.iloc[take]))
    return pd.DataFrame(draws)


def task_clustered_ci(pred: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in pred.groupby("split"):
        group = attach_opportunity(group.reset_index(drop=True))
        point = screening_stats(group)
        schemes = {
            "row_level": None,
            "cluster_company": "company",
            "cluster_plant": "plant",
            "cluster_source": "source",
            "block_issue_quarter": "issue_block",
        }
        for scheme, unit in schemes.items():
            if unit is not None and group[unit].nunique() < 5:
                continue
            draws = _bootstrap(group, screening_stats, unit, N_BOOT, seed=R.BASE_SEED)
            for metric in [c for c in draws.columns if c not in ("n_test", "prevalence")]:
                series = draws[metric].dropna()
                if series.empty:
                    continue
                rows.append({
                    "split": split,
                    "resampling": scheme,
                    "resampling_unit": unit or "row",
                    "n_units": int(group[unit].nunique()) if unit else int(len(group)),
                    "metric": metric,
                    "point_estimate": point.get(metric, float("nan")),
                    "ci_low": float(series.quantile(0.025)),
                    "ci_high": float(series.quantile(0.975)),
                    "ci_width": float(series.quantile(0.975) - series.quantile(0.025)),
                    "n_test": point["n_test"],
                    "prevalence": point["prevalence"],
                    "n_companies": int(group["company"].nunique()),
                    "n_plants": int(group["plant"].nunique()),
                    "n_sources": int(group["source"].nunique()),
                })
            print(f"[ci] {split} / {scheme} done", flush=True)
    frame = pd.DataFrame(rows)
    R.write_table(frame, "clustered_bootstrap_ci.csv")

    wide = frame.pivot_table(
        index=["split", "metric", "point_estimate"], columns="resampling", values="ci_width"
    ).reset_index()
    R.write_table(wide, "bootstrap_ci_width_comparison.csv")
    return frame


# --------------------------------------------------------------------------
# decision curve and cost sensitivity  (Reviewer 1, comment 4)
# --------------------------------------------------------------------------

def _net_benefit(y: np.ndarray, s: np.ndarray, pt: float) -> float:
    n = len(y)
    flag = s >= pt
    tp = float(np.sum(flag & (y == 1)))
    fp = float(np.sum(flag & (y == 0)))
    return tp / n - (fp / n) * (pt / (1 - pt))


def task_decision_curve(pred: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    xgb = None
    if XGB.exists():
        xgb = pd.read_csv(XGB)[["split", "row_index", "xgb_probability"]]

    rows: list[dict[str, Any]] = []
    thresholds = np.round(np.arange(0.02, 0.81, 0.01), 3)
    for split, group in pred.groupby("split"):
        group = group.reset_index(drop=True)
        y = group["y_true"].to_numpy()
        prevalence = float(y.mean())
        models = {"ACRM": group["proba"].to_numpy()}
        if xgb is not None:
            merged = group.merge(xgb[xgb["split"] == split], on=["split", "row_index"], how="left")
            if merged["xgb_probability"].notna().all():
                models["XGBoost baseline"] = merged["xgb_probability"].to_numpy()
        for pt in thresholds:
            odds = pt / (1 - pt)
            nb_all = prevalence - (1 - prevalence) * odds
            for label, s in models.items():
                nb = _net_benefit(y, s, pt)
                rows.append({
                    "split": split,
                    "threshold_probability": pt,
                    "strategy": label,
                    "net_benefit": nb,
                    "net_benefit_review_all": nb_all,
                    "net_benefit_review_none": 0.0,
                    "reviews_avoided_per_100": (nb - nb_all) / odds * 100 if odds else np.nan,
                    "share_flagged": float(np.mean(s >= pt)),
                })
        print(f"[dca] {split} done", flush=True)
    frame = pd.DataFrame(rows)
    R.write_table(frame, "decision_curve.csv")

    # Cost-sensitive review budget: cost of reviewing one EPD is the unit, the
    # cost of leaving a high-carbon mix unexamined is `ratio` units.
    cost_rows: list[dict[str, Any]] = []
    fractions = np.round(np.arange(0.02, 1.001, 0.02), 3)
    for split, group in pred.groupby("split"):
        y = group["y_true"].to_numpy()
        s = group["proba"].to_numpy()
        n = len(y)
        pos = float(y.sum())
        order = np.argsort(-s, kind="stable")
        for ratio in [2, 5, 10, 20, 50, 100]:
            best = None
            for frac in fractions:
                k = max(int(round(n * frac)), 1)
                captured = float(y[order[:k]].sum())
                cost_model = k + ratio * (pos - captured)
                cost_random = k + ratio * (pos * (1 - frac))
                row = {
                    "split": split, "cost_ratio_missed_to_review": ratio,
                    "review_fraction": frac, "cost_model": cost_model,
                    "cost_random": cost_random,
                    "cost_review_all": float(n), "cost_review_none": ratio * pos,
                }
                if best is None or cost_model < best["cost_model"]:
                    best = row
                cost_rows.append(row)
            best = dict(best)
            best["is_optimum"] = True
            cost_rows.append(best)
    cost = pd.DataFrame(cost_rows)
    R.write_table(cost, "cost_sensitive_budget.csv")

    optima = (
        cost[cost.get("is_optimum", False) == True]  # noqa: E712
        .drop(columns=["is_optimum"])
        .assign(
            saving_vs_review_all=lambda d: 1 - d["cost_model"] / d["cost_review_all"],
            saving_vs_review_none=lambda d: 1 - d["cost_model"] / d["cost_review_none"],
            saving_vs_random=lambda d: 1 - d["cost_model"] / d["cost_random"],
        )
    )
    R.write_table(optima, "cost_sensitive_optima.csv")
    return frame


# --------------------------------------------------------------------------
# practice-heuristic ranking baselines  (Reviewer 2, comment 4)
# --------------------------------------------------------------------------

def task_heuristics(pred: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(R.BASE_SEED)
    rows: list[dict[str, Any]] = []
    keys = keys.set_index("row_index")

    for split, group in pred.groupby("split"):
        group = attach_opportunity(group.reset_index(drop=True))
        test_ids = set(group["row_index"])
        train = keys.loc[~keys.index.isin(test_ids)]
        supplier_mean = train.groupby("company")["gwp_per_ksi"].mean()
        global_mean = float(train["gwp_per_ksi"].mean())
        test_keys = keys.loc[group["row_index"]]

        scm_any = test_keys["scm_any"].to_numpy() if "scm_any" in test_keys.columns else np.zeros(len(group))
        supplier_score = test_keys["company"].map(supplier_mean).to_numpy()
        coverage = float(np.mean(~pd.isna(supplier_score)))
        supplier_score = np.where(pd.isna(supplier_score), global_mean, supplier_score)

        candidates: dict[str, np.ndarray] = {
            "ACRM (proposed)": group["proba"].to_numpy(),
            "Unordered review": rng.random(len(group)),
            "Arrival order (oldest first)": -group["issue_date"].astype("int64").to_numpy(),
            "Highest declared strength first": test_keys["strength_psi"].to_numpy(),
            "Lowest declared strength first": -test_keys["strength_psi"].to_numpy(),
            "No declared SCM first": 1 - scm_any + rng.random(len(group)) * 1e-6,
            "Supplier carbon history": supplier_score,
            "Highest declared GWP first": test_keys["gwp"].to_numpy(),
        }
        for label, score in candidates.items():
            tmp = group.copy()
            tmp["score"] = score
            stats = screening_stats(tmp, score_col="score")
            row = {"split": split, "strategy": label}
            row.update(stats)
            if label == "Supplier carbon history":
                row["supplier_history_coverage"] = coverage
            rows.append(row)
            print(f"[heur] {split} / {label}: cap@20 "
                  f"{stats.get('top_20pct_capture', float('nan')):.3f}", flush=True)
    frame = pd.DataFrame(rows)
    R.write_table(frame, "practice_heuristic_baselines.csv")
    return frame


# --------------------------------------------------------------------------
# within-strength-class performance  (Reviewer 2, comment 1)
# --------------------------------------------------------------------------

def task_within_class(pred: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score, average_precision_score

    rows: list[dict[str, Any]] = []
    for split, group in pred.groupby("split"):
        for band, sub in group.groupby("strength_bin_500"):
            y = sub["y_true"].to_numpy()
            if len(sub) < 200 or y.sum() < 10 or y.sum() == len(y):
                continue
            s = sub["proba"].to_numpy()
            rows.append({
                "split": split,
                "strength_bin_500_psi": band,
                "n_test": int(len(sub)),
                "prevalence": float(y.mean()),
                "roc_auc_within_class": float(roc_auc_score(y, s)),
                "average_precision_within_class": float(average_precision_score(y, s)),
                "top_20pct_capture_within_class": R.topk_capture(y, s, 0.20),
                "top_20pct_precision_within_class": R.topk_precision(y, s, 0.20),
            })
    frame = pd.DataFrame(rows).sort_values(["split", "strength_bin_500_psi"])
    R.write_table(frame, "within_strength_class.csv")

    summary = (
        frame.groupby("split")
        .apply(
            lambda d: pd.Series({
                "n_classes": len(d),
                "weighted_roc_auc": np.average(d["roc_auc_within_class"], weights=d["n_test"]),
                "min_roc_auc": d["roc_auc_within_class"].min(),
                "max_roc_auc": d["roc_auc_within_class"].max(),
                "weighted_top20_capture": np.average(
                    d["top_20pct_capture_within_class"], weights=d["n_test"]
                ),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    R.write_table(summary, "within_strength_class_summary.csv")
    return frame


TASKS = {
    "clustered_ci": task_clustered_ci,
    "decision_curve": task_decision_curve,
    "heuristics": task_heuristics,
    "within_class": task_within_class,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS))
    args = parser.parse_args()
    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    pred, keys = _load()
    for name in requested:
        print(f"=== task {name} ===", flush=True)
        TASKS[name](pred, keys)
    R.write_manifest(
        {
            "analysis": "JCCE revision decision-analytic batch",
            "bootstrap_repeats": N_BOOT,
            "review_fractions": REVIEW_FRACTIONS,
            "tasks_completed": requested,
            "predictions_source": str(PRED),
        },
        "decision_manifest.json",
    )


if __name__ == "__main__":
    main()
