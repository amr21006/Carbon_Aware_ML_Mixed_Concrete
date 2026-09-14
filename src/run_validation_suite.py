"""Revision validation experiments for the JCCE major revision.

Each task re-executes the published ACRM configuration under a different
validation condition. Outputs are written incrementally so that a long batch
can be inspected while it is still running.

Tasks
  baseline    refit the three reported holdouts and export enriched predictions
  stability   repeated group splits across seeds and a rolling-origin backtest
  ablations   feature-subset ablations, including identity and text removal
  targets     alternative target normalisations and label thresholds
  duplicates  duplicate audit and a de-duplicated sensitivity fit
  threshold   trimming and label threshold estimated on training rows only
  operator    leave-one-programme-operator-out and leave-one-region-out
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import pandas as pd

import pipeline_common as R
import concrete_epd_pipeline as M
import run_acrm_model as A

SEEDS = list(range(1, 11))
PUBLISHED = ["group_company", "group_epd_source", "temporal_latest20"]


class _FS:
    def __init__(self, include_lci_sources: bool) -> None:
        self.include_lci_sources = include_lci_sources


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# baseline
# --------------------------------------------------------------------------

def task_baseline(ctx: R.Context) -> pd.DataFrame:
    """Refit the three reported holdouts and export predictions carrying every
    clustering key the uncertainty and decision analyses need."""
    splits = ctx.published_splits()
    app_family = A.make_application_family(ctx.base).reset_index(drop=True)
    curing_family = A.make_curing_family(ctx.base["curing_days"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    preds: list[pd.DataFrame] = []
    for name in PUBLISHED:
        train_idx, test_idx = splits[name]
        t0 = time.time()
        # Use the reported model's stored out-of-sample scores when they are
        # available, so every downstream analysis describes the same booster
        # that the main results tables report. GPU histogram building is not
        # bit-reproducible, and a refit differs from the reported model in the
        # third decimal of every metric.
        stored = R.RESULTS_DIR / "acrm_single_model_predictions.csv"
        proba = None
        if stored.exists():
            sp = pd.read_csv(stored)
            sp = sp[sp["split"] == name].set_index("row_index")
            if set(sp.index) == set(test_idx):
                proba = sp.loc[test_idx, "acrm_probability"].to_numpy()
                _log(f"baseline {name}: using stored reported-model scores")
        if proba is None:
            proba = R.fit_predict(
                ctx.model_frame, ctx.y, ctx.numeric_cols, ctx.categorical_cols, train_idx, test_idx
            )
        y_test = np.asarray(ctx.y.iloc[test_idx])
        row = {"split": name, "n_train": int(len(train_idx))}
        row.update(R.metric_row(y_test, proba))
        row["n_company_test"] = int(ctx.company.iloc[test_idx].nunique())
        row["n_plant_test"] = int(ctx.plant.iloc[test_idx].nunique())
        row["n_source_test"] = int(ctx.source.iloc[test_idx].nunique())
        row["runtime_s"] = round(time.time() - t0, 1)
        rows.append(row)
        _log(f"baseline {name}: AUC {row['roc_auc']:.4f} AP {row['average_precision']:.4f}")
        preds.append(
            pd.DataFrame(
                {
                    "split": name,
                    "row_index": test_idx,
                    "y_true": y_test,
                    "proba": proba,
                    "company": ctx.company.iloc[test_idx].to_numpy(),
                    "plant": ctx.plant.iloc[test_idx].to_numpy(),
                    "source": ctx.source.iloc[test_idx].to_numpy(),
                    "operator": ctx.operator.iloc[test_idx].to_numpy(),
                    "region": ctx.region.iloc[test_idx].to_numpy(),
                    "strength_psi": ctx.strength.iloc[test_idx].to_numpy(),
                    "strength_bin_500": ctx.strength_bin.iloc[test_idx].to_numpy(),
                    "gwp": ctx.gwp.iloc[test_idx].to_numpy(),
                    "gwp_per_ksi": ctx.gwp_per_ksi.iloc[test_idx].to_numpy(),
                    "issue_date": ctx.issue_date.iloc[test_idx].to_numpy(),
                    "application_family": app_family.iloc[test_idx].to_numpy(),
                    "curing_family": curing_family.iloc[test_idx].to_numpy(),
                }
            )
        )
    R.write_table(pd.concat(preds, ignore_index=True), "acrm_predictions_enriched.csv")
    out = pd.DataFrame(rows)
    R.write_table(out, "baseline_metrics.csv")
    return out


# --------------------------------------------------------------------------
# stability across seeds and folds  (Reviewer 1, comment 2)
# --------------------------------------------------------------------------

def task_stability(ctx: R.Context) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = {"group_company": ctx.company, "group_epd_source": ctx.source}
    for split_name, groups in grouping.items():
        for seed in SEEDS:
            train_idx, test_idx = ctx.group_split(groups, seed=seed)
            proba = R.fit_predict(
                ctx.model_frame, ctx.y, ctx.numeric_cols, ctx.categorical_cols,
                train_idx, test_idx, seed=seed,
            )
            y_test = np.asarray(ctx.y.iloc[test_idx])
            row = {"design": split_name, "replicate": seed, "n_train": int(len(train_idx))}
            row.update(R.metric_row(y_test, proba))
            row["n_clusters_test"] = int(groups.iloc[test_idx].nunique())
            rows.append(row)
            _log(f"stability {split_name} seed {seed}: AUC {row['roc_auc']:.4f} "
                 f"cap@20 {row['top_20pct_recall_capture']:.4f}")
            R.write_table(pd.DataFrame(rows), "stability_replicates.csv")

    # Rolling-origin temporal backtest: successive forward windows.
    for i, end_fraction in enumerate([0.70, 0.80, 0.90, 1.00], start=1):
        train_idx, test_idx = ctx.temporal_split(test_fraction=0.10, end_fraction=end_fraction)
        proba = R.fit_predict(
            ctx.model_frame, ctx.y, ctx.numeric_cols, ctx.categorical_cols, train_idx, test_idx
        )
        y_test = np.asarray(ctx.y.iloc[test_idx])
        dates = ctx.issue_date.iloc[test_idx]
        row = {"design": "temporal_rolling_origin", "replicate": i, "n_train": int(len(train_idx))}
        row.update(R.metric_row(y_test, proba))
        row["n_clusters_test"] = int(ctx.company.iloc[test_idx].nunique())
        row["test_window_start"] = str(dates.min())
        row["test_window_end"] = str(dates.max())
        rows.append(row)
        _log(f"stability rolling origin {i} (end {end_fraction:.2f}): AUC {row['roc_auc']:.4f}")
        R.write_table(pd.DataFrame(rows), "stability_replicates.csv")

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("design")[
            ["roc_auc", "average_precision", "top_10pct_recall_capture",
             "top_20pct_recall_capture", "top_20pct_precision", "ece_10_bins", "prevalence"]
        ]
        .agg(["mean", "std", "min", "max", "count"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    R.write_table(summary.reset_index(), "stability_summary.csv")
    return frame


# --------------------------------------------------------------------------
# feature ablations  (Reviewer 1 comment 2; Reviewer 2 comment 1)
# --------------------------------------------------------------------------

def _identity_columns(cols: list[str]) -> list[str]:
    keys = ("company", "plant", "state", "region")
    return [c for c in cols if any(k in c.lower() for k in keys)]


def _strength_columns(cols: list[str]) -> list[str]:
    keys = ("strength", "psi")
    return [c for c in cols if any(k in c.lower() for k in keys)]


def task_ablations(ctx: R.Context) -> pd.DataFrame:
    base_text = ctx.model_frame["text_feature"]
    no_lci_text = (
        M.build_text_feature(ctx.base, _FS(include_lci_sources=False))
        + " "
        + A.selected_app_tokens(ctx.base)
    ).reset_index(drop=True)

    variants: dict[str, dict[str, Any]] = {
        "full_acrm": {},
        "no_supplier_identity": {
            "drop_categorical": _identity_columns(ctx.categorical_cols),
            "drop_numeric": [],
        },
        "no_free_text": {"text": pd.Series(["none"] * ctx.n)},
        "no_lci_source_text": {"text": no_lci_text},
        "structured_metadata_only": {
            "text": pd.Series(["none"] * ctx.n),
            "drop_categorical": _identity_columns(ctx.categorical_cols),
        },
        "no_strength_features": {
            "drop_numeric": _strength_columns(ctx.numeric_cols),
            "drop_categorical": _strength_columns(ctx.categorical_cols),
        },
    }

    splits = ctx.published_splits()
    rows: list[dict[str, Any]] = []
    for name, spec in variants.items():
        frame = ctx.model_frame.copy()
        if "text" in spec:
            frame["text_feature"] = spec["text"].to_numpy()
        numeric = [c for c in ctx.numeric_cols if c not in set(spec.get("drop_numeric", []))]
        categorical = [
            c for c in ctx.categorical_cols if c not in set(spec.get("drop_categorical", []))
        ]
        for split_name in PUBLISHED:
            train_idx, test_idx = splits[split_name]
            proba = R.fit_predict(frame, ctx.y, numeric, categorical, train_idx, test_idx)
            y_test = np.asarray(ctx.y.iloc[test_idx])
            row = {
                "variant": name,
                "split": split_name,
                "n_numeric": len(numeric),
                "n_categorical": len(categorical),
                "text_included": "text" not in spec or spec["text"].iloc[0] != "none",
            }
            row.update(R.metric_row(y_test, proba))
            rows.append(row)
            _log(f"ablation {name} / {split_name}: AUC {row['roc_auc']:.4f} "
                 f"cap@20 {row['top_20pct_recall_capture']:.4f}")
            R.write_table(pd.DataFrame(rows), "ablation_metrics.csv")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# alternative target definitions  (Reviewer 1 comment 1)
# --------------------------------------------------------------------------

def _labels_within_class(ctx: R.Context, quantile: float) -> pd.Series:
    """Top-decile carbon intensity computed inside each 500 psi strength class,
    pooling classes with fewer than 200 records."""
    ci = ctx.gwp_per_ksi
    bins = ctx.strength_bin.copy()
    counts = bins.value_counts()
    small = set(counts[counts < 200].index)
    bins = bins.where(~bins.isin(small), "pooled_sparse_classes")
    out = pd.Series(0, index=ci.index, dtype=int)
    for value, idx in bins.groupby(bins).groups.items():
        sub = ci.loc[idx]
        out.loc[idx] = (sub >= sub.quantile(quantile)).astype(int)
    return out


def task_targets(ctx: R.Context) -> pd.DataFrame:
    primary = ctx.y
    strength = ctx.strength.clip(lower=1)
    definitions: dict[str, pd.Series] = {
        "primary_gwp_per_ksi": primary,
        "within_strength_class": _labels_within_class(ctx, R.MAIN_QUANTILE),
        "gwp_per_declared_unit": (ctx.gwp >= ctx.gwp.quantile(R.MAIN_QUANTILE)).astype(int),
        "gwp_per_sqrt_strength": (
            lambda s: (s >= s.quantile(R.MAIN_QUANTILE)).astype(int)
        )(ctx.gwp / np.sqrt(strength / 1000.0)),
        "gwp_per_strength_pow075": (
            lambda s: (s >= s.quantile(R.MAIN_QUANTILE)).astype(int)
        )(ctx.gwp / np.power(strength / 1000.0, 0.75)),
    }

    splits = ctx.published_splits()
    rows: list[dict[str, Any]] = []
    for name, labels in definitions.items():
        labels = labels.reset_index(drop=True)
        agree = float((labels == primary).mean())
        both = float(((labels == 1) & (primary == 1)).sum())
        union = float(((labels == 1) | (primary == 1)).sum())
        jaccard = both / union if union else float("nan")
        po = agree
        pe = float(labels.mean() * primary.mean() + (1 - labels.mean()) * (1 - primary.mean()))
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        for split_name in PUBLISHED:
            train_idx, test_idx = splits[split_name]
            proba = R.fit_predict(
                ctx.model_frame, labels, ctx.numeric_cols, ctx.categorical_cols, train_idx, test_idx
            )
            y_test = np.asarray(labels.iloc[test_idx])
            row = {
                "target": name,
                "split": split_name,
                "positive_rate_overall": float(labels.mean()),
                "agreement_with_primary": round(agree, 4),
                "jaccard_with_primary": round(jaccard, 4),
                "cohens_kappa_with_primary": round(kappa, 4),
            }
            row.update(R.metric_row(y_test, proba))
            rows.append(row)
            _log(f"target {name} / {split_name}: AUC {row['roc_auc']:.4f} "
                 f"kappa {kappa:.3f}")
            R.write_table(pd.DataFrame(rows), "target_sensitivity.csv")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# duplicates and versions  (Reviewer 1 comment 2)
# --------------------------------------------------------------------------

def task_duplicates(ctx: R.Context) -> pd.DataFrame:
    b = ctx.base
    audit: list[dict[str, Any]] = []
    key_sets = {
        "company_plant_label_strength_gwp": [
            "Company", "Plant", "Mixture Label", "strength_psi", M.TARGET_COL
        ],
        "company_plant_label": ["Company", "Plant", "Mixture Label"],
        "company_plant_label_issue_date": [
            "Company", "Plant", "Mixture Label", "EPD Date of Issue"
        ],
    }
    for name, keys in key_sets.items():
        keys = [k for k in keys if k in b.columns]
        audit.append({
            "key": name,
            "columns": "; ".join(keys),
            "n_duplicate_rows": int(b.duplicated(subset=keys).sum()),
            "n_unique_keys": int(b.drop_duplicates(subset=keys).shape[0]),
            "n_rows": int(len(b)),
        })
    R.write_table(pd.DataFrame(audit), "duplicate_audit.csv")

    # Version rule: retain the most recent declaration per company / plant /
    # mixture label, which is how a procurement team would treat a reissue.
    keys = [k for k in ["Company", "Plant", "Mixture Label"] if k in b.columns]
    order = b["EPD Date of Issue"]
    keep = (
        b.assign(_o=pd.to_datetime(order, errors="coerce"), _i=np.arange(len(b)))
        .sort_values(["_o", "_i"])
        .drop_duplicates(subset=keys, keep="last")["_i"]
        .sort_values()
        .to_numpy()
    )
    _log(f"duplicates: retaining {len(keep)} of {len(b)} records under the latest-issue rule")

    sub_frame = ctx.model_frame.iloc[keep].reset_index(drop=True)
    sub_y = ctx.y.iloc[keep].reset_index(drop=True)
    sub_company = ctx.company.iloc[keep].reset_index(drop=True)
    sub_source = ctx.source.iloc[keep].reset_index(drop=True)
    sub_date = ctx.issue_date.iloc[keep].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    from sklearn.model_selection import GroupShuffleSplit

    designs = {"group_company": sub_company, "group_epd_source": sub_source}
    for split_name, groups in designs.items():
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=R.BASE_SEED + 17)
        train_idx, test_idx = next(splitter.split(np.arange(len(sub_y)), sub_y, groups=groups))
        proba = R.fit_predict(
            sub_frame, sub_y, ctx.numeric_cols, ctx.categorical_cols, train_idx, test_idx
        )
        y_test = np.asarray(sub_y.iloc[test_idx])
        row = {"sample": "deduplicated_latest_issue", "split": split_name,
               "n_rows": int(len(sub_y)), "n_train": int(len(train_idx))}
        row.update(R.metric_row(y_test, proba))
        rows.append(row)
        _log(f"dedup {split_name}: AUC {row['roc_auc']:.4f}")
        R.write_table(pd.DataFrame(rows), "duplicate_sensitivity.csv")

    order_idx = np.argsort(sub_date.fillna(pd.Timestamp("2100-01-01")).to_numpy(), kind="stable")
    cut = int(round(len(order_idx) * 0.8))
    train_idx, test_idx = order_idx[:cut], order_idx[cut:]
    proba = R.fit_predict(
        sub_frame, sub_y, ctx.numeric_cols, ctx.categorical_cols, train_idx, test_idx
    )
    y_test = np.asarray(sub_y.iloc[test_idx])
    row = {"sample": "deduplicated_latest_issue", "split": "temporal_latest20",
           "n_rows": int(len(sub_y)), "n_train": int(len(train_idx))}
    row.update(R.metric_row(y_test, proba))
    rows.append(row)
    _log(f"dedup temporal: AUC {row['roc_auc']:.4f}")
    R.write_table(pd.DataFrame(rows), "duplicate_sensitivity.csv")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# training-only trimming and label threshold  (Reviewer 1 comment 2)
# --------------------------------------------------------------------------

def task_threshold(ctx: R.Context) -> pd.DataFrame:
    """The reported labels use a 90th percentile and a 99.5th percentile trim
    computed on the pooled sample. Here both are estimated on training rows
    only and applied unchanged to the test rows."""
    splits = ctx.published_splits()
    rows: list[dict[str, Any]] = []
    for split_name in PUBLISHED:
        train_idx, test_idx = splits[split_name]
        train_ci = ctx.gwp_per_ksi.iloc[train_idx]
        train_gwp = ctx.gwp.iloc[train_idx]
        thr = float(train_ci.quantile(R.MAIN_QUANTILE))
        trim_ci = float(train_ci.quantile(0.995))
        trim_gwp = float(train_gwp.quantile(0.995))
        labels = (ctx.gwp_per_ksi >= thr).astype(int)
        keep = (ctx.gwp_per_ksi < trim_ci) & (ctx.gwp < trim_gwp)
        tr = np.array([i for i in train_idx if keep.iloc[i]])
        te = np.array([i for i in test_idx if keep.iloc[i]])
        proba = R.fit_predict(ctx.model_frame, labels, ctx.numeric_cols, ctx.categorical_cols, tr, te)
        y_test = np.asarray(labels.iloc[te])
        row = {
            "split": split_name,
            "labeling": "train_only_threshold_and_trim",
            "train_threshold_kg_per_ksi": round(thr, 2),
            "pooled_threshold_kg_per_ksi": round(float(ctx.gwp_per_ksi.quantile(R.MAIN_QUANTILE)), 2),
            "n_train": int(len(tr)),
            "label_changes_vs_pooled": int((labels != ctx.y).sum()),
        }
        row.update(R.metric_row(y_test, proba))
        rows.append(row)
        _log(f"train-only threshold {split_name}: thr {thr:.2f} AUC {row['roc_auc']:.4f}")
        R.write_table(pd.DataFrame(rows), "train_only_threshold.csv")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# quasi-external validation  (Reviewer 1 comment 4)
# --------------------------------------------------------------------------

def task_operator(ctx: R.Context) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for held in sorted(ctx.operator.unique()):
        mask = (ctx.operator == held).to_numpy()
        if mask.sum() < 200 or (~mask).sum() < 200:
            continue
        train_idx = np.flatnonzero(~mask)
        test_idx = np.flatnonzero(mask)
        proba = R.fit_predict(
            ctx.model_frame, ctx.y, ctx.numeric_cols, ctx.categorical_cols, train_idx, test_idx
        )
        y_test = np.asarray(ctx.y.iloc[test_idx])
        row = {"design": "leave_one_programme_operator_out", "held_out": held,
               "n_train": int(len(train_idx))}
        row.update(R.metric_row(y_test, proba))
        rows.append(row)
        _log(f"operator holdout {held}: AUC {row['roc_auc']:.4f} n={len(test_idx)}")
        R.write_table(pd.DataFrame(rows), "quasi_external_validation.csv")

    counts = ctx.region.value_counts()
    for held in counts[counts >= 2000].index[:5]:
        mask = (ctx.region == held).to_numpy()
        train_idx = np.flatnonzero(~mask)
        test_idx = np.flatnonzero(mask)
        proba = R.fit_predict(
            ctx.model_frame, ctx.y, ctx.numeric_cols, ctx.categorical_cols, train_idx, test_idx
        )
        y_test = np.asarray(ctx.y.iloc[test_idx])
        row = {"design": "leave_one_region_out", "held_out": held, "n_train": int(len(train_idx))}
        row.update(R.metric_row(y_test, proba))
        rows.append(row)
        _log(f"region holdout {held}: AUC {row['roc_auc']:.4f} n={len(test_idx)}")
        R.write_table(pd.DataFrame(rows), "quasi_external_validation.csv")
    return pd.DataFrame(rows)


TASKS = {
    "baseline": task_baseline,
    "stability": task_stability,
    "ablations": task_ablations,
    "targets": task_targets,
    "duplicates": task_duplicates,
    "threshold": task_threshold,
    "operator": task_operator,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS))
    args = parser.parse_args()
    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TASKS]
    if unknown:
        raise SystemExit(f"unknown tasks: {unknown}; available: {list(TASKS)}")

    ctx = R.get_context()
    started = time.time()
    completed: list[str] = []
    for name in requested:
        t0 = time.time()
        _log(f"=== task {name} ===")
        TASKS[name](ctx)
        completed.append(name)
        _log(f"=== task {name} done in {time.time() - t0:.0f}s ===")

    R.write_manifest(
        {
            "analysis": "JCCE revision validation batch",
            "dataset_version": 5,
            "csv_path": str(R.V5_CSV),
            "rows": ctx.n,
            "target_quantile": R.MAIN_QUANTILE,
            "seeds": SEEDS,
            "tasks_completed": completed,
            "runtime_seconds": round(time.time() - started, 1),
            "clusters": {
                "companies": int(ctx.company.nunique()),
                "company_plants": int(ctx.plant.nunique()),
                "epd_source_links": int(ctx.source.nunique()),
                "programme_operators": int(ctx.operator.nunique()),
                "regions": int(ctx.region.nunique()),
            },
        },
        "validation_manifest.json",
    )


if __name__ == "__main__":
    main()
