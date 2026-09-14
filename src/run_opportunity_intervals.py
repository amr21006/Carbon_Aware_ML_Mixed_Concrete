"""Cluster-bootstrap intervals for the comparable-group opportunity screen.

The point estimates are reproduced from the published ACRM predictions using the
same group hierarchy, minimum group size, and clipping rule as the reported
analysis, so only the uncertainty method changes. The reference percentiles are
recomputed inside every bootstrap draw, because the comparable group is itself
estimated from the same pool.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

import pipeline_common as R
import concrete_epd_pipeline as M
import run_concrete_epd_procurement_opportunity as O

PRED_PATH = R.RESULTS_DIR / "acrm_single_model_predictions.csv"
MIN_ALTERNATIVES = 5
REVIEW_FRACTIONS = [0.10, 0.20]
N_BOOT = 1000
GROUP_ORDER = ["strict_group", "national_spec_group", "regional_strength_group"]


def opportunity_summary(pool: pd.DataFrame, review_fraction: float) -> dict[str, float]:
    """Vectorised equivalent of the reported opportunity screen."""
    n_select = max(1, int(np.ceil(len(pool) * review_fraction)))
    order = np.argsort(pool["score"].to_numpy())[-n_select:][::-1]
    flagged = pool.iloc[order]

    p25 = pd.Series(np.nan, index=flagged.index, dtype=float)
    for col in GROUP_ORDER:
        pending = p25.isna()
        if not pending.any():
            break
        # An aggregation with a lambda is an order of magnitude slower here, and
        # this runs inside the bootstrap loop.
        grouped = pool.groupby(col, dropna=False, sort=False)["gwp_per_ksi"]
        sizes = grouped.size()
        q25 = grouped.quantile(0.25)
        mapped = flagged[col].map(q25.where(sizes >= MIN_ALTERNATIVES))
        p25 = p25.where(~pending, mapped)

    gwp = flagged["gwp_per_ksi"].to_numpy()
    ref = p25.to_numpy()
    abs_red = np.clip(gwp - ref, 0, None)
    rel_red = abs_red / gwp
    has_group = ~np.isnan(ref)
    y = flagged["y_true"].to_numpy()

    true_high = y == 1
    th_alt = true_high & has_group
    pos_total = float(pool["y_true"].sum())

    out: dict[str, float] = {
        "flagged_count": int(len(flagged)),
        "flagged_true_high_carbon_count": int(true_high.sum()),
        "capture": float(true_high.sum() / pos_total) if pos_total else np.nan,
        "precision": float(true_high.mean()) if len(flagged) else np.nan,
        "comparable_group_coverage_true_high_carbon": (
            float(th_alt.sum() / true_high.sum()) if true_high.sum() else np.nan
        ),
    }
    if th_alt.sum():
        out["median_relative_reduction_to_p25_true_high_carbon"] = float(np.median(rel_red[th_alt]))
        out["share_true_high_carbon_with_10pct_opportunity"] = float(np.mean(rel_red[th_alt] >= 0.10))
        out["share_true_high_carbon_with_20pct_opportunity"] = float(np.mean(rel_red[th_alt] >= 0.20))
        out["median_absolute_reduction_to_p25_true_high_carbon"] = float(np.median(abs_red[th_alt]))
    return out


def build_pool(base: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    pool = base.iloc[pred["row_index"].to_numpy()].copy().reset_index(drop=True)
    pool["y_true"] = pred["y_true"].to_numpy()
    pool["score"] = pred["acrm_probability"].to_numpy()
    pool["company"] = pool["Company"].fillna("missing").astype(str)
    pool["plant"] = pool["company"] + "||" + pool["Plant"].fillna("missing").astype(str)
    pool["source"] = pool["EPD Source Link"].fillna("missing").astype(str)
    pool["issue_block"] = (
        pd.to_datetime(pool["EPD Date of Issue"], errors="coerce").dt.to_period("Q").astype(str)
    )
    return pool


def main() -> None:
    os.environ["CONCRETE_EPD_CSV"] = str(R.V5_CSV)
    base, _ = M.load_and_prepare()
    base = O.add_context_columns(base).reset_index(drop=True)
    predictions = pd.read_csv(PRED_PATH)

    point_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(R.BASE_SEED)

    for split in ["temporal_latest20", "group_company", "group_epd_source"]:
        pred = predictions[predictions["split"] == split].reset_index(drop=True)
        pool = build_pool(base, pred)
        for frac in REVIEW_FRACTIONS:
            point = opportunity_summary(pool, frac)
            point_rows.append({"split": split, "review_fraction": frac, **point})
            print(f"[point] {split} top{int(frac*100)}%: "
                  f"median opportunity {point.get('median_relative_reduction_to_p25_true_high_carbon', float('nan')):.6f} "
                  f"capture {point.get('capture', float('nan')):.4f}", flush=True)

            for scheme, unit in [("row_level", None), ("cluster_company", "company"),
                                 ("cluster_plant", "plant"), ("cluster_source", "source"),
                                 ("block_issue_quarter", "issue_block")]:
                if unit is not None and pool[unit].nunique() < 5:
                    continue
                draws: list[dict[str, float]] = []
                if unit is None:
                    for _ in range(N_BOOT):
                        take = rng.integers(0, len(pool), len(pool))
                        draws.append(opportunity_summary(pool.iloc[take].reset_index(drop=True), frac))
                else:
                    groups = pool.groupby(unit).indices
                    names = np.array(list(groups.keys()), dtype=object)
                    for _ in range(N_BOOT):
                        picked = rng.integers(0, len(names), len(names))
                        take = np.concatenate([groups[names[i]] for i in picked])
                        draws.append(opportunity_summary(pool.iloc[take].reset_index(drop=True), frac))
                frame = pd.DataFrame(draws)
                for metric in ["capture", "precision",
                               "median_relative_reduction_to_p25_true_high_carbon",
                               "share_true_high_carbon_with_10pct_opportunity",
                               "share_true_high_carbon_with_20pct_opportunity"]:
                    if metric not in frame.columns:
                        continue
                    s = frame[metric].dropna()
                    if s.empty:
                        continue
                    ci_rows.append({
                        "split": split, "review_fraction": frac, "resampling": scheme,
                        "resampling_unit": unit or "row",
                        "n_units": int(pool[unit].nunique()) if unit else len(pool),
                        "metric": metric,
                        "point_estimate": point.get(metric, np.nan),
                        "ci_low": float(s.quantile(0.025)), "ci_high": float(s.quantile(0.975)),
                        "ci_width": float(s.quantile(0.975) - s.quantile(0.025)),
                    })
                print(f"   [ci] {scheme} done", flush=True)

    R.write_table(pd.DataFrame(point_rows), "opportunity_point_estimates.csv")
    R.write_table(pd.DataFrame(ci_rows), "opportunity_clustered_ci.csv")
    R.write_manifest({
        "analysis": "comparable-group opportunity with clustered uncertainty",
        "predictions_source": str(PRED_PATH),
        "min_alternatives": MIN_ALTERNATIVES,
        "bootstrap_repeats": N_BOOT,
        "group_hierarchy": GROUP_ORDER,
        "note": "Point estimates reproduce the reported screen; only the resampling scheme changes.",
    }, "opportunity_manifest.json")


if __name__ == "__main__":
    main()
