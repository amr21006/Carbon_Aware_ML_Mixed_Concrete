"""Procurement opportunity analysis for high-carbon concrete EPD screening.

The predictive model identifies mixes that should be reviewed. This script
adds a construction-management decision layer: for flagged mixes, it estimates
whether lower-carbon alternatives exist within a comparable specification
neighborhood. The analysis is retrospective and should be interpreted as an
opportunity screen, not proof that any specific substitution is feasible.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from concrete_epd_pipeline import MAIN_QUANTILE, RESULTS_DIR, load_and_prepare, make_target


ROOT = Path(__file__).resolve().parents[1]
V5_CSV = ROOT / "data" / "raw" / "concrete_epd_mendeley_v5.csv"
PREDICTIONS = RESULTS_DIR / "domain_lgbm_predictions.csv"
DEFAULT_SCORE_COLUMN = "domain_lgbm_probability"
OUT_DETAILS = RESULTS_DIR / "procurement_opportunity_details.csv"
OUT_SUMMARY = RESULTS_DIR / "procurement_opportunity_summary.csv"
OUT_BY_GROUP = RESULTS_DIR / "procurement_opportunity_by_group.csv"
OUT_MANIFEST = RESULTS_DIR / "procurement_opportunity_manifest.json"


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": "missing", "nan": "missing", "none": "missing", "false": "missing", "true": "missing"})
    )


def application_family(base: pd.DataFrame) -> pd.Series:
    flags = {
        "structural": "Application Category: Structural",
        "hardscape": "Application Category: Hardscape",
        "paving": "Application Category: Paving",
        "infrastructure": "Application Category: Infrastructure",
        "filler": "Application Category: Filler",
    }
    out = pd.Series("other_or_unspecified", index=base.index, dtype="object")
    for label, col in flags.items():
        if col in base.columns:
            out = out.mask(pd.to_numeric(base[col], errors="coerce").fillna(0) > 0, label)
    primary = normalize_text(base.get("Primary Application?", pd.Series("", index=base.index)))
    other = normalize_text(base.get("Other Application Label?", pd.Series("", index=base.index)))
    out = out.mask((out == "other_or_unspecified") & (primary != "missing"), primary)
    out = out.mask((out == "other_or_unspecified") & (other != "missing"), other)
    return out


def curing_family(base: pd.DataFrame) -> pd.Series:
    curing = pd.to_numeric(base["curing_days"], errors="coerce")
    return pd.cut(
        curing,
        bins=[-np.inf, 7, 28, 56, np.inf],
        labels=["early", "standard_28", "extended_56", "long"],
    ).astype(str).replace({"nan": "missing"})


def add_context_columns(base: pd.DataFrame) -> pd.DataFrame:
    frame = base.copy()
    frame["application_family"] = application_family(base)
    frame["curing_family"] = curing_family(base)
    frame["region_family"] = normalize_text(frame["U.S. Region of Plant"])
    frame["state_family"] = normalize_text(frame["Plant Location - State"])
    frame["strength_family"] = frame["strength_bin_500"].astype(int).astype(str)
    frame["strict_group"] = (
        frame["region_family"]
        + "|"
        + frame["application_family"]
        + "|"
        + frame["strength_family"]
        + "|"
        + frame["curing_family"]
    )
    frame["regional_strength_group"] = frame["region_family"] + "|" + frame["strength_family"]
    frame["national_spec_group"] = (
        frame["application_family"] + "|" + frame["strength_family"] + "|" + frame["curing_family"]
    )
    return frame


def group_reference_stats(pool: pd.DataFrame, group_col: str, min_alternatives: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, g in pool.groupby(group_col, dropna=False):
        if len(g) < min_alternatives:
            continue
        rows.append(
            {
                group_col: group,
                f"{group_col}_n": int(len(g)),
                f"{group_col}_median_gwp_per_ksi": float(g["gwp_per_ksi"].median()),
                f"{group_col}_p25_gwp_per_ksi": float(g["gwp_per_ksi"].quantile(0.25)),
                f"{group_col}_p10_gwp_per_ksi": float(g["gwp_per_ksi"].quantile(0.10)),
                f"{group_col}_min_gwp_per_ksi": float(g["gwp_per_ksi"].min()),
            }
        )
    return pd.DataFrame(rows)


def attach_reference_stats(flagged: pd.DataFrame, pool: pd.DataFrame, min_alternatives: int) -> pd.DataFrame:
    out = flagged.copy()
    reference_order = ["strict_group", "national_spec_group", "regional_strength_group"]
    for group_col in reference_order:
        refs = group_reference_stats(pool, group_col, min_alternatives)
        if refs.empty:
            continue
        out = out.merge(refs, on=group_col, how="left")

    chosen_rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        chosen: dict[str, Any] = {
            "reference_group_type": "none",
            "reference_group_n": np.nan,
            "reference_median_gwp_per_ksi": np.nan,
            "reference_p25_gwp_per_ksi": np.nan,
            "reference_p10_gwp_per_ksi": np.nan,
            "reference_min_gwp_per_ksi": np.nan,
        }
        for group_col in reference_order:
            n_col = f"{group_col}_n"
            if n_col in out.columns and pd.notna(row.get(n_col)):
                chosen = {
                    "reference_group_type": group_col,
                    "reference_group_n": row[n_col],
                    "reference_median_gwp_per_ksi": row[f"{group_col}_median_gwp_per_ksi"],
                    "reference_p25_gwp_per_ksi": row[f"{group_col}_p25_gwp_per_ksi"],
                    "reference_p10_gwp_per_ksi": row[f"{group_col}_p10_gwp_per_ksi"],
                    "reference_min_gwp_per_ksi": row[f"{group_col}_min_gwp_per_ksi"],
                }
                break
        chosen_rows.append(chosen)
    chosen_df = pd.DataFrame(chosen_rows, index=out.index)
    out = pd.concat([out, chosen_df], axis=1)
    out["absolute_reduction_to_p25"] = (
        out["gwp_per_ksi"] - out["reference_p25_gwp_per_ksi"]
    ).clip(lower=0)
    out["relative_reduction_to_p25"] = out["absolute_reduction_to_p25"] / out["gwp_per_ksi"]
    out["absolute_reduction_to_p10"] = (
        out["gwp_per_ksi"] - out["reference_p10_gwp_per_ksi"]
    ).clip(lower=0)
    out["relative_reduction_to_p10"] = out["absolute_reduction_to_p10"] / out["gwp_per_ksi"]
    out["has_comparable_alternative_group"] = out["reference_group_type"] != "none"
    out["has_10pct_reduction_opportunity"] = out["relative_reduction_to_p25"] >= 0.10
    out["has_20pct_reduction_opportunity"] = out["relative_reduction_to_p25"] >= 0.20
    return out


def summarize(details: pd.DataFrame, split: str, review_fraction: float) -> dict[str, Any]:
    y_true = details["y_true"].to_numpy()
    positives = float(y_true.sum())
    captured = float(details["y_true"].sum())
    alt = details[details["has_comparable_alternative_group"]]
    true_high = details[details["y_true"] == 1]
    true_high_alt = true_high[true_high["has_comparable_alternative_group"]]
    return {
        "split": split,
        "review_fraction": review_fraction,
        "flagged_count": int(len(details)),
        "flagged_true_high_carbon_count": int(captured),
        "flagged_positive_rate": float(details["y_true"].mean()) if len(details) else np.nan,
        "base_positive_rate_in_flagged_pool": float(details["base_positive_rate"].iloc[0]),
        "comparable_group_coverage_all_flagged": float(len(alt) / len(details)) if len(details) else np.nan,
        "comparable_group_coverage_true_high_carbon": (
            float(len(true_high_alt) / len(true_high)) if len(true_high) else np.nan
        ),
        "median_current_gwp_per_ksi_all_flagged": float(details["gwp_per_ksi"].median()),
        "median_current_gwp_per_ksi_true_high_carbon": float(true_high["gwp_per_ksi"].median())
        if len(true_high)
        else np.nan,
        "median_reference_p25_gwp_per_ksi_true_high_carbon": float(
            true_high_alt["reference_p25_gwp_per_ksi"].median()
        )
        if len(true_high_alt)
        else np.nan,
        "median_absolute_reduction_to_p25_true_high_carbon": float(
            true_high_alt["absolute_reduction_to_p25"].median()
        )
        if len(true_high_alt)
        else np.nan,
        "median_relative_reduction_to_p25_true_high_carbon": float(
            true_high_alt["relative_reduction_to_p25"].median()
        )
        if len(true_high_alt)
        else np.nan,
        "share_true_high_carbon_with_10pct_opportunity": float(
            true_high_alt["has_10pct_reduction_opportunity"].mean()
        )
        if len(true_high_alt)
        else np.nan,
        "share_true_high_carbon_with_20pct_opportunity": float(
            true_high_alt["has_20pct_reduction_opportunity"].mean()
        )
        if len(true_high_alt)
        else np.nan,
        "total_absolute_reduction_to_p25_true_high_carbon": float(
            true_high_alt["absolute_reduction_to_p25"].sum()
        )
        if len(true_high_alt)
        else np.nan,
        "note": "Retrospective opportunity screen within comparable EPD groups; not a guaranteed substitution estimate.",
    }


def by_group(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, g in details.groupby("application_family", dropna=False):
        high = g[g["y_true"] == 1]
        high_alt = high[high["has_comparable_alternative_group"]]
        rows.append(
            {
                "application_family": group,
                "flagged_count": int(len(g)),
                "flagged_true_high_carbon_count": int(len(high)),
                "median_current_gwp_per_ksi_true_high_carbon": float(high["gwp_per_ksi"].median())
                if len(high)
                else np.nan,
                "median_relative_reduction_to_p25_true_high_carbon": float(
                    high_alt["relative_reduction_to_p25"].median()
                )
                if len(high_alt)
                else np.nan,
                "share_true_high_carbon_with_10pct_opportunity": float(
                    high_alt["has_10pct_reduction_opportunity"].mean()
                )
                if len(high_alt)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("flagged_true_high_carbon_count", ascending=False)


def run(
    split: str,
    review_fraction: float,
    min_alternatives: int,
    predictions_path: Path,
    score_column: str,
    model_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    os.environ["CONCRETE_EPD_CSV"] = str(V5_CSV)
    base, dataset_summary = load_and_prepare()
    base = add_context_columns(base)
    y = make_target(base, MAIN_QUANTILE).reset_index(drop=True)
    predictions = pd.read_csv(predictions_path)
    pred = predictions[predictions["split"] == split].copy().reset_index(drop=True)
    if score_column not in pred.columns:
        raise ValueError(f"Score column '{score_column}' not found in {predictions_path}")
    n_select = max(1, int(np.ceil(len(pred) * review_fraction)))
    flagged_order = np.argsort(pred[score_column].to_numpy())[-n_select:][::-1]
    flagged_pred = pred.iloc[flagged_order].copy()
    pool = base.iloc[pred["row_index"].to_numpy()].copy()
    pool["y_true"] = pred["y_true"].to_numpy()
    pool["model_probability"] = pred[score_column].to_numpy()
    pool["base_positive_rate"] = float(pool["y_true"].mean())

    flagged = base.iloc[flagged_pred["row_index"].to_numpy()].copy()
    flagged["row_index"] = flagged_pred["row_index"].to_numpy()
    flagged["y_true"] = flagged_pred["y_true"].to_numpy()
    flagged["model_probability"] = flagged_pred[score_column].to_numpy()
    flagged["base_positive_rate"] = float(pool["y_true"].mean())
    details = attach_reference_stats(flagged, pool, min_alternatives)
    keep_cols = [
        "row_index",
        "y_true",
        "model_probability",
        "base_positive_rate",
        "Company",
        "Plant",
        "Plant Location - State",
        "U.S. Region of Plant",
        "application_family",
        "strength_psi",
        "strength_bin_500",
        "curing_days",
        "curing_family",
        "gwp_per_ksi",
        "reference_group_type",
        "reference_group_n",
        "reference_p25_gwp_per_ksi",
        "reference_p10_gwp_per_ksi",
        "absolute_reduction_to_p25",
        "relative_reduction_to_p25",
        "absolute_reduction_to_p10",
        "relative_reduction_to_p10",
        "has_comparable_alternative_group",
        "has_10pct_reduction_opportunity",
        "has_20pct_reduction_opportunity",
    ]
    details = details[keep_cols]
    summary = pd.DataFrame([summarize(details, split, review_fraction)])
    grouped = by_group(details)
    manifest = {
        "dataset_version": 5,
        "csv_path": str(V5_CSV),
        "predictions": str(predictions_path),
        "score_column": score_column,
        "model_label": model_label,
        "split": split,
        "review_fraction": review_fraction,
        "min_alternatives_per_reference_group": min_alternatives,
        "target": "top decile of A1-A3 GWP per ksi of compressive strength",
        "target_quantile": MAIN_QUANTILE,
        "model_rows_after_trim": dataset_summary["model_rows_after_trim"],
        "interpretation": "Retrospective procurement opportunity screen, not guaranteed substitution feasibility.",
    }
    return details, summary, grouped, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="temporal_latest20")
    parser.add_argument("--review-fraction", type=float, default=0.20)
    parser.add_argument("--min-alternatives", type=int, default=5)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--model-label", default="Domain-feature LightGBM single model")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--no-generic-aliases", action="store_true")
    return parser.parse_args()


def output_paths(split: str, review_fraction: float, output_tag: str = "") -> tuple[Path, Path, Path, Path]:
    pct = int(round(review_fraction * 100))
    tag = f"{output_tag}_" if output_tag else ""
    stem = f"concrete_epd_v5_procurement_opportunity_{tag}{split}_top{pct}"
    return (
        RESULTS_DIR / f"{stem}_details.csv",
        RESULTS_DIR / f"{stem}_summary.csv",
        RESULTS_DIR / f"{stem}_by_group.csv",
        RESULTS_DIR / f"{stem}_manifest.json",
    )


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    details, summary, grouped, manifest = run(
        args.split,
        args.review_fraction,
        args.min_alternatives,
        args.predictions,
        args.score_column,
        args.model_label,
    )
    out_details, out_summary, out_by_group, out_manifest = output_paths(
        args.split,
        args.review_fraction,
        args.output_tag,
    )
    details.to_csv(out_details, index=False)
    summary.to_csv(out_summary, index=False)
    grouped.to_csv(out_by_group, index=False)
    if not args.no_generic_aliases:
        details.to_csv(OUT_DETAILS, index=False)
        summary.to_csv(OUT_SUMMARY, index=False)
        grouped.to_csv(OUT_BY_GROUP, index=False)
    manifest["created_utc"] = pd.Timestamp.utcnow().isoformat()
    manifest["runtime_seconds"] = round(time.time() - started, 3)
    manifest["outputs"] = [str(out_details), str(out_summary), str(out_by_group)]
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_csv(index=False))
    print(grouped.head(10).to_csv(index=False))


if __name__ == "__main__":
    main()
