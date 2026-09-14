"""Bootstrap uncertainty and review-efficiency metrics for EPD screening.

The main paper claim is operational: a top-k review list captures most
high-carbon concrete EPDs. This script estimates confidence intervals for
top-k capture, precision, lift, AUC/AP, review-efficiency metrics, and the
procurement opportunity estimates.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PREDICTIONS = RESULTS_DIR / "domain_lgbm_predictions.csv"
DEFAULT_SCORE_COLUMN = "domain_lgbm_probability"
OUT_CI = RESULTS_DIR / "uncertainty_analysis.csv"
OUT_DECISION = RESULTS_DIR / "decision_efficiency_summary.csv"
OUT_MANIFEST = RESULTS_DIR / "uncertainty_analysis_manifest.json"


def topk_metrics(y_true: np.ndarray, score: np.ndarray, review_fraction: float) -> dict[str, float]:
    n = len(y_true)
    positives = float(np.sum(y_true))
    n_review = max(1, int(math.ceil(n * review_fraction)))
    selected = np.argsort(score)[-n_review:]
    tp = float(np.sum(y_true[selected]))
    fp = float(n_review - tp)
    base_rate = positives / n if n else np.nan
    precision = tp / n_review if n_review else np.nan
    capture = tp / positives if positives else np.nan
    lift = precision / base_rate if base_rate else np.nan
    random_reviews_for_same_tp = tp / base_rate if base_rate else np.nan
    random_reviews_for_same_tp = min(random_reviews_for_same_tp, float(n)) if pd.notna(random_reviews_for_same_tp) else np.nan
    review_savings_vs_random = random_reviews_for_same_tp - n_review if pd.notna(random_reviews_for_same_tp) else np.nan
    review_savings_fraction_vs_random = (
        review_savings_vs_random / random_reviews_for_same_tp
        if pd.notna(random_reviews_for_same_tp) and random_reviews_for_same_tp > 0
        else np.nan
    )
    reviews_per_true_high_carbon = n_review / tp if tp else np.nan
    random_reviews_per_true_high_carbon = n / positives if positives else np.nan
    out = {
        "n_test": float(n),
        "reviewed_count": float(n_review),
        "true_high_carbon_total": positives,
        "true_high_carbon_captured": tp,
        "false_positive_reviews": fp,
        "base_positive_rate": base_rate,
        "topk_precision": precision,
        "topk_recall_capture": capture,
        "topk_lift": lift,
        "reviews_per_true_high_carbon": reviews_per_true_high_carbon,
        "random_reviews_per_true_high_carbon": random_reviews_per_true_high_carbon,
        "random_reviews_for_same_true_high_carbon_capture": random_reviews_for_same_tp,
        "review_savings_vs_random": review_savings_vs_random,
        "review_savings_fraction_vs_random": review_savings_fraction_vs_random,
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
        out["average_precision"] = float(average_precision_score(y_true, score))
    else:
        out["roc_auc"] = np.nan
        out["average_precision"] = np.nan
    return out


def opportunity_metrics(details: pd.DataFrame) -> dict[str, float]:
    high = details[details["y_true"] == 1].copy()
    high_alt = high[high["has_comparable_alternative_group"].astype(bool)].copy()
    return {
        "captured_high_carbon_with_comparable_group": float(len(high_alt)),
        "comparable_group_coverage_true_high_carbon": float(len(high_alt) / len(high)) if len(high) else np.nan,
        "median_relative_reduction_to_p25_true_high_carbon": float(
            high_alt["relative_reduction_to_p25"].median()
        )
        if len(high_alt)
        else np.nan,
        "median_absolute_reduction_to_p25_true_high_carbon": float(
            high_alt["absolute_reduction_to_p25"].median()
        )
        if len(high_alt)
        else np.nan,
        "share_true_high_carbon_with_10pct_opportunity": float(
            high_alt["has_10pct_reduction_opportunity"].astype(bool).mean()
        )
        if len(high_alt)
        else np.nan,
        "share_true_high_carbon_with_20pct_opportunity": float(
            high_alt["has_20pct_reduction_opportunity"].astype(bool).mean()
        )
        if len(high_alt)
        else np.nan,
    }


def bootstrap_prediction_metrics(
    split_predictions: pd.DataFrame,
    review_fraction: float,
    score_column: str,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], pd.DataFrame]:
    y = split_predictions["y_true"].to_numpy(dtype=int)
    score = split_predictions[score_column].to_numpy(dtype=float)
    point = topk_metrics(y, score, review_fraction)
    rows: list[dict[str, float]] = []
    n = len(y)
    for _ in range(repeats):
        sample = rng.integers(0, n, size=n)
        rows.append(topk_metrics(y[sample], score[sample], review_fraction))
    return point, pd.DataFrame(rows)


def bootstrap_opportunity_metrics(
    details: pd.DataFrame,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], pd.DataFrame]:
    point = opportunity_metrics(details)
    rows: list[dict[str, float]] = []
    n = len(details)
    for _ in range(repeats):
        sample = rng.integers(0, n, size=n)
        rows.append(opportunity_metrics(details.iloc[sample]))
    return point, pd.DataFrame(rows)


def ci_rows(
    point: dict[str, float],
    boot: pd.DataFrame,
    split: str,
    review_fraction: float,
    analysis_type: str,
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, value in point.items():
        series = pd.to_numeric(boot[metric], errors="coerce").dropna() if metric in boot.columns else pd.Series(dtype=float)
        rows.append(
            {
                "split": split,
                "review_fraction": review_fraction,
                "analysis_type": analysis_type,
                "metric": metric,
                "point": value,
                "ci_low": float(series.quantile(0.025)) if len(series) else np.nan,
                "ci_high": float(series.quantile(0.975)) if len(series) else np.nan,
                "bootstrap_repeats": repeats,
                "bootstrap_valid_repeats": int(len(series)),
            }
        )
    return rows


def details_path(split: str, review_fraction: float, output_tag: str = "") -> Path:
    pct = int(round(review_fraction * 100))
    tag = f"{output_tag}_" if output_tag else ""
    return RESULTS_DIR / f"concrete_epd_v5_procurement_opportunity_{tag}{split}_top{pct}_details.csv"


def parse_review_fractions(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_splits(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        default="group_company,group_epd_source,temporal_latest20",
        help="Comma-separated split names.",
    )
    parser.add_argument(
        "--review-fractions",
        default="0.20",
        help="Comma-separated review fractions.",
    )
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--output-tag", default="")
    return parser.parse_args()


def output_paths(output_tag: str) -> tuple[Path, Path, Path]:
    if not output_tag:
        return OUT_CI, OUT_DECISION, OUT_MANIFEST
    return (
        RESULTS_DIR / f"concrete_epd_v5_{output_tag}_uncertainty_analysis.csv",
        RESULTS_DIR / f"concrete_epd_v5_{output_tag}_decision_efficiency_summary.csv",
        RESULTS_DIR / f"concrete_epd_v5_{output_tag}_uncertainty_analysis_manifest.json",
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    rng = np.random.default_rng(args.seed)
    predictions = pd.read_csv(args.predictions)
    if args.score_column not in predictions.columns:
        raise ValueError(f"Score column '{args.score_column}' not found in {args.predictions}")
    out_ci, out_decision, out_manifest = output_paths(args.output_tag)
    all_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    splits = parse_splits(args.splits)
    review_fractions = parse_review_fractions(args.review_fractions)

    for split in splits:
        split_predictions = predictions[predictions["split"] == split].copy()
        if split_predictions.empty:
            continue
        for review_fraction in review_fractions:
            point, boot = bootstrap_prediction_metrics(
                split_predictions,
                review_fraction,
                args.score_column,
                args.repeats,
                rng,
            )
            all_rows.extend(ci_rows(point, boot, split, review_fraction, "topk_prediction", args.repeats))
            decision_rows.append(
                {
                    "split": split,
                    "review_fraction": review_fraction,
                    **point,
                }
            )
            path = details_path(split, review_fraction, args.output_tag)
            if path.exists():
                details = pd.read_csv(path)
                opp_point, opp_boot = bootstrap_opportunity_metrics(details, args.repeats, rng)
                all_rows.extend(
                    ci_rows(opp_point, opp_boot, split, review_fraction, "procurement_opportunity", args.repeats)
                )

    ci = pd.DataFrame(all_rows)
    decision = pd.DataFrame(decision_rows)
    ci.to_csv(out_ci, index=False)
    decision.to_csv(out_decision, index=False)
    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "runtime_seconds": round(time.time() - started, 3),
        "predictions": str(args.predictions),
        "score_column": args.score_column,
        "output_tag": args.output_tag,
        "splits": splits,
        "review_fractions": review_fractions,
        "bootstrap_repeats": args.repeats,
        "seed": args.seed,
        "interpretation": (
            "Prediction metrics are bootstrapped over split-level out-of-sample rows. "
            "Opportunity metrics are bootstrapped over the flagged review list and estimate "
            "uncertainty in the observed opportunity summary, not guaranteed substitution feasibility."
        ),
        "outputs": [str(out_ci), str(out_decision)],
    }
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    key_metrics = ci[
        ci["metric"].isin(
            [
                "topk_recall_capture",
                "topk_precision",
                "review_savings_fraction_vs_random",
                "median_relative_reduction_to_p25_true_high_carbon",
                "share_true_high_carbon_with_10pct_opportunity",
            ]
        )
    ]
    print(key_metrics.to_csv(index=False))


if __name__ == "__main__":
    main()
