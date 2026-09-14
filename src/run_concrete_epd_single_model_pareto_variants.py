"""Single-model Pareto variants for concrete EPD screening.

This script evaluates non-ensemble algorithmic variants for the version-5
concrete EPD study:

1. Pareto-CARM thresholded single model: the fixed CARM-XGB score model with
   its operating threshold selected from an inner training split only.
2. Domain-feature LightGBM single model: one LightGBM classifier trained on
   the CARM domain feature representation.

The script is intentionally conservative. It compares each variant against
existing metric leaders and does not claim universal dominance when metrics
conflict.
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from run_concrete_epd_custom_single_model import (
    CARM_BOOST_FIXED,
    custom_feature_frame,
    fit_predict_candidate,
    make_preprocessor,
)
from concrete_epd_pipeline import (
    MAIN_QUANTILE,
    RESULTS_DIR,
    load_and_prepare,
    make_splits,
    make_target,
)


ROOT = Path(__file__).resolve().parents[1]
V5_CSV = ROOT / "data" / "raw" / "concrete_epd_mendeley_v5.csv"
RANDOM_STATE = 42
SPLITS = ["group_company", "group_epd_source", "temporal_latest20"]
METRICS = [
    "roc_auc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "specificity",
]

OUT_VARIANTS = RESULTS_DIR / "single_model_pareto_variants.csv"
OUT_COMPARISON = RESULTS_DIR / "single_model_pareto_vs_metric_leaders.csv"
OUT_MANIFEST = RESULTS_DIR / "single_model_pareto_manifest.json"


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(specificity_score(y_true, pred)),
    }


def inner_split(
    base: pd.DataFrame,
    y: pd.Series,
    outer_train: np.ndarray,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    outer_train = np.asarray(outer_train)
    if split_name == "group_company":
        groups = base["Company"].fillna("missing").astype(str).iloc[outer_train]
        inner_train_pos, inner_val_pos = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE + 17).split(
                outer_train,
                y.iloc[outer_train],
                groups=groups,
            )
        )
        return outer_train[inner_train_pos], outer_train[inner_val_pos]

    if split_name == "group_epd_source":
        groups = (
            base["EPD Source Link"]
            .fillna(base["Company"].astype(str) + "|" + base["Plant"].astype(str))
            .astype(str)
            .iloc[outer_train]
        )
        inner_train_pos, inner_val_pos = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE + 23).split(
                outer_train,
                y.iloc[outer_train],
                groups=groups,
            )
        )
        return outer_train[inner_train_pos], outer_train[inner_val_pos]

    dates = base["issue_date"].fillna(pd.Timestamp("1900-01-01")).iloc[outer_train]
    cutoff = dates.quantile(0.80)
    return outer_train[np.flatnonzero(dates < cutoff)], outer_train[np.flatnonzero(dates >= cutoff)]


def choose_pareto_threshold(y_val: np.ndarray, proba: np.ndarray) -> float:
    thresholds = np.unique(np.quantile(proba, np.linspace(0.01, 0.99, 301)))
    best: tuple[float, float, float, float, float, float, float] | None = None
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        bal = balanced_accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, zero_division=0)
        prec = precision_score(y_val, pred, zero_division=0)
        rec = recall_score(y_val, pred, zero_division=0)
        spec = specificity_score(y_val, pred)
        utility = 0.52 * bal + 0.38 * f1 + 0.06 * min(rec, 0.90) + 0.04 * prec
        row = (utility, bal, f1, prec, rec, spec, float(threshold))
        if best is None or row > best:
            best = row
    assert best is not None
    return best[-1]


def run_pareto_carm(
    base: pd.DataFrame,
    y: pd.Series,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    model_frame: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in SPLITS:
        outer_train, outer_test = splits[split_name]
        inner_train_idx, inner_val_idx = inner_split(base, y, outer_train, split_name)
        started = time.time()
        val_proba = fit_predict_candidate(
            CARM_BOOST_FIXED,
            model_frame,
            numeric_cols,
            categorical_cols,
            inner_train_idx,
            inner_val_idx,
            y,
        )
        threshold = choose_pareto_threshold(y.iloc[inner_val_idx].to_numpy(), val_proba)
        test_proba = fit_predict_candidate(
            CARM_BOOST_FIXED,
            model_frame,
            numeric_cols,
            categorical_cols,
            outer_train,
            outer_test,
            y,
        )
        rows.append(
            {
                "algorithm": "Pareto-CARM thresholded single model",
                "split": split_name,
                "selected_candidate": CARM_BOOST_FIXED.name,
                "threshold": threshold,
                "threshold_source": "inner_training_split",
                "n_train": int(len(outer_train)),
                "n_test": int(len(outer_test)),
                "test_positive_rate": float(y.iloc[outer_test].mean()),
                "fit_seconds": round(time.time() - started, 3),
                **evaluate(y.iloc[outer_test].to_numpy(), test_proba, threshold),
            }
        )
    return rows


def fit_predict_domain_lgbm(
    model_frame: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y: pd.Series,
) -> np.ndarray:
    from lightgbm import LGBMClassifier

    preprocessor = make_preprocessor(CARM_BOOST_FIXED, numeric_cols, categorical_cols)
    x_train = preprocessor.fit_transform(model_frame.iloc[train_idx])
    x_test = preprocessor.transform(model_frame.iloc[test_idx])
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=63,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        objective="binary",
        device_type="gpu",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        verbose=-1,
    )
    try:
        model.fit(x_train, y.iloc[train_idx].to_numpy())
    except Exception:
        model.set_params(device_type="cpu")
        model.fit(x_train, y.iloc[train_idx].to_numpy())
    return model.predict_proba(x_test)[:, 1]


def run_domain_lgbm(
    y: pd.Series,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    model_frame: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in SPLITS:
        train_idx, test_idx = splits[split_name]
        started = time.time()
        proba = fit_predict_domain_lgbm(
            model_frame,
            numeric_cols,
            categorical_cols,
            train_idx,
            test_idx,
            y,
        )
        rows.append(
            {
                "algorithm": "Domain-feature LightGBM single model",
                "split": split_name,
                "selected_candidate": "LightGBM_CARM_domain_features",
                "threshold": 0.5,
                "threshold_source": "fixed_default",
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_positive_rate": float(y.iloc[test_idx].mean()),
                "fit_seconds": round(time.time() - started, 3),
                **evaluate(y.iloc[test_idx].to_numpy(), proba, 0.5),
            }
        )
    return rows


def compare_to_metric_leaders(results: pd.DataFrame) -> pd.DataFrame:
    benchmark_path = RESULTS_DIR / "final_all_algorithm_metrics.csv"
    benchmark = pd.read_csv(benchmark_path)
    rows: list[dict[str, Any]] = []
    for _, result in results.iterrows():
        split = result["split"]
        split_benchmark = benchmark[benchmark["split"] == split]
        if split_benchmark.empty:
            continue
        for metric in METRICS:
            leader_idx = split_benchmark[metric].idxmax()
            leader = split_benchmark.loc[leader_idx]
            rows.append(
                {
                    "algorithm": result["algorithm"],
                    "split": split,
                    "metric": metric,
                    "value": result[metric],
                    "leader_algorithm": leader["algorithm"],
                    "leader_value": leader[metric],
                    "delta_vs_leader": result[metric] - leader[metric],
                    "wins_metric": bool(result[metric] > leader[metric]),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["all", "pareto-carm", "domain-lgbm"],
        default="all",
        help="Variant to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CONCRETE_EPD_CSV"] = str(V5_CSV)
    started = time.time()

    base, dataset_summary = load_and_prepare()
    y = make_target(base, MAIN_QUANTILE).reset_index(drop=True)
    splits = make_splits(base, y)
    model_frame, numeric_cols, categorical_cols = custom_feature_frame(base)

    rows: list[dict[str, Any]] = []
    if args.variant in {"all", "pareto-carm"}:
        rows.extend(run_pareto_carm(base, y, splits, model_frame, numeric_cols, categorical_cols))
    if args.variant in {"all", "domain-lgbm"}:
        rows.extend(run_domain_lgbm(y, splits, model_frame, numeric_cols, categorical_cols))

    results = pd.DataFrame(rows)
    comparison = compare_to_metric_leaders(results)
    results.to_csv(OUT_VARIANTS, index=False)
    comparison.to_csv(OUT_COMPARISON, index=False)

    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "runtime_seconds": round(time.time() - started, 3),
        "dataset_version": 5,
        "csv_path": str(V5_CSV),
        "target": "top decile of A1-A3 GWP per ksi of compressive strength",
        "target_quantile": MAIN_QUANTILE,
        "not_an_ensemble_of_algorithms": True,
        "variant": args.variant,
        "splits": SPLITS,
        "dataset_rows_after_trim": dataset_summary["model_rows_after_trim"],
        "outputs": [str(OUT_VARIANTS), str(OUT_COMPARISON)],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(results.to_csv(index=False))
    print(comparison.groupby(["algorithm", "split"])["wins_metric"].sum().to_string())


if __name__ == "__main__":
    main()
