"""Latest-dataset validation for the concrete EPD study.

This script validates the SOTA XGBoost baseline and the proposed single-model
CARM-Boost algorithm on Mendeley dataset version 5. It does not overwrite the
version-2 result files.
"""

from __future__ import annotations

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

from run_concrete_epd_custom_single_model import (
    CARM_BOOST_FIXED,
    custom_feature_frame,
    fit_predict_candidate,
)
from run_concrete_epd_model import (
    MAIN_QUANTILE,
    RAW_DIR,
    RESULTS_DIR,
    FeatureSet,
    build_text_feature,
    feature_columns,
    load_and_prepare,
    make_model,
    make_preprocessor,
    make_splits,
    make_target,
)


ROOT = Path(__file__).resolve().parents[1]
V5_CSV = ROOT / "data" / "raw" / "concrete_epd_mendeley_v5.csv"
OUTPUT_RESULTS = RESULTS_DIR / "concrete_epd_v5_validation_results.csv"
OUTPUT_PREDICTIONS = RESULTS_DIR / "concrete_epd_v5_validation_predictions.csv"
OUTPUT_BOOTSTRAP = RESULTS_DIR / "concrete_epd_v5_paired_bootstrap.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "concrete_epd_v5_dataset_summary.csv"
OUTPUT_MANIFEST = RESULTS_DIR / "concrete_epd_v5_validation_manifest.json"

RANDOM_STATE = 42
BOOTSTRAP_REPEATS = 500
SPLITS = ["group_company", "group_epd_source", "temporal_latest20"]


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
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


def fit_xgboost_sota(
    base: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    numeric_cols, categorical_cols, _ = feature_columns()
    frame = base[numeric_cols + categorical_cols].copy()
    frame["text_feature"] = build_text_feature(base, FeatureSet("metadata_plus_lci_sources", True))
    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    x_train = preprocessor.fit_transform(frame.iloc[train_idx])
    x_test = preprocessor.transform(frame.iloc[test_idx])
    model = make_model(y.iloc[train_idx])
    model.fit(x_train, y.iloc[train_idx])
    return model.predict_proba(x_test)[:, 1]


def paired_bootstrap(
    y_true: np.ndarray,
    carm_proba: np.ndarray,
    xgb_proba: np.ndarray,
    repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    rows: list[dict[str, float]] = []
    for _ in range(repeats):
        sample = rng.integers(0, n, size=n)
        ys = y_true[sample]
        if len(np.unique(ys)) < 2:
            continue
        carm = carm_proba[sample]
        xgb = xgb_proba[sample]
        rows.append(
            {
                "auc_delta": roc_auc_score(ys, carm) - roc_auc_score(ys, xgb),
                "ap_delta": average_precision_score(ys, carm)
                - average_precision_score(ys, xgb),
                "accuracy_delta": accuracy_score(ys, carm >= 0.5)
                - accuracy_score(ys, xgb >= 0.5),
                "balanced_accuracy_delta": balanced_accuracy_score(ys, carm >= 0.5)
                - balanced_accuracy_score(ys, xgb >= 0.5),
                "f1_delta": f1_score(ys, carm >= 0.5, zero_division=0)
                - f1_score(ys, xgb >= 0.5, zero_division=0),
            }
        )
    boot = pd.DataFrame(rows)
    out: dict[str, float] = {}
    for col in boot.columns:
        out[f"{col}_mean"] = float(boot[col].mean())
        out[f"{col}_ci_low"] = float(boot[col].quantile(0.025))
        out[f"{col}_ci_high"] = float(boot[col].quantile(0.975))
        out[f"{col}_p_gt_0"] = float((boot[col] > 0).mean())
    return out


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    os.environ["CONCRETE_EPD_CSV"] = str(V5_CSV)
    base, dataset_summary = load_and_prepare()
    y = make_target(base, MAIN_QUANTILE).reset_index(drop=True)
    splits = make_splits(base, y)
    carm_frame, carm_numeric_cols, carm_categorical_cols = custom_feature_frame(base)

    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for split_name in SPLITS:
        train_idx, test_idx = splits[split_name]
        y_test = y.iloc[test_idx].to_numpy()
        print(f"\nRunning v5 split: {split_name}")

        started = time.time()
        xgb_proba = fit_xgboost_sota(base, y, train_idx, test_idx)
        xgb_metrics = evaluate(y_test, xgb_proba)
        result_rows.append(
            {
                "dataset_version": 5,
                "algorithm": "XGBoost GPU SOTA baseline",
                "split": split_name,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_positive_rate": float(y_test.mean()),
                "fit_seconds": round(time.time() - started, 3),
                **xgb_metrics,
            }
        )
        print(
            f"XGBoost v5 | {split_name} | AUC={xgb_metrics['roc_auc']:.4f} "
            f"ACC={xgb_metrics['accuracy']:.4f} BAL={xgb_metrics['balanced_accuracy']:.4f}"
        )

        started = time.time()
        carm_proba = fit_predict_candidate(
            CARM_BOOST_FIXED,
            carm_frame,
            carm_numeric_cols,
            carm_categorical_cols,
            train_idx,
            test_idx,
            y,
        )
        carm_metrics = evaluate(y_test, carm_proba)
        result_rows.append(
            {
                "dataset_version": 5,
                "algorithm": "CARM-Boost fixed single model",
                "split": split_name,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_positive_rate": float(y_test.mean()),
                "fit_seconds": round(time.time() - started, 3),
                **carm_metrics,
            }
        )
        print(
            f"CARM-Boost v5 | {split_name} | AUC={carm_metrics['roc_auc']:.4f} "
            f"ACC={carm_metrics['accuracy']:.4f} BAL={carm_metrics['balanced_accuracy']:.4f}"
        )

        boot = paired_bootstrap(y_test, carm_proba, xgb_proba)
        boot.update(
            {
                "dataset_version": 5,
                "split": split_name,
                "carm_auc": carm_metrics["roc_auc"],
                "xgb_auc": xgb_metrics["roc_auc"],
                "auc_delta": carm_metrics["roc_auc"] - xgb_metrics["roc_auc"],
            }
        )
        bootstrap_rows.append(boot)

        prediction_rows.append(
            pd.DataFrame(
                {
                    "dataset_version": 5,
                    "split": split_name,
                    "row_index": test_idx,
                    "y_true": y_test,
                    "xgb_probability": xgb_proba,
                    "carm_probability": carm_proba,
                    "gwp_per_ksi": base["gwp_per_ksi"].iloc[test_idx].to_numpy(),
                    "company": base["Company"].iloc[test_idx].to_numpy(),
                    "plant": base["Plant"].iloc[test_idx].to_numpy(),
                }
            )
        )

    return (
        pd.DataFrame(result_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(bootstrap_rows),
        dataset_summary,
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results, predictions, bootstrap, dataset_summary = run()
    results.to_csv(OUTPUT_RESULTS, index=False)
    predictions.to_csv(OUTPUT_PREDICTIONS, index=False)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)
    pd.DataFrame(
        [{"item": key, "value": value} for key, value in dataset_summary.items() if key != "outcome_columns_excluded"]
    ).to_csv(OUTPUT_SUMMARY, index=False)
    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "runtime_seconds": round(time.time() - started, 3),
        "dataset": {
            "mendeley_dataset": "r4jgxk2mhn",
            "version": 5,
            "csv_path": str(V5_CSV),
        },
        "target": "top decile of A1-A3 GWP per ksi of compressive strength",
        "target_quantile": MAIN_QUANTILE,
        "splits": SPLITS,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "algorithms": ["XGBoost GPU SOTA baseline", "CARM-Boost fixed single model"],
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
