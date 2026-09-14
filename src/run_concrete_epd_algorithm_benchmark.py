"""Benchmark concrete EPD screening against multiple ML algorithms.

The main study model is XGBoost. This benchmark compares it with a broad set
of conventional and advanced classifiers under the same target, features, and
holdout splits used in the reviewer-ready concrete EPD study.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
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
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from xgboost import XGBClassifier

from concrete_epd_pipeline import (
    FEATURE_SETS,
    FIGURES_DIR,
    MAIN_QUANTILE,
    RESULTS_DIR,
    FeatureSet,
    build_text_feature,
    feature_columns,
    load_and_prepare,
    make_preprocessor,
    make_splits,
    make_target,
)


RANDOM_STATE = 42
FEATURE_SET = FeatureSet("metadata_plus_lci_sources", True)
BENCHMARK_SPLITS = ["group_company", "temporal_latest20"]
OUTPUT_CSV = RESULTS_DIR / "algorithm_benchmark.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "algorithm_benchmark_summary.csv"
OUTPUT_MANIFEST = RESULTS_DIR / "algorithm_benchmark_manifest.json"
OUTPUT_FIGURE = FIGURES_DIR / "concrete_epd_algorithm_benchmark_auc.png"


def make_feature_matrix(base: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    numeric_cols, categorical_cols, _ = feature_columns()
    frame = base[numeric_cols + categorical_cols].copy()
    frame["text_feature"] = build_text_feature(base, FEATURE_SET)
    return frame, make_preprocessor(numeric_cols, categorical_cols)


def positive_scores(model: Any, x_test: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ranking scores and binary predictions for a fitted classifier."""
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(x_test))[:, 1]
        predictions = (scores >= 0.5).astype(int)
        return scores, predictions
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x_test))
        predictions = np.asarray(model.predict(x_test)).astype(int)
        return scores, predictions
    predictions = np.asarray(model.predict(x_test)).astype(int)
    return predictions.astype(float), predictions


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def metric_row(
    algorithm: str,
    family: str,
    split_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    fit_seconds: float,
    n_train: int,
    n_test: int,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "algorithm": algorithm,
        "family": family,
        "split": split_name,
        "target_quantile": MAIN_QUANTILE,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "test_positive_rate": float(np.mean(y_true)),
        "majority_class_accuracy": float(max(np.mean(y_true), 1 - np.mean(y_true))),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity_score(y_true, predictions)),
        "fit_seconds": round(float(fit_seconds), 3),
        "notes": notes,
    }


def sparse_algorithms(y_train: np.ndarray) -> list[tuple[str, str, Any, str]]:
    pos = int(np.sum(y_train))
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)
    class_weight = "balanced"
    return [
        (
            "Dummy majority baseline",
            "baseline",
            DummyClassifier(strategy="most_frequent"),
            "Sanity baseline, not a SOTA model.",
        ),
        (
            "Complement Naive Bayes",
            "probabilistic linear",
            ComplementNB(alpha=0.2),
            "Fast probabilistic benchmark often used for sparse text-like features.",
        ),
        (
            "SGD logistic regression",
            "linear",
            make_pipeline(
                MaxAbsScaler(),
                SGDClassifier(
                    loss="log_loss",
                    class_weight=class_weight,
                    alpha=1e-5,
                    max_iter=3000,
                    tol=1e-4,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            "Linear large-scale logistic classifier.",
        ),
        (
            "Linear SVM SGD",
            "linear margin",
            make_pipeline(
                MaxAbsScaler(),
                SGDClassifier(
                    loss="hinge",
                    class_weight=class_weight,
                    alpha=1e-5,
                    max_iter=3000,
                    tol=1e-4,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            "Linear support-vector benchmark using SGD.",
        ),
        (
            "Passive-Aggressive",
            "linear margin",
            make_pipeline(
                MaxAbsScaler(),
                PassiveAggressiveClassifier(
                    class_weight=class_weight,
                    max_iter=3000,
                    tol=1e-4,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            "Online large-margin classifier.",
        ),
        (
            "Ridge classifier",
            "linear",
            make_pipeline(MaxAbsScaler(), RidgeClassifier(class_weight=class_weight)),
            "Regularised linear classifier.",
        ),
        (
            "Random Forest",
            "bagging ensemble",
            RandomForestClassifier(
                n_estimators=220,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "Nonlinear bagging ensemble.",
        ),
        (
            "Extra Trees",
            "bagging ensemble",
            ExtraTreesClassifier(
                n_estimators=260,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "Extremely randomised tree ensemble.",
        ),
        (
            "XGBoost GPU",
            "gradient boosting",
            XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.035,
                subsample=0.90,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                device="cuda",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                reg_lambda=1.5,
            ),
            "Main study model.",
        ),
    ]


def optional_sparse_algorithms(y_train: np.ndarray) -> list[tuple[str, str, Any, str]]:
    algorithms: list[tuple[str, str, Any, str]] = []
    pos = int(np.sum(y_train))
    neg = int(len(y_train) - pos)
    try:
        from lightgbm import LGBMClassifier

        algorithms.append(
            (
                "LightGBM GPU",
                "gradient boosting",
                LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.035,
                    num_leaves=63,
                    subsample=0.90,
                    colsample_bytree=0.85,
                    objective="binary",
                    device_type="gpu",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced",
                    verbose=-1,
                ),
                "GPU gradient-boosting benchmark.",
            )
        )
    except Exception as exc:
        algorithms.append(("LightGBM GPU unavailable", "skipped", None, str(exc)))
    return algorithms


def dense_latent_algorithms(y_train: np.ndarray) -> list[tuple[str, str, Any, str]]:
    pos = int(np.sum(y_train))
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)
    algorithms: list[tuple[str, str, Any, str]] = [
        (
            "HistGradientBoosting SVD",
            "gradient boosting",
            HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.05,
                max_leaf_nodes=31,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "Histogram gradient boosting on 160-dimensional latent features.",
        ),
        (
            "MLP SVD",
            "neural network",
            MLPClassifier(
                hidden_layer_sizes=(96, 32),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=90,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
            "Shallow neural-network benchmark on latent features.",
        ),
    ]
    try:
        from catboost import CatBoostClassifier

        algorithms.append(
            (
                "CatBoost CPU SVD",
                "gradient boosting",
                CatBoostClassifier(
                    iterations=260,
                    depth=6,
                    learning_rate=0.04,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    class_weights=[1.0, float(scale_pos_weight)],
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                ),
                "CatBoost benchmark on latent features; GPU was avoided due to 4 GB VRAM.",
            )
        )
    except Exception as exc:
        algorithms.append(("CatBoost unavailable", "skipped", None, str(exc)))
    return algorithms


def fit_and_score(
    algorithm: str,
    family: str,
    model: Any,
    notes: str,
    split_name: str,
    x_train: Any,
    x_test: Any,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    if model is None:
        return {
            "algorithm": algorithm,
            "family": family,
            "split": split_name,
            "target_quantile": MAIN_QUANTILE,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "status": "skipped",
            "notes": notes,
        }
    started = time.time()
    model.fit(x_train, y_train)
    fit_seconds = time.time() - started
    scores, predictions = positive_scores(model, x_test)
    row = metric_row(
        algorithm,
        family,
        split_name,
        y_test,
        scores,
        predictions,
        fit_seconds,
        len(y_train),
        len(y_test),
        notes,
    )
    row["status"] = "completed"
    return row


def run_benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    base, dataset_summary = load_and_prepare()
    y = make_target(base, MAIN_QUANTILE)
    model_frame, preprocessor = make_feature_matrix(base)
    splits = make_splits(base, y)
    rows: list[dict[str, Any]] = []

    for split_name in BENCHMARK_SPLITS:
        train_idx, test_idx = splits[split_name]
        y_train = y.iloc[train_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()
        print(f"\nPreparing split: {split_name}")
        started = time.time()
        x_train = preprocessor.fit_transform(model_frame.iloc[train_idx])
        x_test = preprocessor.transform(model_frame.iloc[test_idx])
        print(
            f"{split_name} sparse matrix train={x_train.shape} test={x_test.shape} "
            f"prep={time.time() - started:.1f}s"
        )

        for algorithm, family, model, notes in (
            sparse_algorithms(y_train) + optional_sparse_algorithms(y_train)
        ):
            try:
                row = fit_and_score(
                    algorithm,
                    family,
                    model,
                    notes,
                    split_name,
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                )
                rows.append(row)
                print(
                    f"{algorithm}: AUC={row.get('roc_auc', np.nan):.4f} "
                    f"ACC={row.get('accuracy', np.nan):.4f} "
                    f"BAL={row.get('balanced_accuracy', np.nan):.4f}"
                )
            except Exception as exc:
                rows.append(
                    {
                        "algorithm": algorithm,
                        "family": family,
                        "split": split_name,
                        "target_quantile": MAIN_QUANTILE,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                        "status": "failed",
                        "notes": f"{notes} Failure: {type(exc).__name__}: {exc}",
                    }
                )
                print(f"{algorithm}: failed with {type(exc).__name__}: {exc}")

        svd = make_pipeline(
            TruncatedSVD(n_components=160, random_state=RANDOM_STATE),
            StandardScaler(),
        )
        started = time.time()
        x_train_dense = svd.fit_transform(x_train)
        x_test_dense = svd.transform(x_test)
        print(
            f"{split_name} dense latent matrix train={x_train_dense.shape} "
            f"test={x_test_dense.shape} svd={time.time() - started:.1f}s"
        )
        for algorithm, family, model, notes in dense_latent_algorithms(y_train):
            try:
                row = fit_and_score(
                    algorithm,
                    family,
                    model,
                    notes,
                    split_name,
                    x_train_dense,
                    x_test_dense,
                    y_train,
                    y_test,
                )
                rows.append(row)
                print(
                    f"{algorithm}: AUC={row.get('roc_auc', np.nan):.4f} "
                    f"ACC={row.get('accuracy', np.nan):.4f} "
                    f"BAL={row.get('balanced_accuracy', np.nan):.4f}"
                )
            except Exception as exc:
                rows.append(
                    {
                        "algorithm": algorithm,
                        "family": family,
                        "split": split_name,
                        "target_quantile": MAIN_QUANTILE,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                        "status": "failed",
                        "notes": f"{notes} Failure: {type(exc).__name__}: {exc}",
                    }
                )
                print(f"{algorithm}: failed with {type(exc).__name__}: {exc}")

    results = pd.DataFrame(rows)
    completed = results[results["status"] == "completed"].copy()
    summary = (
        completed.pivot_table(
            index=["algorithm", "family"],
            columns="split",
            values=["roc_auc", "accuracy", "balanced_accuracy", "f1", "average_precision"],
            aggfunc="first",
        )
        .sort_values(("roc_auc", "temporal_latest20"), ascending=False)
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return results, summary


def plot_summary(summary: pd.DataFrame) -> None:
    auc_cols = ["roc_auc_group_company", "roc_auc_temporal_latest20"]
    if not all(col in summary.columns for col in auc_cols):
        return
    plot_data = summary[["algorithm"] + auc_cols].copy()
    plot_data = plot_data.sort_values("roc_auc_temporal_latest20", ascending=True)
    y_pos = np.arange(len(plot_data))
    plt.figure(figsize=(8.5, max(5.2, 0.38 * len(plot_data))), dpi=150)
    plt.barh(y_pos - 0.18, plot_data["roc_auc_group_company"], height=0.34, label="Unseen company")
    plt.barh(y_pos + 0.18, plot_data["roc_auc_temporal_latest20"], height=0.34, label="Temporal latest 20%")
    plt.axvline(0.90, color="0.25", linestyle="--", linewidth=1)
    plt.yticks(y_pos, plot_data["algorithm"])
    plt.xlabel("ROC-AUC")
    plt.xlim(0.0, 1.0)
    plt.title("Algorithm benchmark for high-carbon concrete risk screening")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results, summary = run_benchmark()
    results.to_csv(OUTPUT_CSV, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    plot_summary(summary)
    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "runtime_seconds": round(time.time() - started, 3),
        "feature_set": FEATURE_SET.__dict__,
        "target": "top decile of A1-A3 GWP per ksi of compressive strength",
        "target_quantile": MAIN_QUANTILE,
        "splits": BENCHMARK_SPLITS,
        "completed_algorithm_count": int(
            results.loc[results["status"] == "completed", "algorithm"].nunique()
        ),
        "completed_algorithms": sorted(
            results.loc[results["status"] == "completed", "algorithm"].unique().tolist()
        ),
        "known_feature_sets_in_main_study": [feature_set.__dict__ for feature_set in FEATURE_SETS],
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
