"""Single-model custom algorithm for concrete EPD high-carbon screening.

This script avoids stacked/blended ensembles. It proposes CARM-Boost
(Carbon-Aware Risk Modelling Boost), a single cost-sensitive boosted-tree
classifier with domain-specific procurement features and training-fold-only
hyperparameter selection.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from run_concrete_epd_model import (
    MAIN_QUANTILE,
    RESULTS_DIR,
    FeatureSet,
    build_text_feature,
    feature_columns,
    load_and_prepare,
    make_splits,
    make_target,
)


RANDOM_STATE = 42
FEATURE_SET = FeatureSet("metadata_plus_lci_sources", True)
SPLITS = ["group_company", "temporal_latest20"]
OUT_RESULTS = RESULTS_DIR / "concrete_epd_custom_single_model_results.csv"
OUT_SELECTION = RESULTS_DIR / "concrete_epd_custom_single_model_selection.csv"
OUT_COMPARISON = RESULTS_DIR / "concrete_epd_custom_vs_sota.csv"
OUT_MANIFEST = RESULTS_DIR / "concrete_epd_custom_single_model_manifest.json"
OUT_FIXED_RESULTS = RESULTS_DIR / "concrete_epd_carm_boost_fixed_results.csv"
OUT_FIXED_COMPARISON = RESULTS_DIR / "concrete_epd_carm_boost_fixed_vs_sota.csv"
OUT_FIXED_MANIFEST = RESULTS_DIR / "concrete_epd_carm_boost_fixed_manifest.json"


@dataclass(frozen=True)
class Candidate:
    name: str
    max_features: int
    min_df: int
    ngram_max: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    gamma: float


CANDIDATES = [
    Candidate("CARM_Boost_A", 12000, 2, 2, 650, 4, 0.025, 2.0, 0.88, 0.88, 0.05, 2.0, 0.0),
    Candidate("CARM_Boost_B", 12000, 2, 3, 700, 4, 0.022, 3.0, 0.90, 0.85, 0.10, 2.5, 0.05),
    Candidate("CARM_Boost_C", 16000, 2, 2, 560, 5, 0.030, 4.0, 0.88, 0.82, 0.05, 2.0, 0.10),
    Candidate("CARM_Boost_D", 16000, 3, 3, 850, 3, 0.020, 2.0, 0.90, 0.90, 0.02, 1.8, 0.0),
    Candidate("CARM_Boost_E", 10000, 2, 2, 500, 5, 0.035, 1.0, 0.92, 0.88, 0.00, 1.5, 0.0),
]

CARM_BOOST_FIXED = Candidate(
    "CARM_Boost_fixed_domain_10k",
    10000,
    3,
    2,
    450,
    5,
    0.032,
    1.0,
    0.90,
    0.88,
    0.0,
    1.5,
    0.0,
)


def add_domain_features(base: pd.DataFrame) -> pd.DataFrame:
    frame = base.copy()
    strength = frame["strength_psi"].astype(float).clip(lower=1)
    curing = frame["curing_days"].astype(float).fillna(frame["curing_days"].median()).clip(lower=1)
    frame["log_strength_psi"] = np.log1p(strength)
    frame["sqrt_strength_psi"] = np.sqrt(strength)
    frame["strength_per_curing_day"] = strength / curing
    frame["is_high_strength"] = (strength >= 5000).astype(int)
    frame["is_low_strength"] = (strength <= 2500).astype(int)
    scm_cols = [
        "has_fly_ash",
        "has_slag",
        "has_silica_fume",
        "has_limestone_cement",
        "has_carbon_cure",
        "has_recycled",
    ]
    frame["scm_indicator_count"] = frame[scm_cols].sum(axis=1)
    frame["has_multiple_scm"] = (frame["scm_indicator_count"] >= 2).astype(int)
    frame["region_strength_bin"] = (
        frame["U.S. Region of Plant"].fillna("missing").astype(str)
        + "|"
        + frame["strength_bin_500"].astype(str)
    )
    frame["state_strength_bin"] = (
        frame["Plant Location - State"].fillna("missing").astype(str)
        + "|"
        + frame["strength_bin_500"].astype(str)
    )
    return frame


def custom_feature_frame(base: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric_cols, categorical_cols, _ = feature_columns()
    numeric_cols = numeric_cols + [
        "log_strength_psi",
        "sqrt_strength_psi",
        "strength_per_curing_day",
        "is_high_strength",
        "is_low_strength",
        "scm_indicator_count",
        "has_multiple_scm",
    ]
    categorical_cols = categorical_cols + ["region_strength_bin", "state_strength_bin"]
    frame = add_domain_features(base)
    model_frame = frame[numeric_cols + categorical_cols].copy()
    model_frame["text_feature"] = build_text_feature(frame, FEATURE_SET)
    return model_frame, numeric_cols, categorical_cols


def make_preprocessor(candidate: Candidate, numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=10)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", ohe),
                    ]
                ),
                categorical_cols,
            ),
            (
                "text",
                TfidfVectorizer(
                    max_features=candidate.max_features,
                    min_df=candidate.min_df,
                    ngram_range=(1, candidate.ngram_max),
                    sublinear_tf=True,
                ),
                "text_feature",
            ),
        ],
        sparse_threshold=0.3,
    )


def make_model(candidate: Candidate, y_train: np.ndarray) -> XGBClassifier:
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    return XGBClassifier(
        n_estimators=candidate.n_estimators,
        max_depth=candidate.max_depth,
        learning_rate=candidate.learning_rate,
        min_child_weight=candidate.min_child_weight,
        subsample=candidate.subsample,
        colsample_bytree=candidate.colsample_bytree,
        reg_alpha=candidate.reg_alpha,
        reg_lambda=candidate.reg_lambda,
        gamma=candidate.gamma,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=neg / max(pos, 1),
    )


def inner_split(base: pd.DataFrame, y: pd.Series, outer_train: np.ndarray, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    outer_train = np.asarray(outer_train)
    if split_name == "group_company":
        groups = base["Company"].fillna("missing").astype(str).iloc[outer_train]
        inner_train_pos, inner_val_pos = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE + 17).split(
                outer_train, y.iloc[outer_train], groups=groups
            )
        )
        return outer_train[inner_train_pos], outer_train[inner_val_pos]

    dates = base["issue_date"].fillna(pd.Timestamp("1900-01-01")).iloc[outer_train]
    cutoff = dates.quantile(0.80)
    return outer_train[np.flatnonzero(dates < cutoff)], outer_train[np.flatnonzero(dates >= cutoff)]


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def choose_threshold(y_val: np.ndarray, proba: np.ndarray) -> float:
    thresholds = np.unique(np.quantile(proba, np.linspace(0.01, 0.99, 199)))
    best = (0.0, 0.0, 0.5)
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        score = (
            balanced_accuracy_score(y_val, pred),
            f1_score(y_val, pred, zero_division=0),
            float(threshold),
        )
        if score[:2] > best[:2]:
            best = score
    return best[2]


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


def fit_predict_candidate(
    candidate: Candidate,
    model_frame: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    y: pd.Series,
) -> np.ndarray:
    preprocessor = make_preprocessor(candidate, numeric_cols, categorical_cols)
    x_train = preprocessor.fit_transform(model_frame.iloc[train_idx])
    x_pred = preprocessor.transform(model_frame.iloc[pred_idx])
    y_train = y.iloc[train_idx].to_numpy()
    model = make_model(candidate, y_train)
    model.fit(x_train, y_train)
    return model.predict_proba(x_pred)[:, 1]


def select_candidate(
    base: pd.DataFrame,
    model_frame: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    y: pd.Series,
    outer_train: np.ndarray,
    split_name: str,
) -> tuple[Candidate, float, list[dict[str, Any]]]:
    train_idx, val_idx = inner_split(base, y, outer_train, split_name)
    rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float], Candidate, float] | None = None
    for candidate in CANDIDATES:
        started = time.time()
        proba = fit_predict_candidate(
            candidate, model_frame, numeric_cols, categorical_cols, train_idx, val_idx, y
        )
        threshold = choose_threshold(y.iloc[val_idx].to_numpy(), proba)
        metrics = evaluate(y.iloc[val_idx].to_numpy(), proba, threshold)
        row = {
            "split": split_name,
            "candidate": candidate.name,
            "threshold": threshold,
            "fit_seconds": round(time.time() - started, 3),
            **metrics,
        }
        rows.append(row)
        score = (metrics["roc_auc"], metrics["average_precision"], metrics["balanced_accuracy"])
        if best is None or score > best[0]:
            best = (score, candidate, threshold)
        print(
            f"{split_name} | inner {candidate.name} | "
            f"AUC={metrics['roc_auc']:.4f} AP={metrics['average_precision']:.4f} "
            f"BAL={metrics['balanced_accuracy']:.4f}"
        )
    assert best is not None
    return best[1], best[2], rows


def compare_to_sota(results: pd.DataFrame) -> pd.DataFrame:
    benchmark = pd.read_csv(RESULTS_DIR / "concrete_epd_algorithm_benchmark_summary.csv")
    sota_rows = []
    for _, row in results.iterrows():
        split = row["split"]
        auc_col = f"roc_auc_{split}"
        acc_col = f"accuracy_{split}"
        bal_col = f"balanced_accuracy_{split}"
        f1_col = f"f1_{split}"
        best_auc_row = benchmark.sort_values(auc_col, ascending=False).iloc[0]
        best_bal_row = benchmark.sort_values(bal_col, ascending=False).iloc[0]
        sota_rows.append(
            {
                "split": split,
                "custom_algorithm": row["algorithm"],
                "custom_auc": row["roc_auc"],
                "custom_accuracy": row["accuracy"],
                "custom_balanced_accuracy": row["balanced_accuracy"],
                "custom_f1": row["f1"],
                "best_sota_auc_algorithm": best_auc_row["algorithm"],
                "best_sota_auc": best_auc_row[auc_col],
                "best_sota_auc_accuracy": best_auc_row[acc_col],
                "best_sota_auc_balanced_accuracy": best_auc_row[bal_col],
                "best_sota_auc_f1": best_auc_row[f1_col],
                "auc_delta_vs_best_sota": row["roc_auc"] - best_auc_row[auc_col],
                "beats_best_sota_auc": bool(row["roc_auc"] > best_auc_row[auc_col]),
                "best_sota_balanced_accuracy_algorithm": best_bal_row["algorithm"],
                "best_sota_balanced_accuracy": best_bal_row[bal_col],
                "balanced_accuracy_delta_vs_best_sota": row["balanced_accuracy"] - best_bal_row[bal_col],
                "beats_best_sota_balanced_accuracy": bool(
                    row["balanced_accuracy"] > best_bal_row[bal_col]
                ),
            }
        )
    return pd.DataFrame(sota_rows)


def run_fixed() -> tuple[pd.DataFrame, pd.DataFrame]:
    base, _ = load_and_prepare()
    y = make_target(base, MAIN_QUANTILE).reset_index(drop=True)
    splits = make_splits(base, y)
    model_frame, numeric_cols, categorical_cols = custom_feature_frame(base)
    result_rows: list[dict[str, Any]] = []
    for split_name in SPLITS:
        outer_train, outer_test = splits[split_name]
        started = time.time()
        proba = fit_predict_candidate(
            CARM_BOOST_FIXED,
            model_frame,
            numeric_cols,
            categorical_cols,
            outer_train,
            outer_test,
            y,
        )
        metrics = evaluate(y.iloc[outer_test].to_numpy(), proba, 0.5)
        row = {
            "algorithm": "CARM-Boost fixed single model",
            "split": split_name,
            "selected_candidate": CARM_BOOST_FIXED.name,
            "threshold": 0.5,
            "n_train": int(len(outer_train)),
            "n_test": int(len(outer_test)),
            "test_positive_rate": float(y.iloc[outer_test].mean()),
            "fit_seconds": round(time.time() - started, 3),
            **metrics,
        }
        result_rows.append(row)
        print(
            f"{split_name} | CARM-Boost fixed | "
            f"AUC={metrics['roc_auc']:.6f} ACC={metrics['accuracy']:.6f} "
            f"BAL={metrics['balanced_accuracy']:.6f}"
        )
    results = pd.DataFrame(result_rows)
    comparison = compare_to_sota(results)
    return results, comparison


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base, dataset_summary = load_and_prepare()
    y = make_target(base, MAIN_QUANTILE).reset_index(drop=True)
    splits = make_splits(base, y)
    model_frame, numeric_cols, categorical_cols = custom_feature_frame(base)

    result_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for split_name in SPLITS:
        outer_train, outer_test = splits[split_name]
        candidate, threshold, rows = select_candidate(
            base, model_frame, numeric_cols, categorical_cols, y, outer_train, split_name
        )
        selection_rows.extend(rows)
        started = time.time()
        proba = fit_predict_candidate(
            candidate, model_frame, numeric_cols, categorical_cols, outer_train, outer_test, y
        )
        metrics = evaluate(y.iloc[outer_test].to_numpy(), proba, threshold)
        row = {
            "algorithm": "CARM-Boost single model",
            "split": split_name,
            "selected_candidate": candidate.name,
            "threshold_from_inner_training": threshold,
            "n_train": int(len(outer_train)),
            "n_test": int(len(outer_test)),
            "test_positive_rate": float(y.iloc[outer_test].mean()),
            "fit_seconds": round(time.time() - started, 3),
            **metrics,
        }
        result_rows.append(row)
        print(
            f"{split_name} | OUTER CARM-Boost | candidate={candidate.name} | "
            f"AUC={metrics['roc_auc']:.4f} ACC={metrics['accuracy']:.4f} "
            f"BAL={metrics['balanced_accuracy']:.4f}"
        )

    results = pd.DataFrame(result_rows)
    selection = pd.DataFrame(selection_rows)
    comparison = compare_to_sota(results)
    return results, selection, comparison


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if "--fixed-only" in sys.argv:
        fixed_results, fixed_comparison = run_fixed()
        fixed_results.to_csv(OUT_FIXED_RESULTS, index=False)
        fixed_comparison.to_csv(OUT_FIXED_COMPARISON, index=False)
        manifest = {
            "created_utc": pd.Timestamp.utcnow().isoformat(),
            "runtime_seconds": round(time.time() - started, 3),
            "algorithm": "CARM-Boost fixed single model",
            "not_an_ensemble_of_algorithms": True,
            "description": (
                "Single cost-sensitive XGBoost classifier with carbon-procurement "
                "domain features and a fixed domain-tuned configuration."
            ),
            "target": "top decile of A1-A3 GWP per ksi of compressive strength",
            "target_quantile": MAIN_QUANTILE,
            "splits": SPLITS,
            "candidate": CARM_BOOST_FIXED.__dict__,
        }
        OUT_FIXED_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return

    results, selection, comparison = run()
    results.to_csv(OUT_RESULTS, index=False)
    selection.to_csv(OUT_SELECTION, index=False)
    comparison.to_csv(OUT_COMPARISON, index=False)
    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "runtime_seconds": round(time.time() - started, 3),
        "algorithm": "CARM-Boost single model",
        "not_an_ensemble_of_algorithms": True,
        "description": (
            "Single cost-sensitive XGBoost classifier with carbon-procurement "
            "domain features and inner-training-fold hyperparameter selection."
        ),
        "target": "top decile of A1-A3 GWP per ksi of compressive strength",
        "target_quantile": MAIN_QUANTILE,
        "splits": SPLITS,
        "candidate_count": len(CANDIDATES),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
