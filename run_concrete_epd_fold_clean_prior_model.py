"""Application-aware single model for concrete EPD screening.

The main algorithm developed here is Application-aware Carbon Risk Model
(ACRM). It is one supervised learner, not a stacked, voting, or blended
ensemble of algorithms. Its novelty is the feature construction and validation
discipline:

* procurement/application features from the EPD schema,
* optional fold-clean historical carbon-risk priors built only from the
  training fold,
* GPU-accelerated single boosted classifier with no test-fold information.

The target remains the top decile of A1-A3 GWP per ksi of compressive strength.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from run_concrete_epd_model import (
    MAIN_QUANTILE,
    RESULTS_DIR,
    build_text_feature,
    feature_columns,
    load_and_prepare,
    make_splits,
    make_target,
)


ROOT = Path(__file__).resolve().parents[1]
V5_CSV = ROOT / "data" / "raw" / "concrete_epd_mendeley_v5.csv"
RANDOM_STATE = 42
SPLITS = ["group_company", "group_epd_source", "temporal_latest20"]
REVIEW_FRACTIONS = [0.05, 0.10, 0.20, 0.30]

OUT_RESULTS = RESULTS_DIR / "concrete_epd_v5_acrm_single_model_results.csv"
OUT_OPERATIONAL = RESULTS_DIR / "concrete_epd_v5_acrm_single_model_operational_metrics.csv"
OUT_PREDICTIONS = RESULTS_DIR / "concrete_epd_v5_acrm_single_model_predictions.csv"
OUT_COMPARISON = RESULTS_DIR / "concrete_epd_v5_acrm_single_model_vs_leaders.csv"
OUT_MANIFEST = RESULTS_DIR / "concrete_epd_v5_acrm_single_model_manifest.json"

APP_COLS = [
    "Application Category: Structural",
    "Application Category: Hardscape",
    "Application Category: Paving",
    "Application Category: Infrastructure",
    "Application Category: Filler",
    "App: Structural (General)?",
    "App: Foundation, Footing?",
    "App: Slab on Grade?",
    "App: Elevated Horizontal?",
    "App: Wall?",
    "App: Column?",
    "App: Sidewalk, Curb, and Hardscape?",
    "App: Paving?",
    "App: Mass Concrete?",
    "App: Bridge Deck and Pier?",
    "App: Cement and Masonry Grout?",
    "App: Flowable Fill?",
    "App: Shotcrete?",
    "App: Lightweight Concrete?",
    "App: Exterior?",
    "App: Interior?",
    "App: Commercial?",
    "App: Residential?",
]


def algorithm_label(args: argparse.Namespace) -> str:
    if args.disable_priors:
        return "ACRM application-aware single model"
    return "ACRM fold-clean carbon-prior single model"


def family_label(args: argparse.Namespace) -> str:
    if args.disable_priors:
        return "custom single learner with application-aware construction features"
    return "custom single learner with application features and fold-clean carbon priors"


def gpu_info() -> dict[str, str]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=20,
        ).strip()
        return {"nvidia_smi": out}
    except Exception as exc:
        return {"nvidia_smi_error": str(exc)}


def boolish(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    ).astype(int)


def numeric_spec(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(series.replace({"No": np.nan, "no": np.nan}), errors="coerce")
    return values.notna().astype(int), values.fillna(0.0)


def make_application_family(frame: pd.DataFrame) -> pd.Series:
    families: list[str] = []
    categories = [
        ("structural", "Application Category: Structural"),
        ("hardscape", "Application Category: Hardscape"),
        ("paving", "Application Category: Paving"),
        ("infrastructure", "Application Category: Infrastructure"),
        ("filler", "Application Category: Filler"),
    ]
    values = {col: boolish(frame[col]) if col in frame else pd.Series(0, index=frame.index) for _, col in categories}
    for idx in frame.index:
        active = [name for name, col in categories if int(values[col].loc[idx]) == 1]
        if not active:
            families.append("general_unspecified")
        elif len(active) == 1:
            families.append(active[0])
        else:
            families.append("multi_" + "_".join(active[:3]))
    return pd.Series(families, index=frame.index)


def make_curing_family(days: pd.Series) -> pd.Series:
    vals = pd.to_numeric(days, errors="coerce")
    out = pd.Series("missing", index=days.index, dtype=object)
    out[(vals > 0) & (vals <= 7)] = "early_7_or_less"
    out[(vals > 7) & (vals <= 28)] = "standard_28"
    out[(vals > 28) & (vals <= 56)] = "extended_56"
    out[vals > 56] = "long_over_56"
    return out


def selected_app_tokens(frame: pd.DataFrame) -> pd.Series:
    labels = []
    for _, row in frame.iterrows():
        active = []
        for col in APP_COLS:
            if col in frame.columns and str(row.get(col, "")).strip().lower() in {"true", "1", "yes"}:
                active.append(re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_"))
        labels.append(" ".join(active))
    return pd.Series(labels, index=frame.index)


def enriched_feature_frame(base: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    numeric_cols, categorical_cols, _ = feature_columns()
    frame = base.copy()

    strength = frame["strength_psi"].astype(float).clip(lower=1)
    curing = frame["curing_days"].astype(float).fillna(frame["curing_days"].median()).clip(lower=1)
    frame["log_strength_psi"] = np.log1p(strength)
    frame["sqrt_strength_psi"] = np.sqrt(strength)
    frame["strength_per_curing_day"] = strength / curing
    frame["is_high_strength"] = (strength >= 5000).astype(int)
    frame["is_low_strength"] = (strength <= 2500).astype(int)
    frame["is_standard_28_day"] = (curing == 28).astype(int)

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

    for col in APP_COLS:
        if col in frame.columns:
            new_col = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")
            frame[new_col] = boolish(frame[col])

    frame["app_indicator_count"] = frame[
        [re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_") for col in APP_COLS if col in frame.columns]
    ].sum(axis=1)
    frame["application_family"] = make_application_family(frame)
    frame["curing_family"] = make_curing_family(frame["curing_days"])

    for source_col, prefix in [
        ("W/C Specified?", "wc"),
        ("Percent Fly Ash Specified?", "fly_ash_pct"),
        ("Percent Slag Specified?", "slag_pct"),
    ]:
        if source_col in frame.columns:
            specified, value = numeric_spec(frame[source_col])
            frame[f"{prefix}_specified"] = specified
            frame[f"{prefix}_value"] = value

    if "Amount of Different Applications" in frame.columns:
        frame["amount_different_applications"] = pd.to_numeric(
            frame["Amount of Different Applications"], errors="coerce"
        ).fillna(0.0)

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
    frame["application_strength_bin"] = (
        frame["application_family"].astype(str) + "|" + frame["strength_bin_500"].astype(str)
    )
    frame["region_application_strength_bin"] = (
        frame["U.S. Region of Plant"].fillna("missing").astype(str)
        + "|"
        + frame["application_family"].astype(str)
        + "|"
        + frame["strength_bin_500"].astype(str)
    )
    frame["operator_application"] = (
        frame["EPD Program Operator"].fillna("missing").astype(str)
        + "|"
        + frame["application_family"].astype(str)
    )
    frame["company_application_strength_bin"] = (
        frame["Company"].fillna("missing").astype(str)
        + "|"
        + frame["application_family"].astype(str)
        + "|"
        + frame["strength_bin_500"].astype(str)
    )

    app_numeric_cols = [
        re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_") for col in APP_COLS if col in frame.columns
    ]
    extra_numeric_cols = [
        "log_strength_psi",
        "sqrt_strength_psi",
        "strength_per_curing_day",
        "is_high_strength",
        "is_low_strength",
        "is_standard_28_day",
        "scm_indicator_count",
        "has_multiple_scm",
        "app_indicator_count",
        "wc_specified",
        "wc_value",
        "fly_ash_pct_specified",
        "fly_ash_pct_value",
        "slag_pct_specified",
        "slag_pct_value",
        "amount_different_applications",
    ]
    numeric_cols = numeric_cols + [c for c in extra_numeric_cols + app_numeric_cols if c in frame.columns]
    categorical_cols = categorical_cols + [
        "application_family",
        "curing_family",
        "region_strength_bin",
        "state_strength_bin",
        "application_strength_bin",
        "region_application_strength_bin",
        "operator_application",
        "company_application_strength_bin",
    ]
    categorical_cols = [c for c in dict.fromkeys(categorical_cols) if c in frame.columns]

    text_feature = build_text_feature(frame, type("FeatureSet", (), {"include_lci_sources": True})())
    text_feature = text_feature + " " + selected_app_tokens(frame)
    model_frame = frame[numeric_cols + categorical_cols].copy()
    model_frame["text_feature"] = text_feature

    prior_cols = [
        "Company",
        "Plant",
        "EPD Program Operator",
        "Plant Location - State",
        "U.S. Region of Plant",
        "Concrete Curation Time",
        "application_family",
        "region_strength_bin",
        "state_strength_bin",
        "application_strength_bin",
        "region_application_strength_bin",
        "operator_application",
        "company_application_strength_bin",
    ]
    prior_cols = [c for c in prior_cols if c in model_frame.columns]
    return model_frame, numeric_cols, categorical_cols, prior_cols


def safe_feature_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def smoothed_stats(
    categories: pd.Series,
    y: pd.Series,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    prior: float,
    smooth: float,
) -> tuple[np.ndarray, np.ndarray]:
    source = pd.DataFrame(
        {
            "category": categories.iloc[source_idx].fillna("missing").astype(str).to_numpy(),
            "y": y.iloc[source_idx].to_numpy(dtype=float),
        }
    )
    stats = source.groupby("category")["y"].agg(["sum", "count"])
    rate = (stats["sum"] + smooth * prior) / (stats["count"] + smooth)
    target_categories = categories.iloc[target_idx].fillna("missing").astype(str)
    encoded_rate = target_categories.map(rate).fillna(prior).to_numpy(dtype=float)
    encoded_count = target_categories.map(stats["count"]).fillna(0.0).to_numpy(dtype=float)
    return encoded_rate, np.log1p(encoded_count)


def oof_splits(
    base: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    split_name: str,
    folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    train_idx = np.asarray(train_idx)
    y_train = y.iloc[train_idx].to_numpy()
    if split_name == "group_company":
        groups = base["Company"].fillna("missing").astype(str).iloc[train_idx].to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups >= 3:
            n_splits = min(folds, n_groups)
            return [
                (train_idx[src], train_idx[val])
                for src, val in GroupKFold(n_splits=n_splits).split(train_idx, y_train, groups=groups)
            ]
    if split_name == "group_epd_source":
        groups = (
            base["EPD Source Link"]
            .fillna(base["Company"].astype(str) + "|" + base["Plant"].astype(str))
            .astype(str)
            .iloc[train_idx]
            .to_numpy()
        )
        n_groups = len(np.unique(groups))
        if n_groups >= 3:
            n_splits = min(folds, n_groups)
            return [
                (train_idx[src], train_idx[val])
                for src, val in GroupKFold(n_splits=n_splits).split(train_idx, y_train, groups=groups)
            ]

    n_splits = min(folds, int(np.bincount(y_train).min())) if len(np.unique(y_train)) == 2 else folds
    n_splits = max(2, n_splits)
    return [
        (train_idx[src], train_idx[val])
        for src, val in StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        ).split(train_idx, y_train)
    ]


def add_fold_clean_prior_features(
    base: pd.DataFrame,
    model_frame: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_name: str,
    prior_cols: list[str],
    smooth: float = 40.0,
    folds: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    out = model_frame.copy()
    prior_feature_cols: list[str] = []
    global_prior = float(y.iloc[train_idx].mean())
    for col in prior_cols:
        rate_name = f"prior_{safe_feature_name(col)}_rate"
        count_name = f"prior_{safe_feature_name(col)}_log_count"
        out[rate_name] = global_prior
        out[count_name] = 0.0
        prior_feature_cols.extend([rate_name, count_name])

        categories = out[col] if col in out.columns else base[col]
        for source_idx, val_idx in oof_splits(base, y, train_idx, split_name, folds):
            rate, count = smoothed_stats(categories, y, source_idx, val_idx, global_prior, smooth)
            out.loc[val_idx, rate_name] = rate
            out.loc[val_idx, count_name] = count
        rate, count = smoothed_stats(categories, y, train_idx, test_idx, global_prior, smooth)
        out.loc[test_idx, rate_name] = rate
        out.loc[test_idx, count_name] = count
    return out, prior_feature_cols


def make_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    prior_feature_cols: list[str],
    max_text_features: int,
) -> ColumnTransformer:
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
                numeric_cols + prior_feature_cols,
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
                    max_features=max_text_features,
                    min_df=2,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
                "text_feature",
            ),
        ],
        sparse_threshold=0.3,
    )


def make_model(y_train: np.ndarray, args: argparse.Namespace) -> XGBClassifier:
    pos = int(np.sum(y_train))
    neg = int(len(y_train) - pos)
    if args.learner == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            min_child_samples=args.min_child_samples,
            min_child_weight=args.min_child_weight,
            subsample=args.subsample,
            subsample_freq=1,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            objective="binary",
            device_type="gpu",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            verbose=-1,
        )
    return XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=neg / max(pos, 1),
    )


def fit_model(model: Any, x_train: Any, y_train: np.ndarray, args: argparse.Namespace) -> Any:
    try:
        model.fit(x_train, y_train)
    except Exception:
        if args.learner == "lgbm" and hasattr(model, "set_params"):
            model.set_params(device_type="cpu")
            model.fit(x_train, y_train)
        else:
            raise
    return model


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def choose_threshold(y_val: np.ndarray, score: np.ndarray, policy: str) -> float:
    thresholds = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 301)))
    best: tuple[float, float] | None = None
    best_threshold = 0.5
    for threshold in thresholds:
        pred = (score >= threshold).astype(int)
        precision = precision_score(y_val, pred, zero_division=0)
        recall = recall_score(y_val, pred, zero_division=0)
        bal = balanced_accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, zero_division=0)
        specificity = specificity_score(y_val, pred)
        if policy == "balanced_f1":
            utility = 0.45 * bal + 0.45 * f1 + 0.05 * precision + 0.05 * recall
        elif policy == "recall_guard":
            utility = 0.30 * bal + 0.30 * f1 + 0.25 * min(recall, 0.90) + 0.15 * precision
        elif policy == "precision_guard":
            utility = 0.35 * bal + 0.35 * f1 + 0.20 * precision + 0.10 * specificity
        else:
            utility = f1
        candidate = (float(utility), float(threshold))
        if best is None or candidate > best:
            best = candidate
            best_threshold = float(threshold)
    return best_threshold


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
        "brier_score": float(brier_score_loss(y_true, proba)),
    }


def topk_metrics(y_true: np.ndarray, score: np.ndarray, review_fraction: float) -> dict[str, float]:
    n = len(y_true)
    n_review = max(1, int(math.ceil(n * review_fraction)))
    positives = float(np.sum(y_true))
    selected = np.argsort(score)[-n_review:]
    true_positive = float(np.sum(y_true[selected]))
    precision = true_positive / n_review
    capture = true_positive / positives if positives else np.nan
    base_rate = positives / n if n else np.nan
    return {
        f"top_{int(review_fraction * 100)}pct_precision": precision,
        f"top_{int(review_fraction * 100)}pct_recall_capture": capture,
        f"top_{int(review_fraction * 100)}pct_lift": precision / base_rate if base_rate else np.nan,
    }


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (proba >= lo) & (proba < hi if hi < 1.0 else proba <= hi)
        if not np.any(mask):
            continue
        ece += np.sum(mask) / n * abs(float(np.mean(y_true[mask])) - float(np.mean(proba[mask])))
    return float(ece)


def fit_predict_split(
    base: pd.DataFrame,
    y: pd.Series,
    model_frame: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    prior_cols: list[str],
    split_name: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, int, dict[str, Any]]:
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    if args.disable_priors:
        augmented = model_frame.copy()
        prior_feature_cols = []
    else:
        augmented, prior_feature_cols = add_fold_clean_prior_features(
            base,
            model_frame,
            y,
            train_idx,
            test_idx,
            split_name,
            prior_cols,
            smooth=args.prior_smoothing,
            folds=args.prior_folds,
        )
    preprocessor = make_preprocessor(
        numeric_cols,
        categorical_cols,
        prior_feature_cols,
        max_text_features=args.max_text_features,
    )
    x_train = preprocessor.fit_transform(augmented.iloc[train_idx])
    x_test = preprocessor.transform(augmented.iloc[test_idx])
    if not sparse.issparse(x_train):
        x_train = sparse.csr_matrix(x_train)
        x_test = sparse.csr_matrix(x_test)
    model = make_model(y.iloc[train_idx].to_numpy(), args)
    model = fit_model(model, x_train, y.iloc[train_idx].to_numpy(), args)
    proba = model.predict_proba(x_test)[:, 1]

    threshold = 0.5
    if args.threshold_policy != "fixed_05":
        inner_splits = oof_splits(base, y, train_idx, split_name, folds=3)
        inner_train, inner_val = inner_splits[-1]
        if args.disable_priors:
            inner_augmented = model_frame.copy()
            inner_prior_features = []
        else:
            inner_model_frame = model_frame.copy()
            inner_augmented, inner_prior_features = add_fold_clean_prior_features(
                base,
                inner_model_frame,
                y,
                inner_train,
                inner_val,
                split_name,
                prior_cols,
                smooth=args.prior_smoothing,
                folds=max(2, min(3, args.prior_folds)),
            )
        inner_preprocessor = make_preprocessor(
            numeric_cols,
            categorical_cols,
            inner_prior_features,
            max_text_features=args.max_text_features,
        )
        x_inner_train = inner_preprocessor.fit_transform(inner_augmented.iloc[inner_train])
        x_inner_val = inner_preprocessor.transform(inner_augmented.iloc[inner_val])
        if not sparse.issparse(x_inner_train):
            x_inner_train = sparse.csr_matrix(x_inner_train)
            x_inner_val = sparse.csr_matrix(x_inner_val)
        inner_model = make_model(y.iloc[inner_train].to_numpy(), args)
        inner_model = fit_model(inner_model, x_inner_train, y.iloc[inner_train].to_numpy(), args)
        val_proba = inner_model.predict_proba(x_inner_val)[:, 1]
        threshold = choose_threshold(
            y.iloc[inner_val].to_numpy(),
            val_proba,
            policy=args.threshold_policy,
        )

    diagnostics = {
        "feature_count_train": int(x_train.shape[1]),
        "prior_feature_count": len(prior_feature_cols),
        "threshold_policy": args.threshold_policy,
    }
    return proba, threshold, int(x_train.shape[1]), diagnostics


def compare_to_leaders(results: pd.DataFrame, operational: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics_path = RESULTS_DIR / "concrete_epd_v5_final_all_algorithm_metrics_with_new_variants.csv"
    op_path = RESULTS_DIR / "concrete_epd_v5_operational_metrics_all_models.csv"
    if metrics_path.exists():
        leaders = pd.read_csv(metrics_path)
        for _, result in results.iterrows():
            split_leaders = leaders[leaders["split"] == result["split"]]
            if split_leaders.empty:
                continue
            for metric in [
                "roc_auc",
                "average_precision",
                "accuracy",
                "balanced_accuracy",
                "f1",
                "precision",
                "recall",
                "specificity",
            ]:
                leader = split_leaders.loc[pd.to_numeric(split_leaders[metric], errors="coerce").idxmax()]
                rows.append(
                    {
                        "metric_family": "classification",
                        "split": result["split"],
                        "metric": metric,
                        "model_value": result[metric],
                        "leader_algorithm": leader["algorithm"],
                        "leader_value": leader[metric],
                        "delta_vs_leader": result[metric] - leader[metric],
                        "wins_metric": bool(result[metric] > leader[metric]),
                    }
                )
    if op_path.exists():
        leaders = pd.read_csv(op_path)
        for _, result in operational.iterrows():
            split_leaders = leaders[leaders["split"] == result["split"]]
            if split_leaders.empty:
                continue
            for metric in [
                "roc_auc",
                "average_precision",
                "top_5pct_precision",
                "top_5pct_recall_capture",
                "top_10pct_precision",
                "top_10pct_recall_capture",
                "top_20pct_precision",
                "top_20pct_recall_capture",
                "top_30pct_precision",
                "top_30pct_recall_capture",
            ]:
                leader = split_leaders.loc[pd.to_numeric(split_leaders[metric], errors="coerce").idxmax()]
                rows.append(
                    {
                        "metric_family": "operational_topk",
                        "split": result["split"],
                        "metric": metric,
                        "model_value": result[metric],
                        "leader_algorithm": leader["algorithm"],
                        "leader_value": leader[metric],
                        "delta_vs_leader": result[metric] - leader[metric],
                        "wins_metric": bool(result[metric] > leader[metric]),
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--learner", choices=["xgb", "lgbm"], default="xgb")
    parser.add_argument("--n-estimators", type=int, default=650)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.028)
    parser.add_argument("--min-child-weight", type=float, default=2.0)
    parser.add_argument("--subsample", type=float, default=0.90)
    parser.add_argument("--colsample-bytree", type=float, default=0.86)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--max-text-features", type=int, default=14000)
    parser.add_argument("--prior-smoothing", type=float, default=40.0)
    parser.add_argument("--prior-folds", type=int, default=5)
    parser.add_argument("--disable-priors", action="store_true")
    parser.add_argument(
        "--threshold-policy",
        choices=["fixed_05", "balanced_f1", "recall_guard", "precision_guard"],
        default="balanced_f1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CONCRETE_EPD_CSV"] = str(V5_CSV)
    base, dataset_summary = load_and_prepare()
    y = make_target(base, MAIN_QUANTILE).reset_index(drop=True)
    splits = make_splits(base, y)
    requested_splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    model_frame, numeric_cols, categorical_cols, prior_cols = enriched_feature_frame(base)
    label = algorithm_label(args)
    family = family_label(args)

    result_rows: list[dict[str, Any]] = []
    operational_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, Any]] = []
    for split_name in requested_splits:
        train_idx, test_idx = splits[split_name]
        print(f"Running {label} | split={split_name}")
        tic = time.time()
        proba, threshold, feature_count, diagnostics = fit_predict_split(
            base,
            y,
            model_frame,
            numeric_cols,
            categorical_cols,
            prior_cols,
            split_name,
            train_idx,
            test_idx,
            args,
        )
        y_test = y.iloc[test_idx].to_numpy()
        metrics = evaluate(y_test, proba, threshold)
        result_rows.append(
            {
                "algorithm": label,
                "family": family,
                "split": split_name,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_positive_rate": float(y_test.mean()),
                "threshold": threshold,
                "feature_count": feature_count,
                "fit_seconds": round(time.time() - tic, 3),
                **metrics,
            }
        )
        op_row = {
            "dataset_version": 5,
            "split": split_name,
            "algorithm": label,
            "n_test": int(len(test_idx)),
            "base_positive_rate": float(y_test.mean()),
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "calibration_brier_score": metrics["brier_score"],
            "calibration_ece_10_bins": expected_calibration_error(y_test, proba, bins=10),
        }
        for review_fraction in REVIEW_FRACTIONS:
            op_row.update(topk_metrics(y_test, proba, review_fraction))
        operational_rows.append(op_row)
        diagnostics_rows.append({"split": split_name, **diagnostics})
        prediction_rows.append(
            pd.DataFrame(
                {
                    "dataset_version": 5,
                    "split": split_name,
                    "row_index": test_idx,
                    "y_true": y_test,
                    "acrm_probability": proba,
                    "threshold": threshold,
                    "gwp_per_ksi": base["gwp_per_ksi"].iloc[test_idx].to_numpy(),
                    "company": base["Company"].iloc[test_idx].to_numpy(),
                    "plant": base["Plant"].iloc[test_idx].to_numpy(),
                    "application_family": model_frame["application_family"].iloc[test_idx].to_numpy(),
                }
            )
        )
        print(
            f"{split_name} | AUC={metrics['roc_auc']:.6f} "
            f"AP={metrics['average_precision']:.6f} "
            f"F1={metrics['f1']:.6f}"
        )

    results = pd.DataFrame(result_rows)
    operational = pd.DataFrame(operational_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    comparison = compare_to_leaders(results, operational)

    results.to_csv(OUT_RESULTS, index=False)
    operational.to_csv(OUT_OPERATIONAL, index=False)
    predictions.to_csv(OUT_PREDICTIONS, index=False)
    comparison.to_csv(OUT_COMPARISON, index=False)

    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "runtime_seconds": round(time.time() - started, 3),
        "algorithm": label,
        "not_a_stacked_or_voting_ensemble": True,
        "single_estimator": (
            "LGBMClassifier, device_type=gpu"
            if args.learner == "lgbm"
            else "XGBClassifier, GPU hist, device=cuda"
        ),
        "dataset_version": 5,
        "csv_path": str(V5_CSV),
        "target": "top decile of A1-A3 GWP per ksi of compressive strength",
        "target_quantile": MAIN_QUANTILE,
        "splits": requested_splits,
        "dataset_rows_after_trim": dataset_summary["model_rows_after_trim"],
        "gpu": gpu_info(),
        "parameters": vars(args),
        "diagnostics": diagnostics_rows,
        "outputs": [
            str(OUT_RESULTS),
            str(OUT_OPERATIONAL),
            str(OUT_PREDICTIONS),
            str(OUT_COMPARISON),
        ],
        "reviewer_guardrail": (
            "Historical prior encodings were disabled for this selected run; reported predictions use "
            "application, specification, supplier/location metadata, non-outcome LCI-source text, and "
            "strength/curing features only."
            if args.disable_priors
            else "Historical prior features are generated with out-of-fold encodings for training rows "
            "and with outer-training-fold-only mappings for test rows."
        ),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(results.to_csv(index=False))
    if not comparison.empty:
        print(comparison.groupby(["metric_family", "split"])["wins_metric"].sum().to_string())


if __name__ == "__main__":
    main()
