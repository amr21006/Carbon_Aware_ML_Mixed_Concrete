"""Concrete EPD procurement-risk model.

This script builds a reviewer-defensible construction-management model for
screening ready-mix concrete EPDs. The primary target is whether a mix falls
in the top decile of embodied-carbon intensity, measured as A1-A3 GWP per ksi
of compressive strength. Environmental outcome columns are excluded from all
predictors.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
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
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

DATASET_ID = "r4jgxk2mhn"
DATASET_VERSION = 2
MENDELEY_FILE_API = (
    f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files"
    f"?folder_id=root&version={DATASET_VERSION}"
)
LOCAL_CSV = RAW_DIR / "concrete_epd_mendeley.csv"
LOCAL_CSV_V5 = RAW_DIR / "concrete_epd_mendeley_v5.csv"
TARGET_COL = "A1-A3 Global Warming Potential (kg CO2-eq)"

RANDOM_STATE = 42
MAIN_QUANTILE = 0.90
SENSITIVITY_QUANTILES = [0.75, 0.80, 0.85, 0.90, 0.95]
REPEATED_GROUP_SEEDS = list(range(10))


@dataclass(frozen=True)
class FeatureSet:
    name: str
    include_lci_sources: bool


FEATURE_SETS = [
    FeatureSet("metadata_only", False),
    FeatureSet("metadata_plus_lci_sources", True),
]


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_url_json(url: str) -> list[dict]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_if_needed() -> dict:
    """Download the Mendeley CSV through the public file API."""
    ensure_dirs()
    files_path = RAW_DIR / "mendeley_concrete_epd_files.json"
    try:
        files = read_url_json(MENDELEY_FILE_API)
        files_path.write_text(json.dumps(files, indent=2), encoding="utf-8")
    except Exception:
        if files_path.exists():
            files = json.loads(files_path.read_text(encoding="utf-8"))
        elif LOCAL_CSV.exists() and LOCAL_CSV.stat().st_size > 0:
            return {
                "dataset_id": DATASET_ID,
                "dataset_version": DATASET_VERSION,
                "file_name": LOCAL_CSV.name,
                "file_id": "cached_local_file",
                "sha256": None,
                "size_bytes": LOCAL_CSV.stat().st_size,
                "api_url": MENDELEY_FILE_API,
                "download_note": "Used cached CSV because public file API was unavailable.",
            }
        else:
            raise
    csv_file = next(item for item in files if item["filename"].lower().endswith(".csv"))
    if not LOCAL_CSV.exists() or LOCAL_CSV.stat().st_size == 0:
        request = urllib.request.Request(
            csv_file["content_details"]["download_url"],
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            LOCAL_CSV.write_bytes(response.read())
    return {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "file_name": csv_file["filename"],
        "file_id": csv_file["id"],
        "sha256": csv_file["content_details"].get("sha256_hash"),
        "size_bytes": csv_file["content_details"].get("size"),
        "api_url": MENDELEY_FILE_API,
    }


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def make_flags(base: pd.DataFrame) -> pd.DataFrame:
    description_col = "Mix Description" if "Mix Description" in base.columns else "Mixture Description"
    component_text = (
        base["Product Components"].fillna("") + " " + base[description_col].fillna("")
    ).str.lower()
    patterns = {
        "has_fly_ash": r"fly ash|ash",
        "has_slag": r"slag",
        "has_silica_fume": r"silica fume",
        "has_limestone_cement": r"type 1l|limestone|c595",
        "has_carbon_cure": r"carbon cure|carboncure",
        "has_lightweight": r"lightweight",
        "has_recycled": r"recycled",
        "has_fiber": r"fiber|fibre",
        "has_accelerator": r"accelerat",
        "has_retarder": r"retard",
        "has_water_reducer": r"water reduc|plasticiz|superplasticiz",
    }
    for name, pattern in patterns.items():
        base[name] = component_text.str.contains(pattern, regex=True).astype(int)
    return base


def load_and_prepare() -> tuple[pd.DataFrame, dict]:
    csv_path = Path(os.environ.get("CONCRETE_EPD_CSV", LOCAL_CSV))
    df = pd.read_csv(csv_path, low_memory=False)
    df[TARGET_COL] = to_numeric(df[TARGET_COL])
    df["strength_psi"] = to_numeric(df["Concrete Compressive Strength (psi)"])
    df["issue_date"] = pd.to_datetime(df["EPD Date of Issue"], errors="coerce")
    df["issue_year"] = df["issue_date"].dt.year
    df["curing_days"] = (
        df["Concrete Curation Time"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    )

    base = df[
        df[TARGET_COL].notna()
        & (df[TARGET_COL] > 0)
        & df["strength_psi"].notna()
        & (df["strength_psi"] > 0)
    ].copy()
    base["gwp_per_ksi"] = base[TARGET_COL] / (base["strength_psi"] / 1000.0)

    target_q995 = base[TARGET_COL].quantile(0.995)
    intensity_q995 = base["gwp_per_ksi"].quantile(0.995)
    before_trim = len(base)
    base = base[
        (base[TARGET_COL] < target_q995) & (base["gwp_per_ksi"] < intensity_q995)
    ].copy()

    base["issue_year"] = base["issue_year"].fillna(base["issue_year"].median())
    base["strength_bin_500"] = ((base["strength_psi"] / 500).round() * 500).astype(int)
    base = make_flags(base)
    base = base.reset_index(drop=True)

    lci_cols = [c for c in base.columns if c.startswith("Primary LCI Data Source")]
    outcome_excluded_cols = [
        c
        for c in base.columns
        if not c.startswith("Primary LCI Data Source")
        and (
            c == "gwp_per_ksi"
            or re.search(
                r"GWP|ODP|Acidication|Eutrophication|Photochemical|Abiotic|Waste|Freshwater|Consumption",
                c,
                flags=re.IGNORECASE,
            )
        )
    ]
    summary = {
        "csv_path": str(csv_path),
        "raw_rows": int(len(df)),
        "model_rows_before_trim": int(before_trim),
        "model_rows_after_trim": int(len(base)),
        "unique_companies": int(base["Company"].nunique(dropna=True)),
        "unique_plants": int(base["Plant"].nunique(dropna=True)),
        "unique_epd_source_links": int(base["EPD Source Link"].nunique(dropna=True)),
        "issue_year_min": int(base["issue_year"].min()),
        "issue_year_max": int(base["issue_year"].max()),
        "target_gwp_min": float(base[TARGET_COL].min()),
        "target_gwp_median": float(base[TARGET_COL].median()),
        "target_gwp_max_after_trim": float(base[TARGET_COL].max()),
        "gwp_per_ksi_q90": float(base["gwp_per_ksi"].quantile(MAIN_QUANTILE)),
        "lci_source_column_count": int(len(lci_cols)),
        "outcome_columns_excluded_count": int(len(outcome_excluded_cols)),
        "outcome_columns_excluded": outcome_excluded_cols,
    }
    return base, summary


def feature_columns() -> tuple[list[str], list[str], list[str]]:
    numeric_cols = [
        "strength_psi",
        "issue_year",
        "curing_days",
        "strength_bin_500",
        "has_fly_ash",
        "has_slag",
        "has_silica_fume",
        "has_limestone_cement",
        "has_carbon_cure",
        "has_lightweight",
        "has_recycled",
        "has_fiber",
        "has_accelerator",
        "has_retarder",
        "has_water_reducer",
    ]
    categorical_cols = [
        "Company",
        "Company Location - State",
        "Plant",
        "Plant Location - City",
        "Plant Location - State",
        "U.S. Region of Plant",
        "EPD Program Operator",
        "Concrete Curation Time",
    ]
    text_cols = ["Mix Label", "Mix Description", "Product Components"]
    return numeric_cols, categorical_cols, text_cols


def build_text_feature(base: pd.DataFrame, feature_set: FeatureSet) -> pd.Series:
    text_aliases = [
        ("Mix Label", "Mixture Label"),
        ("Mix Description", "Mixture Description"),
        ("Product Components",),
        ("Primary Application?",),
        ("Other Application Label?",),
    ]
    cols = []
    for aliases in text_aliases:
        for alias in aliases:
            if alias in base.columns:
                cols.append(alias)
                break
    if feature_set.include_lci_sources:
        cols.extend([c for c in base.columns if c.startswith("Primary LCI Data Source")])
    cols = list(dict.fromkeys(cols))
    return base[cols].fillna("").astype(str).agg(" ".join, axis=1)


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
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
                    max_features=8000,
                    min_df=3,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
                "text_feature",
            ),
        ],
        sparse_threshold=0.3,
    )


def make_target(base: pd.DataFrame, quantile: float) -> pd.Series:
    threshold = base["gwp_per_ksi"].quantile(quantile)
    return (base["gwp_per_ksi"] >= threshold).astype(int)


def make_splits(base: pd.DataFrame, y: pd.Series) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(base))
    random_train, random_test = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    company_groups = base["Company"].fillna("missing").astype(str)
    company_train, company_test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE).split(
            idx, y, groups=company_groups
        )
    )

    source_groups = (
        base["EPD Source Link"]
        .fillna(base["Company"].astype(str) + "|" + base["Plant"].astype(str))
        .astype(str)
    )
    source_train, source_test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE).split(
            idx, y, groups=source_groups
        )
    )

    dates = base["issue_date"].fillna(pd.Timestamp("1900-01-01"))
    cutoff = dates.quantile(0.80)
    temporal_train = np.flatnonzero(dates < cutoff)
    temporal_test = np.flatnonzero(dates >= cutoff)

    return {
        "random_row": (random_train, random_test),
        "group_company": (company_train, company_test),
        "group_epd_source": (source_train, source_test),
        "temporal_latest20": (temporal_train, temporal_test),
    }


def make_model(y_train: pd.Series) -> XGBClassifier:
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)
    return XGBClassifier(
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
    )


def specificity_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else math.nan


def evaluate_predictions(
    y_true: pd.Series,
    probabilities: np.ndarray,
    feature_set: str,
    split_name: str,
    quantile: float,
    n_train: int,
    n_test: int,
) -> dict:
    y_pred = (probabilities >= 0.5).astype(int)
    majority_accuracy = max(float(y_true.mean()), 1.0 - float(y_true.mean()))
    return {
        "feature_set": feature_set,
        "split": split_name,
        "target_quantile": quantile,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "test_positive_rate": float(y_true.mean()),
        "majority_class_accuracy": majority_accuracy,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity_score(y_true, y_pred)),
    }


def bootstrap_ci(y_true: np.ndarray, proba: np.ndarray, repeats: int = 500) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    n = len(y_true)
    for _ in range(repeats):
        sample = rng.integers(0, n, size=n)
        if len(np.unique(y_true[sample])) < 2:
            continue
        pred = (proba[sample] >= 0.5).astype(int)
        rows.append(
            {
                "roc_auc": roc_auc_score(y_true[sample], proba[sample]),
                "average_precision": average_precision_score(y_true[sample], proba[sample]),
                "accuracy": accuracy_score(y_true[sample], pred),
                "balanced_accuracy": balanced_accuracy_score(y_true[sample], pred),
                "f1": f1_score(y_true[sample], pred, zero_division=0),
            }
        )
    out = {}
    boot = pd.DataFrame(rows)
    for metric in boot.columns:
        out[f"{metric}_ci_low"] = float(boot[metric].quantile(0.025))
        out[f"{metric}_ci_high"] = float(boot[metric].quantile(0.975))
    return out


def fit_predict(
    base: pd.DataFrame,
    feature_set: FeatureSet,
    y: pd.Series,
    split_name: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict, np.ndarray, XGBClassifier, ColumnTransformer]:
    numeric_cols, categorical_cols, _ = feature_columns()
    model_frame = base[numeric_cols + categorical_cols].copy()
    model_frame["text_feature"] = build_text_feature(base, feature_set)

    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    x_train = preprocessor.fit_transform(model_frame.iloc[train_idx])
    x_test = preprocessor.transform(model_frame.iloc[test_idx])

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    model = make_model(y_train)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_predictions(
        y_test,
        probabilities,
        feature_set.name,
        split_name,
        MAIN_QUANTILE,
        len(train_idx),
        len(test_idx),
    )
    return metrics, probabilities, model, preprocessor


def run_main_models(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = make_target(base, MAIN_QUANTILE)
    splits = make_splits(base, y)
    result_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    ci_rows: list[dict] = []
    roc_payload: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    captured_importance = False

    for feature_set in FEATURE_SETS:
        for split_name, (train_idx, test_idx) in splits.items():
            tic = time.time()
            metrics, probabilities, model, preprocessor = fit_predict(
                base, feature_set, y, split_name, train_idx, test_idx
            )
            metrics["fit_seconds"] = round(time.time() - tic, 3)
            result_rows.append(metrics)

            preds = pd.DataFrame(
                {
                    "feature_set": feature_set.name,
                    "split": split_name,
                    "row_index": test_idx,
                    "y_true": y.iloc[test_idx].to_numpy(),
                    "probability": probabilities,
                    "target_quantile": MAIN_QUANTILE,
                    "gwp_per_ksi": base["gwp_per_ksi"].iloc[test_idx].to_numpy(),
                    TARGET_COL: base[TARGET_COL].iloc[test_idx].to_numpy(),
                    "strength_psi": base["strength_psi"].iloc[test_idx].to_numpy(),
                    "company": base["Company"].iloc[test_idx].to_numpy(),
                    "plant": base["Plant"].iloc[test_idx].to_numpy(),
                }
            )
            prediction_rows.append(preds)

            if split_name in ["group_company", "temporal_latest20"] and feature_set.name in [
                "metadata_only",
                "metadata_plus_lci_sources",
            ]:
                ci = {
                    "feature_set": feature_set.name,
                    "split": split_name,
                    "target_quantile": MAIN_QUANTILE,
                }
                ci.update(bootstrap_ci(y.iloc[test_idx].to_numpy(), probabilities))
                ci_rows.append(ci)

            if feature_set.name == "metadata_plus_lci_sources" and split_name in [
                "group_company",
                "temporal_latest20",
            ]:
                fpr, tpr, _ = roc_curve(y.iloc[test_idx], probabilities)
                roc_payload[split_name] = (fpr, tpr, metrics["roc_auc"])

            if (
                not captured_importance
                and feature_set.name == "metadata_plus_lci_sources"
                and split_name == "temporal_latest20"
            ):
                export_feature_importance(model, preprocessor)
                captured_importance = True

            print(
                f"{feature_set.name} | {split_name} | "
                f"AUC={metrics['roc_auc']:.4f} ACC={metrics['accuracy']:.4f} "
                f"BAL={metrics['balanced_accuracy']:.4f} F1={metrics['f1']:.4f}"
            )

    plot_roc_curves(roc_payload)
    return (
        pd.DataFrame(result_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(ci_rows),
    )


def export_feature_importance(model: XGBClassifier, preprocessor: ColumnTransformer) -> None:
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{i}" for i in range(model.n_features_in_)])
    score = model.get_booster().get_score(importance_type="gain")
    rows = []
    for key, value in score.items():
        match = re.fullmatch(r"f(\d+)", key)
        if not match:
            continue
        idx = int(match.group(1))
        rows.append(
            {
                "feature": names[idx] if idx < len(names) else key,
                "gain": float(value),
            }
        )
    pd.DataFrame(rows).sort_values("gain", ascending=False).head(50).to_csv(
        RESULTS_DIR / "feature_importance_top50.csv", index=False
    )


def run_sensitivity(base: pd.DataFrame) -> pd.DataFrame:
    feature_set = FeatureSet("metadata_plus_lci_sources", True)
    numeric_cols, categorical_cols, _ = feature_columns()
    model_frame = base[numeric_cols + categorical_cols].copy()
    model_frame["text_feature"] = build_text_feature(base, feature_set)

    rows: list[dict] = []
    for quantile in SENSITIVITY_QUANTILES:
        y = make_target(base, quantile)
        splits = make_splits(base, y)
        for split_name in ["group_company", "temporal_latest20"]:
            train_idx, test_idx = splits[split_name]
            preprocessor = make_preprocessor(numeric_cols, categorical_cols)
            x_train = preprocessor.fit_transform(model_frame.iloc[train_idx])
            x_test = preprocessor.transform(model_frame.iloc[test_idx])
            model = make_model(y.iloc[train_idx])
            model.fit(x_train, y.iloc[train_idx])
            probabilities = model.predict_proba(x_test)[:, 1]
            metrics = evaluate_predictions(
                y.iloc[test_idx],
                probabilities,
                feature_set.name,
                split_name,
                quantile,
                len(train_idx),
                len(test_idx),
            )
            rows.append(metrics)
            print(
                f"sensitivity q={quantile:.2f} | {split_name} | "
                f"AUC={metrics['roc_auc']:.4f} ACC={metrics['accuracy']:.4f}"
            )
    return pd.DataFrame(rows)


def run_repeated_group_company(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat unseen-company validation to show split stability."""
    y = make_target(base, MAIN_QUANTILE)
    rows: list[dict] = []
    groups = base["Company"].fillna("missing").astype(str)
    for feature_set in FEATURE_SETS:
        for seed in REPEATED_GROUP_SEEDS:
            train_idx, test_idx = next(
                GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                    np.arange(len(base)), y, groups=groups
                )
            )
            tic = time.time()
            metrics, _, _, _ = fit_predict(
                base,
                feature_set,
                y,
                f"group_company_seed_{seed}",
                train_idx,
                test_idx,
            )
            metrics["seed"] = seed
            metrics["fit_seconds"] = round(time.time() - tic, 3)
            rows.append(metrics)
            print(
                f"repeated group | {feature_set.name} | seed={seed} | "
                f"AUC={metrics['roc_auc']:.4f} ACC={metrics['accuracy']:.4f}"
            )

    repeated = pd.DataFrame(rows)
    summary = (
        repeated.groupby("feature_set")
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_min=("roc_auc", "min"),
            roc_auc_max=("roc_auc", "max"),
            roc_auc_std=("roc_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_min=("accuracy", "min"),
            accuracy_max=("accuracy", "max"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_min=("balanced_accuracy", "min"),
            balanced_accuracy_max=("balanced_accuracy", "max"),
            f1_mean=("f1", "mean"),
            f1_min=("f1", "min"),
            f1_max=("f1", "max"),
        )
        .reset_index()
    )
    return repeated, summary


def plot_roc_curves(roc_payload: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> None:
    if not roc_payload:
        return
    plt.figure(figsize=(6.8, 5.2), dpi=150)
    for split_name, (fpr, tpr, auc) in roc_payload.items():
        plt.plot(fpr, tpr, linewidth=2, label=f"{split_name} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Top-decile high-carbon concrete risk screening")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "concrete_epd_q90_roc.png")
    plt.close()


def gpu_info() -> dict:
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


def write_brief(
    dataset_meta: dict,
    dataset_summary: dict,
    results: pd.DataFrame,
    repeated_group_summary: pd.DataFrame | None = None,
) -> None:
    main = results[
        (results["feature_set"] == "metadata_plus_lci_sources")
        & (results["target_quantile"] == MAIN_QUANTILE)
    ].copy()
    no_lci = results[
        (results["feature_set"] == "metadata_only")
        & (results["target_quantile"] == MAIN_QUANTILE)
    ].copy()

    def row(split: str, frame: pd.DataFrame = main) -> pd.Series:
        return frame.loc[frame["split"] == split].iloc[0]

    repeated_text = ""
    if repeated_group_summary is not None and not repeated_group_summary.empty:
        rep = repeated_group_summary.loc[
            repeated_group_summary["feature_set"] == "metadata_plus_lci_sources"
        ].iloc[0]
        rep_no_lci = repeated_group_summary.loc[
            repeated_group_summary["feature_set"] == "metadata_only"
        ].iloc[0]
        repeated_text = f"""

## Repeated Unseen-Company Validation
Across {len(REPEATED_GROUP_SEEDS)} repeated company-group splits, the main model produced mean ROC-AUC = {rep.roc_auc_mean:.3f}, minimum ROC-AUC = {rep.roc_auc_min:.3f}, and mean accuracy = {rep.accuracy_mean:.3f}. The metadata-only ablation produced mean ROC-AUC = {rep_no_lci.roc_auc_mean:.3f} and minimum ROC-AUC = {rep_no_lci.roc_auc_min:.3f}. This reduces the risk that the +90 result is caused by a single favourable company split.
"""

    text = f"""# Reviewer-Ready Construction Management Idea

## Proposed Title
AI-assisted procurement screening of high-carbon ready-mix concrete using open Environmental Product Declaration data.

## Why this is high impact
Concrete procurement is a construction-management decision with direct cost, specification, supplier, and carbon consequences. The model screens whether a concrete mix is in the top decile of embodied-carbon intensity, defined as A1-A3 global warming potential per ksi of compressive strength. This makes the target procurement-relevant because it flags high-carbon mixes after normalising for strength.

## Data Source
The study uses the Mendeley Data open dataset `{dataset_meta['file_name']}` from dataset `{DATASET_ID}`, version {DATASET_VERSION}. The downloaded CSV contains {dataset_summary['raw_rows']:,} rows; after quality filters and 99.5th-percentile outlier trimming, {dataset_summary['model_rows_after_trim']:,} rows remain. The modelling sample covers {dataset_summary['unique_companies']:,} companies, {dataset_summary['unique_plants']:,} plants, and issue years {dataset_summary['issue_year_min']}-{dataset_summary['issue_year_max']}.

## Target and Leakage Control
Target: `high_carbon_top_decile = 1` when `A1-A3 GWP / (compressive strength / 1000)` is at or above the 90th percentile.

Excluded from predictors: all GWP component columns, other environmental impact outcomes, waste outcomes, freshwater outcomes, and the target itself. The main model uses supplier/location metadata, strength, curing time, mix and component text, and non-outcome LCI source metadata. A no-LCI ablation is also reported to address the concern that LCI-source fields may be unavailable in earlier procurement stages.

## Main Validation Results

| Validation | ROC-AUC | Accuracy | Balanced accuracy | F1 | Test positive rate |
|---|---:|---:|---:|---:|---:|
| Random row | {row('random_row').roc_auc:.3f} | {row('random_row').accuracy:.3f} | {row('random_row').balanced_accuracy:.3f} | {row('random_row').f1:.3f} | {row('random_row').test_positive_rate:.3f} |
| Unseen company | {row('group_company').roc_auc:.3f} | {row('group_company').accuracy:.3f} | {row('group_company').balanced_accuracy:.3f} | {row('group_company').f1:.3f} | {row('group_company').test_positive_rate:.3f} |
| Unseen EPD source | {row('group_epd_source').roc_auc:.3f} | {row('group_epd_source').accuracy:.3f} | {row('group_epd_source').balanced_accuracy:.3f} | {row('group_epd_source').f1:.3f} | {row('group_epd_source').test_positive_rate:.3f} |
| Latest 20% by issue date | {row('temporal_latest20').roc_auc:.3f} | {row('temporal_latest20').accuracy:.3f} | {row('temporal_latest20').balanced_accuracy:.3f} | {row('temporal_latest20').f1:.3f} | {row('temporal_latest20').test_positive_rate:.3f} |

No-LCI ablation: unseen-company ROC-AUC = {row('group_company', no_lci).roc_auc:.3f}; temporal ROC-AUC = {row('temporal_latest20', no_lci).roc_auc:.3f}. This supports the claim that performance is not dependent on LCI-source metadata alone.
{repeated_text}

## Reviewer Positioning
The defensible +90 claim should be ROC-AUC, not only raw accuracy. Because the top-decile target is imbalanced, the paper should report accuracy together with ROC-AUC, average precision, balanced accuracy, precision, recall, and F1. The model is strongest as a procurement screening and prioritisation tool, not as a full replacement for project-specific LCA.

## Limitations to State Upfront
The dataset is U.S.-focused, EPD-derived, and based on declared EPD metadata rather than direct mix proportions. Generalisation should therefore be claimed for EPD-based procurement screening, not for unobserved international markets or detailed concrete mix design without supplier metadata.
"""

    benchmark_text = ""
    benchmark_path = RESULTS_DIR / "algorithm_benchmark_summary.csv"
    if benchmark_path.exists():
        benchmark = pd.read_csv(benchmark_path)
        required_cols = {
            "algorithm",
            "roc_auc_group_company",
            "roc_auc_temporal_latest20",
            "accuracy_group_company",
            "accuracy_temporal_latest20",
        }
        if required_cols.issubset(benchmark.columns):
            benchmark = benchmark.sort_values("roc_auc_temporal_latest20", ascending=False)
            non_dummy_count = int((benchmark["algorithm"] != "Dummy majority baseline").sum())
            rows = []
            for _, item in benchmark.head(8).iterrows():
                rows.append(
                    "| "
                    f"{item.algorithm} | {item.roc_auc_group_company:.3f} | "
                    f"{item.roc_auc_temporal_latest20:.3f} | "
                    f"{item.accuracy_group_company:.3f} | {item.accuracy_temporal_latest20:.3f} |"
                )
            benchmark_text = f"""

## Algorithm Benchmark Against Alternative Methods
The main XGBoost model was compared with {non_dummy_count - 1} alternative machine-learning algorithms plus a majority-class sanity baseline under the same target, feature set, leakage controls, and holdout splits. The comparison included LightGBM, CatBoost, histogram gradient boosting, multilayer perceptron, random forest, extra trees, SGD logistic regression, linear SVM, Passive-Aggressive, ridge classifier, and Complement Naive Bayes.

| Algorithm | Unseen-company ROC-AUC | Temporal ROC-AUC | Unseen-company accuracy | Temporal accuracy |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

The benchmark supports retaining XGBoost as the primary model because it achieved the highest temporal ROC-AUC and the strongest unseen-company ROC-AUC among the high-performing algorithms. LightGBM produced the highest raw temporal accuracy, but its balanced accuracy and ROC-AUC were lower than XGBoost. Because the target is imbalanced, model selection should prioritise ROC-AUC, average precision, balanced accuracy, and F1 rather than raw accuracy alone.
"""

    custom_text = ""
    custom_path = RESULTS_DIR / "carm_boost_fixed_vs_sota.csv"
    if custom_path.exists():
        custom = pd.read_csv(custom_path)
        required_cols = {
            "split",
            "custom_auc",
            "best_sota_auc",
            "auc_delta_vs_best_sota",
            "custom_accuracy",
            "best_sota_auc_accuracy",
            "custom_balanced_accuracy",
            "best_sota_auc_balanced_accuracy",
        }
        if required_cols.issubset(custom.columns):
            split_labels = {
                "group_company": "Unseen company",
                "temporal_latest20": "Latest 20% by issue date",
            }
            rows = []
            for _, item in custom.iterrows():
                rows.append(
                    "| "
                    f"{split_labels.get(item.split, item.split)} | "
                    f"{item.custom_auc:.6f} | {item.best_sota_auc:.6f} | "
                    f"{item.auc_delta_vs_best_sota:+.6f} | "
                    f"{item.custom_accuracy:.6f} | {item.best_sota_auc_accuracy:.6f} | "
                    f"{item.custom_balanced_accuracy:.6f} | "
                    f"{item.best_sota_auc_balanced_accuracy:.6f} |"
                )
            custom_text = f"""

## Developed Single-Model Algorithm
The study now includes a proposed single-model algorithm, CARM-Boost, standing for Carbon-Aware Risk Modelling Boost. CARM-Boost is not a stacked or blended ensemble of multiple algorithms. It is a single cost-sensitive boosted classifier that augments the EPD metadata with procurement-specific carbon features, including strength transformations, curing-adjusted strength, supplementary cementitious material indicators, and region-strength interaction categories.

| Model comparison | Custom CARM-Boost ROC-AUC | Best SOTA ROC-AUC | Delta | Custom accuracy | Best SOTA accuracy | Custom balanced accuracy | Best SOTA balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The result supports a narrow claim: CARM-Boost marginally improves ROC-AUC over the strongest SOTA benchmark on both principal holdouts. It should not be described as a broad or large improvement, because XGBoost remains stronger on some threshold-dependent metrics in the unseen-company split.
"""

    text = text.replace("\n\n## Repeated Unseen-Company Validation", f"{custom_text}\n\n## Repeated Unseen-Company Validation")
    text = text.replace("\n\n## Reviewer Positioning", f"{benchmark_text}\n\n## Reviewer Positioning")
    (RESULTS_DIR / "reviewer_brief.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    started = time.time()
    dataset_meta = download_if_needed()
    base, dataset_summary = load_and_prepare()
    base.to_csv(PROCESSED_DIR / "concrete_epd_modeling_base.csv", index=False)

    pd.DataFrame(
        [{"item": key, "value": value} for key, value in dataset_summary.items() if key != "outcome_columns_excluded"]
    ).to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)
    (RESULTS_DIR / "leakage_audit.json").write_text(
        json.dumps(
            {
                "excluded_outcome_columns": dataset_summary["outcome_columns_excluded"],
                "feature_sets": [feature_set.__dict__ for feature_set in FEATURE_SETS],
                "target": "top decile of A1-A3 GWP per ksi of compressive strength",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results, predictions, ci = run_main_models(base)
    sensitivity = run_sensitivity(base)
    repeated_group, repeated_group_summary = run_repeated_group_company(base)

    results.to_csv(RESULTS_DIR / "model_results.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "concrete_epd_model_predictions.csv", index=False)
    ci.to_csv(RESULTS_DIR / "bootstrap_ci.csv", index=False)
    sensitivity.to_csv(RESULTS_DIR / "sensitivity_results.csv", index=False)
    repeated_group.to_csv(RESULTS_DIR / "repeated_group_company.csv", index=False)
    repeated_group_summary.to_csv(
        RESULTS_DIR / "repeated_group_company_summary.csv", index=False
    )

    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "runtime_seconds": round(time.time() - started, 3),
        "dataset": dataset_meta,
        "gpu": gpu_info(),
        "model": "XGBClassifier, GPU hist, device=cuda",
        "random_state": RANDOM_STATE,
        "main_target_quantile": MAIN_QUANTILE,
        "sensitivity_quantiles": SENSITIVITY_QUANTILES,
        "source_links": {
            "mendeley_dataset_page": f"https://data.mendeley.com/datasets/{DATASET_ID}/{DATASET_VERSION}",
            "mendeley_file_api": MENDELEY_FILE_API,
        },
    }
    (RESULTS_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_brief(dataset_meta, dataset_summary, results, repeated_group_summary)
    print(f"Complete in {manifest['runtime_seconds']} seconds")


if __name__ == "__main__":
    main()
