"""Learning-to-rank experiment for top-20 concrete EPD screening.

The procurement decision is a ranking problem: reviewers often need a small
shortlist of high-risk mixes, not only a calibrated probability. This script
tests single-model LambdaMART-style rankers against the existing CARM feature
representation and reports top-k capture metrics.
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
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from run_concrete_epd_custom_single_model import CARM_BOOST_FIXED, custom_feature_frame, make_preprocessor
from run_concrete_epd_model import MAIN_QUANTILE, RESULTS_DIR, load_and_prepare, make_splits, make_target


ROOT = Path(__file__).resolve().parents[1]
V5_CSV = ROOT / "data" / "raw" / "concrete_epd_mendeley_v5.csv"
RANDOM_STATE = 42
SPLITS = ["group_company", "group_epd_source", "temporal_latest20"]
OUT_RESULTS = RESULTS_DIR / "concrete_epd_v5_p20_ranker_results.csv"
OUT_MANIFEST = RESULTS_DIR / "concrete_epd_v5_p20_ranker_manifest.json"


def query_ids(base: pd.DataFrame) -> pd.Series:
    region = base["U.S. Region of Plant"].fillna("missing").astype(str)
    strength = base["strength_bin_500"].fillna(-1).astype(str)
    curing = base["curing_days"].fillna(base["curing_days"].median()).round().astype(int).astype(str)
    return region + "|" + strength + "|" + curing


def sort_by_qid(x: sparse.spmatrix, y: np.ndarray, qid: np.ndarray) -> tuple[sparse.spmatrix, np.ndarray, list[int]]:
    order = np.argsort(qid, kind="mergesort")
    qid_sorted = qid[order]
    _, counts = np.unique(qid_sorted, return_counts=True)
    return x[order], y[order], counts.astype(int).tolist()


def top_fraction_metrics(y_true: np.ndarray, score: np.ndarray, fraction: float) -> dict[str, float]:
    n_select = max(1, int(np.ceil(len(y_true) * fraction)))
    selected = np.argsort(score)[-n_select:]
    positives = float(y_true.sum())
    selected_positive = float(y_true[selected].sum())
    precision = selected_positive / n_select
    capture = selected_positive / positives if positives else np.nan
    lift = precision / float(y_true.mean()) if float(y_true.mean()) else np.nan
    return {
        f"top_{int(fraction * 100)}pct_precision": precision,
        f"top_{int(fraction * 100)}pct_recall_capture": capture,
        f"top_{int(fraction * 100)}pct_lift": lift,
    }


def evaluate_scores(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "average_precision": float(average_precision_score(y_true, score)),
    }
    for fraction in [0.05, 0.10, 0.20, 0.30]:
        metrics.update(top_fraction_metrics(y_true, score, fraction))
    return metrics


def fit_lgbm_ranker(x_train: sparse.spmatrix, y_train: np.ndarray, group: list[int], x_test: sparse.spmatrix) -> np.ndarray:
    from lightgbm import LGBMRanker

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=420,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=25,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        device_type="gpu",
        verbose=-1,
    )
    try:
        model.fit(x_train, y_train, group=group)
    except Exception:
        model.set_params(device_type="cpu")
        model.fit(x_train, y_train, group=group)
    return model.predict(x_test)


def fit_xgb_ranker(x_train: sparse.spmatrix, y_train: np.ndarray, group: list[int], x_test: sparse.spmatrix) -> np.ndarray:
    from xgboost import XGBRanker

    model = XGBRanker(
        objective="rank:pairwise",
        n_estimators=420,
        max_depth=5,
        learning_rate=0.035,
        min_child_weight=1.0,
        subsample=0.90,
        colsample_bytree=0.88,
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        reg_lambda=1.5,
        eval_metric="ndcg",
    )
    model.fit(x_train, y_train, group=group, verbose=False)
    return model.predict(x_test)


def run(algorithms: list[str], splits_to_run: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    os.environ["CONCRETE_EPD_CSV"] = str(V5_CSV)
    base, dataset_summary = load_and_prepare()
    y = make_target(base, MAIN_QUANTILE).reset_index(drop=True)
    splits = make_splits(base, y)
    model_frame, numeric_cols, categorical_cols = custom_feature_frame(base)
    qid = query_ids(base).to_numpy()

    rows: list[dict[str, Any]] = []
    for split_name in splits_to_run:
        train_idx, test_idx = splits[split_name]
        preprocessor = make_preprocessor(CARM_BOOST_FIXED, numeric_cols, categorical_cols)
        started = time.time()
        x_train_raw = preprocessor.fit_transform(model_frame.iloc[train_idx])
        x_test = preprocessor.transform(model_frame.iloc[test_idx])
        x_train, y_train, group = sort_by_qid(
            x_train_raw,
            y.iloc[train_idx].to_numpy(),
            qid[train_idx],
        )
        y_test = y.iloc[test_idx].to_numpy()
        prep_seconds = time.time() - started

        for algorithm in algorithms:
            started = time.time()
            if algorithm == "lgbm":
                score = fit_lgbm_ranker(x_train, y_train, group, x_test)
                name = "P20-CARM LambdaMART LightGBM ranker"
            elif algorithm == "xgb":
                score = fit_xgb_ranker(x_train, y_train, group, x_test)
                name = "P20-CARM XGBoost pairwise ranker"
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            rows.append(
                {
                    "dataset_version": 5,
                    "algorithm": name,
                    "split": split_name,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_positive_rate": float(y_test.mean()),
                    "query_group_count": int(len(group)),
                    "median_query_group_size": float(np.median(group)),
                    "prep_seconds": round(prep_seconds, 3),
                    "fit_seconds": round(time.time() - started, 3),
                    **evaluate_scores(y_test, score),
                }
            )
            print(
                f"{name} | {split_name} | "
                f"AUC={rows[-1]['roc_auc']:.4f} AP={rows[-1]['average_precision']:.4f} "
                f"top20={rows[-1]['top_20pct_recall_capture']:.4f}",
                flush=True,
            )

    return pd.DataFrame(rows), dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["lgbm", "xgb", "all"], default="lgbm")
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algorithms = ["lgbm", "xgb"] if args.algorithm == "all" else [args.algorithm]
    splits_to_run = SPLITS if args.split == "all" else [args.split]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results, dataset_summary = run(algorithms, splits_to_run)
    results.to_csv(OUT_RESULTS, index=False)
    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "runtime_seconds": round(time.time() - started, 3),
        "dataset_version": 5,
        "csv_path": str(V5_CSV),
        "target": "top decile of A1-A3 GWP per ksi of compressive strength",
        "target_quantile": MAIN_QUANTILE,
        "not_an_ensemble_of_algorithms": True,
        "algorithm_option": args.algorithm,
        "split_option": args.split,
        "query_definition": "U.S. region | 500-psi strength bin | curing days",
        "dataset_rows_after_trim": dataset_summary["model_rows_after_trim"],
        "outputs": [str(OUT_RESULTS)],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(results.to_csv(index=False))


if __name__ == "__main__":
    main()
