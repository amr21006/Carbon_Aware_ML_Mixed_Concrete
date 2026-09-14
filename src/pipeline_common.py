"""Shared machinery for the JCCE revision analyses.

Wraps the published ACRM pipeline so that the revision experiments reuse the
exact preprocessing, feature construction, and learner configuration of the
reported run. Nothing here changes the primary model; it only re-executes it
under alternative targets, splits, seeds, feature subsets, and resampling
schemes requested by the reviewers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupShuffleSplit

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import concrete_epd_pipeline as M  # noqa: E402
import run_acrm_model as A  # noqa: E402

RESULTS_DIR = Path(M.RESULTS_DIR)
MAIN_QUANTILE = M.MAIN_QUANTILE
# The reported study uses Mendeley dataset version 5. concrete_epd_pipeline
# selects the CSV through this environment variable and otherwise falls back to
# the earlier version, so it must be set before load_and_prepare is called.
V5_CSV = Path(A.V5_CSV)
BASE_SEED = 42
REVIEW_FRACTIONS = [0.05, 0.10, 0.20, 0.30]

# Configuration of the published ACRM run (results/concrete_epd_v5_acrm_single_model_manifest.json).
ACRM_PARAMS: dict[str, Any] = {
    "splits": "group_company,group_epd_source,temporal_latest20",
    "learner": "lgbm",
    "n_estimators": 650,
    "max_depth": -1,
    "num_leaves": 63,
    "min_child_samples": 20,
    "learning_rate": 0.028,
    "min_child_weight": 2.0,
    "subsample": 0.90,
    "colsample_bytree": 0.86,
    "reg_alpha": 0.05,
    "reg_lambda": 2.0,
    "gamma": 0.02,
    "max_text_features": 14000,
    "prior_smoothing": 40.0,
    "prior_folds": 5,
    "disable_priors": True,
    "threshold_policy": "fixed_05",
}


def acrm_args(**overrides: Any) -> argparse.Namespace:
    params = dict(ACRM_PARAMS)
    params.update(overrides)
    return argparse.Namespace(**params)


# --------------------------------------------------------------------------
# Data context
# --------------------------------------------------------------------------

class Context:
    """Prepared modelling frame plus the grouping keys the revision needs."""

    def __init__(self) -> None:
        os.environ["CONCRETE_EPD_CSV"] = str(V5_CSV)
        base, summary = M.load_and_prepare()
        self.base = base.reset_index(drop=True)
        self.summary = summary
        frame, numeric_cols, categorical_cols, prior_cols = A.enriched_feature_frame(self.base)
        self.model_frame = frame.reset_index(drop=True)
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.prior_cols = prior_cols
        self.y = M.make_target(self.base, MAIN_QUANTILE).reset_index(drop=True)

        b = self.base
        self.company = b["Company"].fillna("missing").astype(str)
        self.plant = (
            b["Company"].fillna("missing").astype(str)
            + "||"
            + b["Plant"].fillna("missing").astype(str)
        )
        self.source = b["EPD Source Link"].fillna("missing").astype(str)
        self.operator = b["EPD Program Operator"].fillna("missing").astype(str)
        self.region = b["U.S. Region of Plant"].fillna("missing").astype(str)
        self.strength = pd.to_numeric(b["strength_psi"], errors="coerce")
        self.strength_bin = b["strength_bin_500"].astype(str)
        self.gwp = pd.to_numeric(b[M.TARGET_COL], errors="coerce")
        self.gwp_per_ksi = pd.to_numeric(b["gwp_per_ksi"], errors="coerce")
        self.issue_date = pd.to_datetime(b["EPD Date of Issue"], errors="coerce")
        self.issue_year = pd.to_numeric(b["issue_year"], errors="coerce")
        self.n = len(self.base)

    # -- splits -----------------------------------------------------------

    def group_split(self, groups: pd.Series, seed: int, test_size: float = 0.2):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(np.arange(self.n), self.y, groups=groups))
        return np.asarray(train_idx), np.asarray(test_idx)

    def temporal_split(self, test_fraction: float = 0.2, end_fraction: float = 1.0):
        """Rolling-origin split: train on the earliest records, test on the
        window immediately after, ending at ``end_fraction`` of the ordered
        sample. ``end_fraction=1.0`` reproduces the published temporal holdout.
        """
        order = np.argsort(
            self.issue_date.fillna(pd.Timestamp("2100-01-01")).to_numpy(), kind="stable"
        )
        n_end = int(round(self.n * end_fraction))
        n_test = int(round(self.n * test_fraction))
        n_test = max(n_test, 1)
        test_idx = order[max(n_end - n_test, 0) : n_end]
        train_idx = order[: max(n_end - n_test, 0)]
        return np.asarray(train_idx), np.asarray(test_idx)

    def published_splits(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """The three holdouts exactly as reported in the manuscript."""
        return M.make_splits(self.base, self.y)


_CONTEXT: Context | None = None


def get_context() -> Context:
    global _CONTEXT
    if _CONTEXT is None:
        t0 = time.time()
        _CONTEXT = Context()
        print(f"[context] prepared {_CONTEXT.n} rows in {time.time() - t0:.1f}s", flush=True)
    return _CONTEXT


# --------------------------------------------------------------------------
# Fit / evaluate
# --------------------------------------------------------------------------

def fit_predict(
    model_frame: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace | None = None,
    seed: int = BASE_SEED,
) -> np.ndarray:
    """Fit ACRM on ``train_idx`` and return high-carbon probabilities for
    ``test_idx``. All preprocessing is fitted on the training rows only."""
    args = args or acrm_args()
    pre = A.make_preprocessor(
        numeric_cols, categorical_cols, [], max_text_features=args.max_text_features
    )
    x_train = pre.fit_transform(model_frame.iloc[train_idx])
    x_test = pre.transform(model_frame.iloc[test_idx])
    if not sparse.issparse(x_train):
        x_train = sparse.csr_matrix(x_train)
        x_test = sparse.csr_matrix(x_test)
    y_train = np.asarray(y.iloc[train_idx])
    model = A.make_model(y_train, args)
    if hasattr(model, "set_params"):
        try:
            model.set_params(random_state=seed)
        except Exception:
            pass
    model = A.fit_model(model, x_train, y_train, args)
    return model.predict_proba(x_test)[:, 1]


def metric_row(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    row = dict(A.evaluate(y_true, proba, threshold))
    row["ece_10_bins"] = A.expected_calibration_error(y_true, proba, bins=10)
    # topk_metrics already prefixes its keys with the review fraction.
    for frac in REVIEW_FRACTIONS:
        row.update(A.topk_metrics(y_true, proba, frac))
    row["n_test"] = int(len(y_true))
    row["prevalence"] = float(np.mean(y_true))
    return row


def topk_capture(y_true: np.ndarray, score: np.ndarray, frac: float) -> float:
    n = len(y_true)
    k = max(int(round(n * frac)), 1)
    order = np.argsort(-score, kind="stable")[:k]
    pos = float(np.sum(y_true))
    return float(np.sum(y_true[order]) / pos) if pos > 0 else float("nan")


def topk_precision(y_true: np.ndarray, score: np.ndarray, frac: float) -> float:
    n = len(y_true)
    k = max(int(round(n * frac)), 1)
    order = np.argsort(-score, kind="stable")[:k]
    return float(np.mean(y_true[order]))


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

def write_table(frame: pd.DataFrame, name: str) -> Path:
    path = RESULTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"[write] {path}  ({len(frame)} rows)", flush=True)
    return path


def write_manifest(payload: dict[str, Any], name: str) -> Path:
    path = RESULTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("created_utc", pd.Timestamp.utcnow().isoformat())
    payload.setdefault("acrm_parameters", ACRM_PARAMS)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[write] {path}", flush=True)
    return path
