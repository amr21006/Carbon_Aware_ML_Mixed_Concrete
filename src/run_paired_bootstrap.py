"""Paired bootstrap of the reported model against the XGBoost baseline.

The published paired comparison resampled CARMBoost, one of the two
domain-feature variants, so it did not test the model the manuscript reports.
Both sets of per-record probabilities were saved on identical test partitions,
so the comparison for ACRM is obtained by resampling those stored predictions
under the same protocol: 500 row-level resamples, seed 42, percentile
intervals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, roc_auc_score)

import pipeline_common as R

RESULTS = R.RESULTS_DIR
RANDOM_STATE = 42
REPEATS = 500
SPLITS = {"group_company": "Unseen company",
          "group_epd_source": "Unseen EPD source",
          "temporal_latest20": "Temporal (final 20%)"}


def paired_bootstrap(y_true, a_proba, b_proba, repeats=REPEATS, cluster=None):
    """Row-level resampling by default; cluster resampling when groups are given."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    if cluster is not None:
        keys = pd.unique(cluster)
        index = {k: np.flatnonzero(cluster == k) for k in keys}
    rows = []
    for _ in range(repeats):
        if cluster is None:
            s = rng.integers(0, n, size=n)
        else:
            picked = rng.integers(0, len(keys), size=len(keys))
            s = np.concatenate([index[keys[i]] for i in picked])
        ys = y_true[s]
        if len(np.unique(ys)) < 2:
            continue
        a, b = a_proba[s], b_proba[s]
        rows.append({
            "auc_delta": roc_auc_score(ys, a) - roc_auc_score(ys, b),
            "ap_delta": average_precision_score(ys, a) - average_precision_score(ys, b),
            "accuracy_delta": accuracy_score(ys, a >= 0.5) - accuracy_score(ys, b >= 0.5),
            "balanced_accuracy_delta": balanced_accuracy_score(ys, a >= 0.5)
            - balanced_accuracy_score(ys, b >= 0.5),
            "f1_delta": f1_score(ys, a >= 0.5, zero_division=0)
            - f1_score(ys, b >= 0.5, zero_division=0),
        })
    boot = pd.DataFrame(rows)
    out = {}
    for col in boot.columns:
        out[f"{col}_mean"] = float(boot[col].mean())
        out[f"{col}_ci_low"] = float(boot[col].quantile(0.025))
        out[f"{col}_ci_high"] = float(boot[col].quantile(0.975))
        out[f"{col}_p_gt_0"] = float((boot[col] > 0).mean())
    return out


def main() -> None:
    acrm = pd.read_csv(RESULTS / "acrm_single_model_predictions.csv")
    val = pd.read_csv(RESULTS / "validation_predictions.csv")
    m = acrm.merge(val[["split", "row_index", "y_true", "xgb_probability",
                        "carm_probability"]],
                   on=["split", "row_index"], suffixes=("", "_v"))
    assert (m["y_true"] == m["y_true_v"]).all(), "label mismatch between exports"
    assert len(m) == len(acrm) == len(val), "prediction exports do not align"

    rows = []
    for key, label in SPLITS.items():
        g = m[m.split == key]
        y = g["y_true"].to_numpy()
        xgb = g["xgb_probability"].to_numpy()
        comp = g["company"].to_numpy()
        for name, proba in (("ACRM", g["acrm_probability"].to_numpy()),
                            ("CARMBoost", g["carm_probability"].to_numpy())):
          for scheme, cl in (("declaration", None), ("company", comp)):
            st = paired_bootstrap(y, proba, xgb, cluster=cl)
            rows.append({
                "model": name, "resampling": scheme,
                "split": key, "validation": label,
                "n_test": len(g),
                "model_auc": roc_auc_score(y, proba),
                "xgb_auc": roc_auc_score(y, xgb),
                "model_ap": average_precision_score(y, proba),
                "xgb_ap": average_precision_score(y, xgb),
                **st,
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "paired_bootstrap_acrm.csv", index=False)

    for name in ("CARMBoost", "ACRM"):
        print(f"\n{name} vs XGBoost baseline (500 resamples, seed 42)")
        for _, r in out[out.model == name].iterrows():
            print(f"  {r.validation:22s} AUC {r.model_auc:.4f} vs {r.xgb_auc:.4f}  "
                  f"d={r.auc_delta_mean:+.4f} [{r.auc_delta_ci_low:+.4f}, "
                  f"{r.auc_delta_ci_high:+.4f}] P={r.auc_delta_p_gt_0:.3f}   "
                  f"AP d={r.ap_delta_mean:+.4f} [{r.ap_delta_ci_low:+.4f}, "
                  f"{r.ap_delta_ci_high:+.4f}]")
    print(f"\nwrote {RESULTS / 'paired_bootstrap_acrm.csv'}")


if __name__ == "__main__":
    main()
