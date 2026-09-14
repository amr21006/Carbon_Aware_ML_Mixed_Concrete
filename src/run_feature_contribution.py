"""Isolate what the application-aware feature construction contributes.

The benchmark in Table 3 compares ACRM with a plain LightGBM, but those two
differ in both features and hyperparameters, so the comparison does not say
which of the two is responsible. This script holds the learner, the
hyperparameters, the target, the splits, and the leakage controls fixed and
varies only the feature set.

  base_metadata_only      the plain predictor set, no ACRM additions
  no_application_features ACRM minus the application-derived keys only
  full_acrm               the reported feature set
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

import pipeline_common as R
import concrete_epd_pipeline as M
import run_acrm_model as A

PUBLISHED = ["group_company", "group_epd_source", "temporal_latest20"]


class _FS:
    def __init__(self, include_lci_sources: bool) -> None:
        self.include_lci_sources = include_lci_sources


def application_columns(cols: list[str]) -> list[str]:
    """Columns that exist only because a mix's application is known."""
    keys = ("application", "app_", "curing_family", "operator_application")
    return [c for c in cols if any(k in c.lower() for k in keys)]


def main() -> None:
    ctx = R.get_context()
    splits = ctx.published_splits()
    base_num, base_cat, _ = M.feature_columns()
    base_num = [c for c in base_num if c in ctx.model_frame.columns]
    base_cat = [c for c in base_cat if c in ctx.model_frame.columns]

    # Text without the appended application tokens, for the base variant.
    base_text = M.build_text_feature(ctx.base, _FS(include_lci_sources=True)).reset_index(drop=True)

    variants: dict[str, dict[str, Any]] = {
        "full_acrm": {"num": ctx.numeric_cols, "cat": ctx.categorical_cols, "text": None},
        "no_application_features": {
            "num": [c for c in ctx.numeric_cols if c not in set(application_columns(ctx.numeric_cols))],
            "cat": [c for c in ctx.categorical_cols if c not in set(application_columns(ctx.categorical_cols))],
            "text": base_text,
        },
        "base_metadata_only": {"num": base_num, "cat": base_cat, "text": base_text},
    }

    rows: list[dict[str, Any]] = []
    for name, spec in variants.items():
        frame = ctx.model_frame.copy()
        if spec["text"] is not None:
            frame["text_feature"] = spec["text"].to_numpy()
        for split_name in PUBLISHED:
            tr, te = splits[split_name]
            t0 = time.time()
            proba = R.fit_predict(frame, ctx.y, spec["num"], spec["cat"], tr, te)
            y_test = np.asarray(ctx.y.iloc[te])
            row = {"variant": name, "split": split_name,
                   "n_numeric": len(spec["num"]), "n_categorical": len(spec["cat"])}
            row.update(R.metric_row(y_test, proba))
            row["runtime_s"] = round(time.time() - t0, 1)
            rows.append(row)
            print(f"[contrib] {name} / {split_name}: AUC {row['roc_auc']:.4f} "
                  f"AP {row['average_precision']:.4f} "
                  f"cap@20 {row['top_20pct_recall_capture']:.4f}", flush=True)
            R.write_table(pd.DataFrame(rows), "feature_contribution.csv")

    frame = pd.DataFrame(rows)
    base = frame[frame["variant"] == "base_metadata_only"].set_index("split")
    lines = []
    for split_name in PUBLISHED:
        f = frame[(frame["variant"] == "full_acrm") & (frame["split"] == split_name)].iloc[0]
        lines.append({
            "split": split_name,
            "auc_base": base.loc[split_name, "roc_auc"],
            "auc_acrm": f["roc_auc"],
            "auc_gain": f["roc_auc"] - base.loc[split_name, "roc_auc"],
            "ap_base": base.loc[split_name, "average_precision"],
            "ap_acrm": f["average_precision"],
            "ap_gain": f["average_precision"] - base.loc[split_name, "average_precision"],
            "cap20_base": base.loc[split_name, "top_20pct_recall_capture"],
            "cap20_acrm": f["top_20pct_recall_capture"],
            "cap20_gain": f["top_20pct_recall_capture"] - base.loc[split_name, "top_20pct_recall_capture"],
        })
    R.write_table(pd.DataFrame(lines).round(4), "feature_contribution_summary.csv")
    R.write_manifest({
        "analysis": "contribution of the application-aware feature construction",
        "held_fixed": "learner, hyperparameters, target, splits, leakage controls",
        "varied": "feature set only",
    }, "feature_contribution_manifest.json")


if __name__ == "__main__":
    main()
