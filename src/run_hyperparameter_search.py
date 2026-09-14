"""Documented hyperparameter selection for ACRM (Reviewer 1, comment 2).

The reported configuration was fixed a priori. This script performs a randomised
search scored by grouped inner cross-validation *inside the training partition
only*, so no test record participates in selection. The selected configuration is
then evaluated once on each of the three reported holdouts.

Search design
  outer   temporal split (train on the earliest 80% of issue dates)
  inner   3-fold GroupKFold by producer company within that training partition
  scoring average precision, because the target is imbalanced
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

import pipeline_common as R

N_CONFIGS = 15
INNER_FOLDS = 3
PUBLISHED = ["group_company", "group_epd_source", "temporal_latest20"]

SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [400, 650, 900],
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.015, 0.028, 0.05],
    "min_child_samples": [10, 20, 40],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.75, 0.86],
    "reg_alpha": [0.0, 0.05, 0.5],
    "reg_lambda": [0.5, 2.0, 8.0],
    "max_text_features": [8000, 14000, 20000],
}


def sample_configs(rng: np.random.Generator, n: int) -> list[dict[str, Any]]:
    """The reported configuration is evaluated first so the search is a fair
    comparison against it rather than a replacement for it."""
    configs = [{k: R.ACRM_PARAMS[k] for k in SEARCH_SPACE}]
    seen = {tuple(sorted(configs[0].items()))}
    while len(configs) < n:
        candidate = {k: v[rng.integers(0, len(v))] for k, v in SEARCH_SPACE.items()}
        key = tuple(sorted(candidate.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append(candidate)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=int, default=N_CONFIGS)
    parser.add_argument("--folds", type=int, default=INNER_FOLDS)
    args = parser.parse_args()

    ctx = R.get_context()
    splits = ctx.published_splits()
    outer_train, _ = splits["temporal_latest20"]
    outer_train = np.asarray(outer_train)

    groups = ctx.company.iloc[outer_train].to_numpy()
    y_train = ctx.y.iloc[outer_train]
    folds = list(GroupKFold(n_splits=args.folds).split(outer_train, y_train, groups=groups))

    rng = np.random.default_rng(R.BASE_SEED)
    configs = sample_configs(rng, args.configs)
    print(f"[hpsearch] {len(configs)} configurations x {args.folds} inner folds "
          f"on {len(outer_train)} training records", flush=True)

    rows: list[dict[str, Any]] = []
    for ci, config in enumerate(configs, start=1):
        scores: list[float] = []
        t0 = time.time()
        for fi, (tr, va) in enumerate(folds, start=1):
            model_args = R.acrm_args(**config)
            proba = R.fit_predict(
                ctx.model_frame, ctx.y, ctx.numeric_cols, ctx.categorical_cols,
                outer_train[tr], outer_train[va], args=model_args,
            )
            scores.append(float(average_precision_score(np.asarray(ctx.y.iloc[outer_train[va]]), proba)))
        row = {
            "config_id": ci,
            "is_reported_configuration": ci == 1,
            **config,
            "inner_ap_mean": float(np.mean(scores)),
            "inner_ap_std": float(np.std(scores)),
            "inner_ap_folds": ";".join(f"{s:.4f}" for s in scores),
            "runtime_s": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(f"[hpsearch] config {ci}/{len(configs)}: inner AP "
              f"{row['inner_ap_mean']:.4f} +/- {row['inner_ap_std']:.4f} "
              f"({row['runtime_s']:.0f}s)", flush=True)
        R.write_table(pd.DataFrame(rows), "hpsearch_configurations.csv")

    frame = pd.DataFrame(rows).sort_values("inner_ap_mean", ascending=False)
    best = frame.iloc[0]
    selected = {k: best[k] for k in SEARCH_SPACE}
    print(f"\n[hpsearch] selected config {int(best.config_id)} "
          f"(reported configuration ranked "
          f"{int(frame.reset_index().index[frame.reset_index().config_id == 1][0]) + 1} "
          f"of {len(frame)})", flush=True)

    holdout_rows: list[dict[str, Any]] = []
    for split_name in PUBLISHED:
        tr, te = splits[split_name]
        proba = R.fit_predict(
            ctx.model_frame, ctx.y, ctx.numeric_cols, ctx.categorical_cols, tr, te,
            args=R.acrm_args(**{k: selected[k] for k in SEARCH_SPACE}),
        )
        row = {"split": split_name, "configuration": "inner-CV selected"}
        row.update(R.metric_row(np.asarray(ctx.y.iloc[te]), proba))
        holdout_rows.append(row)
        print(f"[hpsearch] holdout {split_name}: AUC {row['roc_auc']:.4f} "
              f"AP {row['average_precision']:.4f} "
              f"cap@20 {row['top_20pct_recall_capture']:.4f}", flush=True)
        R.write_table(pd.DataFrame(holdout_rows), "hpsearch_selected_holdouts.csv")

    R.write_manifest({
        "analysis": "grouped inner-CV randomised hyperparameter search",
        "search_space": SEARCH_SPACE,
        "n_configurations": len(configs),
        "inner_folds": args.folds,
        "inner_grouping": "producer company",
        "scoring": "average precision",
        "outer_training_partition": "temporal split, earliest 80% of issue dates",
        "selected_configuration": {k: (int(v) if isinstance(v, (np.integer,)) else float(v)
                                       if isinstance(v, (np.floating,)) else v)
                                   for k, v in selected.items()},
        "note": "Selection used training records only; no holdout record entered the search.",
    }, "hpsearch_manifest.json")


if __name__ == "__main__":
    main()
