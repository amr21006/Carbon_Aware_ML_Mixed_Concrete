"""The hardest shift available inside the data: a new supplier in a future period.

The reported holdouts vary one thing at a time, a producer never seen or a period
never seen. Deployment presents both at once. This script evaluates that compound
condition, which is strictly harder than either single shift and is the closest
in-sample approximation to validation on an independent corpus.

  company_and_time   held-out producers, evaluated only on their later declarations
  new_entrants       producers whose first declaration falls after the cutoff
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import pipeline_common as R

SEEDS = [1, 2, 3, 4, 5]
CUTOFF_Q = 0.80


def main() -> None:
    ctx = R.get_context()
    dates = ctx.issue_date.fillna(pd.Timestamp("2100-01-01"))
    cutoff = dates.quantile(CUTOFF_Q)
    early = (dates <= cutoff).to_numpy()
    late = ~early
    print(f"[setup] cutoff {cutoff.date()} | early {early.sum()} | late {late.sum()}", flush=True)

    rows: list[dict[str, Any]] = []

    # --- design 1: held-out producers, evaluated only on their later declarations ---
    for seed in SEEDS:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        keep_idx, held_idx = next(splitter.split(np.arange(ctx.n), ctx.y, groups=ctx.company))
        held = np.zeros(ctx.n, dtype=bool); held[held_idx] = True
        train_idx = np.flatnonzero(early & ~held)
        test_idx = np.flatnonzero(late & held)
        if len(test_idx) < 300 or ctx.y.iloc[test_idx].sum() < 25:
            print(f"[skip] seed {seed}: test {len(test_idx)} rows", flush=True)
            continue
        t0 = time.time()
        proba = R.fit_predict(ctx.model_frame, ctx.y, ctx.numeric_cols, ctx.categorical_cols,
                              train_idx, test_idx, seed=seed)
        y_test = np.asarray(ctx.y.iloc[test_idx])
        row = {"design": "company_and_time", "replicate": seed,
               "n_train": int(len(train_idx)),
               "n_companies_test": int(ctx.company.iloc[test_idx].nunique()),
               "company_overlap": int(len(set(ctx.company.iloc[train_idx]) &
                                          set(ctx.company.iloc[test_idx])))}
        row.update(R.metric_row(y_test, proba))
        row["runtime_s"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"[compound] seed {seed}: n_test {row['n_test']} prev {row['prevalence']:.3f} "
              f"AUC {row['roc_auc']:.4f} cap@20 {row['top_20pct_recall_capture']:.4f} "
              f"overlap {row['company_overlap']}", flush=True)
        R.write_table(pd.DataFrame(rows), "compound_shift.csv")

    # --- design 2: producers whose first declaration falls after the cutoff ---
    first_seen = dates.groupby(ctx.company).transform("min")
    entrant = (first_seen > cutoff).to_numpy()
    train_idx = np.flatnonzero(early & ~entrant)
    test_idx = np.flatnonzero(late & entrant)
    print(f"[setup] new-entrant producers: {ctx.company[entrant].nunique()} "
          f"covering {len(test_idx)} declarations", flush=True)
    if len(test_idx) >= 200 and ctx.y.iloc[test_idx].sum() >= 20:
        proba = R.fit_predict(ctx.model_frame, ctx.y, ctx.numeric_cols, ctx.categorical_cols,
                              train_idx, test_idx)
        y_test = np.asarray(ctx.y.iloc[test_idx])
        row = {"design": "new_entrants", "replicate": 1, "n_train": int(len(train_idx)),
               "n_companies_test": int(ctx.company.iloc[test_idx].nunique()),
               "company_overlap": 0}
        row.update(R.metric_row(y_test, proba))
        rows.append(row)
        print(f"[compound] new entrants: n_test {row['n_test']} prev {row['prevalence']:.3f} "
              f"AUC {row['roc_auc']:.4f} cap@20 {row['top_20pct_recall_capture']:.4f}", flush=True)
    else:
        print(f"[compound] new-entrant design not evaluable: {len(test_idx)} declarations", flush=True)

    frame = pd.DataFrame(rows)
    R.write_table(frame, "compound_shift.csv")
    if not frame.empty:
        s = frame[frame.design == "company_and_time"]
        if not s.empty:
            summary = pd.DataFrame([{
                "design": "company_and_time", "replicates": len(s),
                "roc_auc_mean": s.roc_auc.mean(), "roc_auc_sd": s.roc_auc.std(),
                "roc_auc_min": s.roc_auc.min(), "roc_auc_max": s.roc_auc.max(),
                "capture20_mean": s.top_20pct_recall_capture.mean(),
                "capture20_sd": s.top_20pct_recall_capture.std(),
                "prevalence_mean": s.prevalence.mean(),
                "n_test_mean": s.n_test.mean(),
            }]).round(4)
            R.write_table(summary, "compound_shift_summary.csv")
            print("\n=== compound shift summary ===")
            print(summary.to_string(index=False))
    R.write_manifest({
        "analysis": "compound producer and period shift",
        "cutoff_quantile": CUTOFF_Q,
        "note": "Train uses only earlier declarations from producers absent from the test set; "
                "test uses only later declarations from producers absent from training.",
    }, "compound_shift_manifest.json")


if __name__ == "__main__":
    main()
