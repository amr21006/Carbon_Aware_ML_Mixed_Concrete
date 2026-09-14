"""Does a ranking objective suit the task better than binary classification?

Section 4.1 defines the task as ranked retrieval under a review budget, but the
model minimizes a binary cross-entropy loss. This script tests whether aligning
the loss with the stated task helps.

Protocol, fixed before the holdouts are touched:
  1. Candidates are compared by three-fold cross-validation grouped by producer
     company, inside the training partition of the temporal holdout only.
  2. The selection criterion is capture at a 20% review budget, which is the
     quantity the paper argues from.
  3. The selected candidate is evaluated once on each reported holdout.
  4. The outcome is reported whichever way it falls.

Calibration is measured as well as capture, because a ranking objective returns
scores rather than probabilities, and the budget-as-probability-cutoff argument
in Section 5.5 depends on calibrated output.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

import pipeline_common as R
import run_acrm_model as A

INNER_FOLDS = 3
# LambdaRank caps a query group at 10,000 rows. The task is a single global
# ranking of the pool, so the training rows are split into random chunks below
# that cap; random assignment makes the within-chunk comparisons an unbiased
# sample of the global ones.
GROUP_CHUNK = 8000
BUDGET = 0.20
PUBLISHED = ["group_company", "group_epd_source", "temporal_latest20"]

CANDIDATES = {
    "binary_crossentropy_reported": {"kind": "classifier"},
    "lambdarank": {"kind": "ranker", "objective": "lambdarank"},
    "rank_xendcg": {"kind": "ranker", "objective": "rank_xendcg"},
}


def capture_at(y: np.ndarray, score: np.ndarray, frac: float = BUDGET) -> float:
    k = max(int(round(len(y) * frac)), 1)
    order = np.argsort(-score, kind="stable")[:k]
    pos = float(y.sum())
    return float(y[order].sum() / pos) if pos else float("nan")


def fit_score(frame, y, num, cat, train_idx, test_idx, spec, seed=R.BASE_SEED):
    """Return raw scores on test_idx. Preprocessing is fitted on train only."""
    args = R.acrm_args()
    pre = A.make_preprocessor(num, cat, [], max_text_features=args.max_text_features)
    x_tr = pre.fit_transform(frame.iloc[train_idx])
    x_te = pre.transform(frame.iloc[test_idx])
    if not sparse.issparse(x_tr):
        x_tr, x_te = sparse.csr_matrix(x_tr), sparse.csr_matrix(x_te)
    y_tr = np.asarray(y.iloc[train_idx])

    if spec["kind"] == "classifier":
        model = A.make_model(y_tr, args)
        model.set_params(random_state=seed)
        model = A.fit_model(model, x_tr, y_tr, args)
        return model.predict_proba(x_te)[:, 1], True

    from lightgbm import LGBMRanker
    ranker = LGBMRanker(
        objective=spec["objective"],
        n_estimators=args.n_estimators, learning_rate=args.learning_rate,
        num_leaves=args.num_leaves, max_depth=args.max_depth,
        min_child_samples=args.min_child_samples, subsample=args.subsample, subsample_freq=1,
        colsample_bytree=args.colsample_bytree, reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda, label_gain=[0, 1],
        lambdarank_truncation_level=50, device_type="gpu",
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    n = x_tr.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    x_tr, y_tr = x_tr[perm], y_tr[perm]
    groups = [GROUP_CHUNK] * (n // GROUP_CHUNK)
    if n % GROUP_CHUNK:
        groups.append(n % GROUP_CHUNK)
    try:
        ranker.fit(x_tr, y_tr, group=groups)
    except Exception:
        ranker.set_params(device_type="cpu")
        ranker.fit(x_tr, y_tr, group=groups)
    return ranker.predict(x_te), False


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=",".join(CANDIDATES))
    cli = ap.parse_args()
    wanted = [c.strip() for c in cli.candidates.split(",") if c.strip()]
    for k in list(CANDIDATES):
        if k not in wanted:
            CANDIDATES.pop(k)
    ctx = R.get_context()
    splits = ctx.published_splits()
    outer_train = np.asarray(splits["temporal_latest20"][0])
    groups = ctx.company.iloc[outer_train].to_numpy()
    folds = list(GroupKFold(n_splits=INNER_FOLDS).split(outer_train, ctx.y.iloc[outer_train], groups))

    print(f"[protocol] selection on {len(outer_train)} training declarations, "
          f"{INNER_FOLDS} folds grouped by company, criterion capture@{int(BUDGET*100)}%\n", flush=True)

    inner: list[dict[str, Any]] = []
    for name, spec in CANDIDATES.items():
        caps, aucs = [], []
        t0 = time.time()
        for fi, (tr, va) in enumerate(folds, 1):
            s, _ = fit_score(ctx.model_frame, ctx.y, ctx.numeric_cols, ctx.categorical_cols,
                             outer_train[tr], outer_train[va], spec)
            yv = np.asarray(ctx.y.iloc[outer_train[va]])
            caps.append(capture_at(yv, s)); aucs.append(roc_auc_score(yv, s))
        row = {"candidate": name, "inner_capture20_mean": float(np.mean(caps)),
               "inner_capture20_sd": float(np.std(caps)),
               "inner_auc_mean": float(np.mean(aucs)),
               "inner_folds": ";".join(f"{c:.4f}" for c in caps),
               "runtime_s": round(time.time() - t0, 1)}
        inner.append(row)
        print(f"[inner] {name:30s} capture@20 {row['inner_capture20_mean']:.4f} "
              f"+/- {row['inner_capture20_sd']:.4f}  AUC {row['inner_auc_mean']:.4f} "
              f"({row['runtime_s']:.0f}s)", flush=True)
        R.write_table(pd.DataFrame(inner), "ranking_objective_inner.csv")

    tab = pd.DataFrame(inner).sort_values("inner_capture20_mean", ascending=False)
    winner = tab.iloc[0]["candidate"]
    reported = "binary_crossentropy_reported"
    print(f"\n[selection] winner on inner CV: {winner}")
    print(f"[selection] reported objective ranked "
          f"{int(tab.reset_index().index[tab.reset_index().candidate == reported][0]) + 1} of {len(tab)}\n",
          flush=True)

    # Evaluate the reported objective and the inner-CV winner once on each holdout.
    to_eval = {reported} | {winner}
    outer: list[dict[str, Any]] = []
    for name in to_eval:
        spec = CANDIDATES[name]
        for split_name in PUBLISHED:
            tr, te = splits[split_name]
            s, is_prob = fit_score(ctx.model_frame, ctx.y, ctx.numeric_cols,
                                   ctx.categorical_cols, tr, te, spec)
            yt = np.asarray(ctx.y.iloc[te])
            row = {"candidate": name, "split": split_name, "outputs_probability": is_prob,
                   "roc_auc": roc_auc_score(yt, s),
                   "average_precision": average_precision_score(yt, s),
                   "capture_10pct": capture_at(yt, s, 0.10),
                   "capture_20pct": capture_at(yt, s, BUDGET)}
            if is_prob:
                row["ece_10_bins"] = A.expected_calibration_error(yt, s, bins=10)
                row["brier"] = brier_score_loss(yt, s)
            outer.append(row)
            print(f"[holdout] {name:30s} {split_name:18s} AUC {row['roc_auc']:.4f} "
                  f"cap@20 {row['capture_20pct']:.4f}", flush=True)
            R.write_table(pd.DataFrame(outer), "ranking_objective_holdout.csv")

    R.write_manifest({
        "analysis": "ranking objective against binary classification",
        "selection": "3-fold company-grouped CV inside the temporal training partition",
        "criterion": f"capture at a {int(BUDGET*100)}% review budget",
        "inner_cv_winner": winner,
        "protocol_note": "Selection preceded any holdout evaluation; the outcome is reported "
                         "regardless of direction.",
    }, "ranking_objective_manifest.json")


if __name__ == "__main__":
    main()
