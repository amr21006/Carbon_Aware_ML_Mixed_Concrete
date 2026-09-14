"""Predict the numerator, divide by the known denominator.

The label is an inequality on GWP divided by compressive strength, and strength
is observed at screening time. A classifier must therefore learn a division
whose denominator it already sees. An alternative is to regress GWP and apply
the division exactly:

    high carbon  <=>  GWP >= tau * strength / 1000

Because the inequality is on a quantity the regression models directly, a
calibrated probability follows from the conditional residual spread,

    P(high carbon | x) = 1 - Phi( ( z(x) - mu(x) ) / sigma )

with z(x) = log1p(tau * strength / 1000), mu(x) the regression mean in log
space, and sigma estimated on held-in folds. The formulation therefore keeps
the probability that Section 5.5 depends on, unlike a ranking objective.

Protocol matches the ranking experiment: selection on company-grouped inner
folds inside the temporal training partition, then one evaluation per holdout,
reported either way.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

import pipeline_common as R
import run_acrm_model as A

INNER_FOLDS = 3
BUDGET = 0.20
TAU = 124.0  # 90th percentile of carbon intensity, as reported
PUBLISHED = ["group_company", "group_epd_source", "temporal_latest20"]


def capture_at(y: np.ndarray, s: np.ndarray, frac: float = BUDGET) -> float:
    k = max(int(round(len(y) * frac)), 1)
    order = np.argsort(-s, kind="stable")[:k]
    pos = float(y.sum())
    return float(y[order].sum() / pos) if pos else float("nan")


def _prep(frame, num, cat, tr, te):
    args = R.acrm_args()
    pre = A.make_preprocessor(num, cat, [], max_text_features=args.max_text_features)
    x_tr = pre.fit_transform(frame.iloc[tr])
    x_te = pre.transform(frame.iloc[te])
    if not sparse.issparse(x_tr):
        x_tr, x_te = sparse.csr_matrix(x_tr), sparse.csr_matrix(x_te)
    return x_tr, x_te, args


def fit_classifier(ctx, frame, tr, te, seed=R.BASE_SEED):
    x_tr, x_te, args = _prep(frame, ctx.numeric_cols, ctx.categorical_cols, tr, te)
    y_tr = np.asarray(ctx.y.iloc[tr])
    m = A.make_model(y_tr, args)
    m.set_params(random_state=seed)
    m = A.fit_model(m, x_tr, y_tr, args)
    return m.predict_proba(x_te)[:, 1]


def fit_regressor(ctx, frame, tr, te, seed=R.BASE_SEED):
    """Regress log GWP, then convert to a probability through the inequality."""
    from lightgbm import LGBMRegressor
    x_tr, x_te, args = _prep(frame, ctx.numeric_cols, ctx.categorical_cols, tr, te)
    g_tr = np.log1p(ctx.gwp.iloc[tr].to_numpy())

    model = LGBMRegressor(
        objective="regression", n_estimators=args.n_estimators,
        learning_rate=args.learning_rate, num_leaves=args.num_leaves,
        max_depth=args.max_depth, min_child_samples=args.min_child_samples,
        subsample=args.subsample, subsample_freq=1,
        colsample_bytree=args.colsample_bytree, reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda, device_type="gpu",
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    try:
        model.fit(x_tr, g_tr)
    except Exception:
        model.set_params(device_type="cpu")
        model.fit(x_tr, g_tr)

    # Residual spread from held-in folds, so sigma never sees the test partition.
    inner = GroupKFold(n_splits=3).split(np.arange(len(tr)), g_tr,
                                         ctx.company.iloc[tr].to_numpy())
    resid = []
    for a, b in inner:
        mi = LGBMRegressor(**model.get_params())
        try:
            mi.fit(x_tr[a], g_tr[a])
        except Exception:
            mi.set_params(device_type="cpu"); mi.fit(x_tr[a], g_tr[a])
        resid.append(g_tr[b] - mi.predict(x_tr[b]))
    sigma = float(np.std(np.concatenate(resid)))

    mu = model.predict(x_te)
    strength = ctx.strength.iloc[te].to_numpy().clip(min=1.0)
    z = np.log1p(TAU * strength / 1000.0)
    prob = 1.0 - stats.norm.cdf((z - mu) / max(sigma, 1e-6))
    return prob, sigma


def main() -> None:
    ctx = R.get_context()
    splits = ctx.published_splits()
    outer_train = np.asarray(splits["temporal_latest20"][0])
    folds = list(GroupKFold(n_splits=INNER_FOLDS).split(
        outer_train, ctx.y.iloc[outer_train], ctx.company.iloc[outer_train].to_numpy()))
    print(f"[protocol] selection on {len(outer_train)} training declarations, "
          f"criterion capture@{int(BUDGET*100)}%\n", flush=True)

    inner: list[dict[str, Any]] = []
    for name in ["classifier_reported", "regression_then_divide"]:
        caps, aucs = [], []
        t0 = time.time()
        for tr, va in folds:
            a, b = outer_train[tr], outer_train[va]
            s = (fit_classifier(ctx, ctx.model_frame, a, b) if name == "classifier_reported"
                 else fit_regressor(ctx, ctx.model_frame, a, b)[0])
            yv = np.asarray(ctx.y.iloc[b])
            caps.append(capture_at(yv, s)); aucs.append(roc_auc_score(yv, s))
        row = {"candidate": name, "inner_capture20_mean": float(np.mean(caps)),
               "inner_capture20_sd": float(np.std(caps)), "inner_auc_mean": float(np.mean(aucs)),
               "inner_folds": ";".join(f"{c:.4f}" for c in caps),
               "runtime_s": round(time.time() - t0, 1)}
        inner.append(row)
        print(f"[inner] {name:24s} capture@20 {row['inner_capture20_mean']:.4f} "
              f"+/- {row['inner_capture20_sd']:.4f}  AUC {row['inner_auc_mean']:.4f} "
              f"({row['runtime_s']:.0f}s)", flush=True)
        R.write_table(pd.DataFrame(inner), "regression_formulation_inner.csv")

    tab = pd.DataFrame(inner).sort_values("inner_capture20_mean", ascending=False)
    print(f"\n[selection] inner-CV winner: {tab.iloc[0]['candidate']}\n", flush=True)

    outer: list[dict[str, Any]] = []
    for name in ["classifier_reported", "regression_then_divide"]:
        for split_name in PUBLISHED:
            tr, te = splits[split_name]
            sigma = np.nan
            if name == "classifier_reported":
                s = fit_classifier(ctx, ctx.model_frame, tr, te)
            else:
                s, sigma = fit_regressor(ctx, ctx.model_frame, tr, te)
            yt = np.asarray(ctx.y.iloc[te])
            row = {"candidate": name, "split": split_name,
                   "roc_auc": roc_auc_score(yt, s),
                   "average_precision": average_precision_score(yt, s),
                   "capture_10pct": capture_at(yt, s, 0.10),
                   "capture_20pct": capture_at(yt, s, BUDGET),
                   "ece_10_bins": A.expected_calibration_error(yt, s, bins=10),
                   "brier": brier_score_loss(yt, s),
                   "residual_sigma": sigma}
            outer.append(row)
            print(f"[holdout] {name:24s} {split_name:18s} AUC {row['roc_auc']:.4f} "
                  f"cap@20 {row['capture_20pct']:.4f} ECE {row['ece_10_bins']:.4f}", flush=True)
            R.write_table(pd.DataFrame(outer), "regression_formulation_holdout.csv")

    R.write_manifest({
        "analysis": "regression on GWP with exact division by observed strength",
        "tau_kg_co2e_per_ksi": TAU,
        "probability": "normal residual model in log space, sigma from held-in folds",
        "selection": "company-grouped 3-fold CV inside the temporal training partition",
    }, "regression_formulation_manifest.json")


if __name__ == "__main__":
    main()
