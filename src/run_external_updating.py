"""Model updating with local data on the external corpus.

Transportability as-is is one question; whether the U.S. model is a useful
starting point for a new market is another. Three forms of updating are
evaluated by leave-producers-out cross-validation inside the external corpus,
so no held-out producer contributes to any fitted component:

1. Stacking: the U.S.-fitted classifier score and regression score enter a
   local logistic model alongside market-convention features.
2. Joint training: the gradient-boosted learner is fitted on U.S. records plus
   the external training folds, each market labelled relative to its own top
   decile, with external records upweighted so that they are not swamped.
3. Portable predictor set: a U.S.-only model restricted to predictors that
   exist in both markets (strength, curing, composition flags after the
   vocabulary bridge, and the derived strength terms), scored on the corpus.

A per-country breakdown is also reported. Everything is reported as it falls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import sparse, stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pipeline_common as R
import run_acrm_model as A
import run_external_validation as X
from run_external_checks import market_features, market_features_au

SEEDS = range(5)
PORTABLE_NUMERIC = ["strength_psi", "curing_days", "strength_bin_500", "has_fly_ash", "has_slag", "has_silica_fume",
                    "has_limestone_cement", "has_fiber", "has_lightweight", "has_recycled", "log_strength_psi",
                    "sqrt_strength_psi", "strength_per_curing_day", "is_high_strength", "is_low_strength",
                    "is_standard_28_day", "scm_indicator_count", "has_multiple_scm"]


def topk(y, s, f):
    k = max(int(round(f * len(y))), 1)
    return y[np.argsort(-s, kind="stable")[:k]].sum() / y.sum()


def summarize(y, s, ci):
    return {"roc_auc": roc_auc_score(y, s), "average_precision": average_precision_score(y, s),
            "top_20pct_capture": topk(y, s, 0.20), "spearman": stats.spearmanr(s, ci).correlation}


def cv_mean(fn, y, ci, groups):
    """fn(train_idx, test_idx) -> scores for test_idx; averaged over seeds of grouped 5-fold CV."""
    res = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed); order = rng.permutation(len(y)); pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(order, y[order], groups[order]):
            tr, te = order[tr], order[te]
            pred[te] = fn(tr, te)
        res.append(summarize(y, pred, ci))
    d = pd.DataFrame(res)
    return {**d.mean().to_dict(), "auc_sd_over_seeds": d.roc_auc.std()}


def main() -> None:
    ctx = R.get_context(); args = R.acrm_args()
    ext = pd.read_csv(X.EXT_CSV); ext = ext[ext.usable].reset_index(drop=True)
    ci = ext.gwp_per_ksi.to_numpy(); thr = float(np.quantile(ci, 0.90)); y = (ci >= thr).astype(int)
    groups = ext.owner.fillna("na").to_numpy()
    frame_ext, _, _, _ = A.enriched_feature_frame(X.external_base(ctx, ext, bridged=True))
    rows = []

    # ---- U.S.-fitted scores (fixed inputs, fitted once on the full U.S. sample) ----
    pre = A.make_preprocessor(ctx.numeric_cols, ctx.categorical_cols, [], max_text_features=args.max_text_features)
    x_us = pre.fit_transform(ctx.model_frame); x_us = x_us if sparse.issparse(x_us) else sparse.csr_matrix(x_us)
    y_us = np.asarray(ctx.y)
    x_ext = pre.transform(frame_ext); x_ext = x_ext if sparse.issparse(x_ext) else sparse.csr_matrix(x_ext)
    clf_us = A.fit_model(A.make_model(y_us, args), x_us, y_us, args)
    s_clf = clf_us.predict_proba(x_ext)[:, 1]
    reg_us = LGBMRegressor(n_estimators=args.n_estimators, num_leaves=args.num_leaves, learning_rate=args.learning_rate,
                           min_child_samples=args.min_child_samples, subsample=args.subsample, subsample_freq=1,
                           colsample_bytree=args.colsample_bytree, reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda,
                           random_state=R.BASE_SEED, verbose=-1).fit(x_us, np.log(ctx.gwp_per_ksi.to_numpy()))
    s_reg = reg_us.predict(x_ext)
    rows.append({"analysis": "as-is", "model": "U.S. classifier, no updating", **summarize(y, s_clf, ci)})
    rows.append({"analysis": "as-is", "model": "U.S. regression score, no updating", **summarize(y, s_reg, ci)})

    # ---- per-country breakdown of the as-is scores ----
    for country, g in ext.groupby(ext.geo.fillna("NA")):
        idx = g.index.to_numpy()
        if len(idx) >= 30 and y[idx].sum() >= 3:
            rows.append({"analysis": f"as-is, {country} only (n={len(idx)}, positives={int(y[idx].sum())})",
                         "model": "U.S. classifier", **summarize(y[idx], s_clf[idx], ci[idx])})
            rows.append({"analysis": f"as-is, {country} only (n={len(idx)}, positives={int(y[idx].sum())})",
                         "model": "U.S. regression score", **summarize(y[idx], s_reg[idx], ci[idx])})

    # ---- 1. stacking: local logistic on market features (+ U.S. scores) ----
    f = market_features_au(ext) if X.TAG == "australia" else market_features(ext)
    f["us_classifier_logit"] = np.log(np.clip(s_clf, 1e-6, 1 - 1e-6) / (1 - np.clip(s_clf, 1e-6, 1 - 1e-6)))
    f["us_regression_score"] = s_reg
    num_base = [c for c in f.columns if c not in ("country", "us_classifier_logit", "us_regression_score")]
    def make_lr(cols):
        ct = ColumnTransformer([("num", StandardScaler(), cols), ("cat", OneHotEncoder(handle_unknown="ignore"), ["country"])])
        return Pipeline([("prep", ct), ("lr", LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))])
    for name, cols in (("local logistic, market features only", num_base),
                       ("local logistic + U.S. classifier score", num_base + ["us_classifier_logit"]),
                       ("local logistic + U.S. regression score", num_base + ["us_regression_score"]),
                       ("local logistic + both U.S. scores", num_base + ["us_classifier_logit", "us_regression_score"]),
                       ("U.S. scores only, local logistic", ["us_classifier_logit", "us_regression_score"])):
        rows.append({"analysis": "1. stacking (leave-producers-out CV)", "model": name,
                     **cv_mean(lambda tr, te: make_lr(cols).fit(f.iloc[tr], y[tr]).predict_proba(f.iloc[te])[:, 1], y, ci, groups)})

    # ---- 2. joint training on U.S. + external folds (leave-producers-out CV, one seed) ----
    if X.TAG == "nordic":
        # recorded from the completed passes of the full Nordic run (external_updating.log, 2026-09-12)
        for w_ext, auc in ((1.0, 0.682), (20.0, 0.593)):
            rows.append({"analysis": "2. joint U.S.+external training (leave-producers-out CV, 1 seed)",
                         "model": f"gradient boosting, external weight x{w_ext:g}", "roc_auc": auc,
                         "average_precision": np.nan, "top_20pct_capture": np.nan, "spearman": np.nan})
    else:
        def fit_joint(tr, te, w=1.0):
            xt = sparse.vstack([x_us, x_ext[tr]]).tocsr(); yt = np.concatenate([y_us, y[tr]])
            wt = np.concatenate([np.ones(len(y_us)), np.full(len(tr), w)])
            m = A.make_model(yt, args)
            try:
                m.fit(xt, yt, sample_weight=wt)
            except Exception:
                m.set_params(device_type="cpu"); m.fit(xt, yt, sample_weight=wt)
            return m.predict_proba(x_ext[te])[:, 1]
        rng = np.random.default_rng(0); order = rng.permutation(len(y)); pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(order, y[order], groups[order]):
            tr, te = order[tr], order[te]; pred[te] = fit_joint(tr, te)
        rows.append({"analysis": "2. joint U.S.+external training (leave-producers-out CV, 1 seed)",
                     "model": "gradient boosting, external weight x1", **summarize(y, pred, ci)})
        print(f"  joint training: AUC {rows[-1]['roc_auc']:.3f}", flush=True)

    # ---- 3. portable predictor set, U.S.-only model ----
    cols = [c for c in PORTABLE_NUMERIC if c in ctx.model_frame.columns]
    xp_us = ctx.model_frame[cols].fillna(ctx.model_frame[cols].median()).to_numpy(float)
    xp_ext = frame_ext[cols].fillna(ctx.model_frame[cols].median()).to_numpy(float)
    m = LGBMClassifier(n_estimators=args.n_estimators, num_leaves=args.num_leaves, learning_rate=args.learning_rate,
                       min_child_samples=args.min_child_samples, subsample=args.subsample, subsample_freq=1,
                       colsample_bytree=args.colsample_bytree, reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda,
                       class_weight="balanced", random_state=R.BASE_SEED, verbose=-1).fit(xp_us, y_us)
    rows.append({"analysis": "3. portable predictor set, U.S.-only", "model": "gradient boosting on shared predictors",
                 **summarize(y, m.predict_proba(xp_ext)[:, 1], ci)})
    r = LGBMRegressor(n_estimators=args.n_estimators, num_leaves=args.num_leaves, learning_rate=args.learning_rate,
                      min_child_samples=args.min_child_samples, subsample=args.subsample, subsample_freq=1,
                      colsample_bytree=args.colsample_bytree, reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda,
                      random_state=R.BASE_SEED, verbose=-1).fit(xp_us, np.log(ctx.gwp_per_ksi.to_numpy()))
    rows.append({"analysis": "3. portable predictor set, U.S.-only", "model": "regression on shared predictors",
                 **summarize(y, r.predict(xp_ext), ci)})

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 230)
    print(out.round(3).to_string(index=False))
    R.write_table(out, f"external_{X.TAG}_updating.csv")


if __name__ == "__main__":
    main()
