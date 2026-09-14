"""Regime-matched controls for the external validation.

The external corpus lies entirely below the U.S. high-carbon threshold, so a
classifier trained to resolve that threshold is being asked to order records
inside a band where it was never required to discriminate. Two controls make
that explicit:

1. Regime-matched U.S. control. The reported model's stored scores are
   evaluated on U.S. test records restricted to the external corpus's carbon
   intensity range, against the top decile of that restricted set.
2. Regression score. A gradient-boosted regressor of log carbon intensity,
   same predictors and configuration, is fitted on U.S. data only and scored
   on the external corpus and on the same restricted U.S. control, so the two
   formulations can be compared on a like-for-like task.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import sparse, stats
from sklearn.metrics import average_precision_score, roc_auc_score

import pipeline_common as R
import run_acrm_model as A
import run_external_validation as X

CAPS = (75.0, 100.0)


def topk(y, s, f):
    k = max(int(round(f * len(y))), 1)
    return y[np.argsort(-s, kind="stable")[:k]].sum() / y.sum()


def evaluate(ci: np.ndarray, s: np.ndarray) -> dict[str, float]:
    thr = np.quantile(ci, 0.90); y = (ci >= thr).astype(int)
    return {"n": len(ci), "threshold": thr, "roc_auc": roc_auc_score(y, s), "average_precision": average_precision_score(y, s),
            "top_20pct_capture": topk(y, s, 0.20), "spearman": stats.spearmanr(s, ci).correlation}


def main() -> None:
    ctx = R.get_context(); args = R.acrm_args()
    ext = pd.read_csv(X.EXT_CSV); ext = ext[ext.usable].reset_index(drop=True)
    ci_ext = ext.gwp_per_ksi.to_numpy()
    ext_max = float(ci_ext.max())
    frame_ext, _, _, _ = A.enriched_feature_frame(X.external_base(ctx, ext, bridged=True))
    stored = pd.read_csv(R.RESULTS_DIR / "acrm_single_model_predictions.csv")
    splits = ctx.published_splits()
    rows = []

    for split in ("group_company", "group_epd_source", "temporal_latest20"):
        tr, te = splits[split]
        pre = A.make_preprocessor(ctx.numeric_cols, ctx.categorical_cols, [], max_text_features=args.max_text_features)
        x_tr = pre.fit_transform(ctx.model_frame.iloc[tr]); x_te = pre.transform(ctx.model_frame.iloc[te])
        x_tr, x_te = (m if sparse.issparse(m) else sparse.csr_matrix(m) for m in (x_tr, x_te))
        x_ext = pre.transform(frame_ext); x_ext = x_ext if sparse.issparse(x_ext) else sparse.csr_matrix(x_ext)
        reg = LGBMRegressor(n_estimators=args.n_estimators, num_leaves=args.num_leaves, learning_rate=args.learning_rate,
                            min_child_samples=args.min_child_samples, subsample=args.subsample, subsample_freq=1,
                            colsample_bytree=args.colsample_bytree, reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda,
                            random_state=R.BASE_SEED, verbose=-1)
        reg.fit(x_tr, np.log(ctx.gwp_per_ksi.iloc[tr].to_numpy()))
        s_te_reg = reg.predict(x_te); s_ext_reg = reg.predict(x_ext)
        st = stored[stored.split == split].set_index("row_index").loc[te]
        s_te_cls = st.acrm_probability.to_numpy(); ci_te = st.gwp_per_ksi.to_numpy()
        assert np.allclose(ci_te, ctx.gwp_per_ksi.iloc[te].to_numpy())

        for cap in CAPS + (ext_max,):
            m = ci_te <= cap
            for name, s in (("classifier (reported)", s_te_cls[m]), ("regression score", s_te_reg[m])):
                rows.append({"where": f"U.S. {split}, CI <= {cap:.1f}", "model": name, **evaluate(ci_te[m], s)})
        rows.append({"where": "U.S. full holdout", "model": "classifier (reported)", **evaluate(ci_te, s_te_cls)})
        rows.append({"where": "U.S. full holdout", "model": "regression score", **evaluate(ci_te, s_te_reg)})
        rows.append({"where": f"External corpus (fitted on U.S. {split} training partition)", "model": "regression score", **evaluate(ci_ext, s_ext_reg)})

    # regression fitted on the full U.S. sample, scored on the external corpus
    pre = A.make_preprocessor(ctx.numeric_cols, ctx.categorical_cols, [], max_text_features=args.max_text_features)
    x_all = pre.fit_transform(ctx.model_frame); x_all = x_all if sparse.issparse(x_all) else sparse.csr_matrix(x_all)
    x_ext = pre.transform(frame_ext); x_ext = x_ext if sparse.issparse(x_ext) else sparse.csr_matrix(x_ext)
    reg = LGBMRegressor(n_estimators=args.n_estimators, num_leaves=args.num_leaves, learning_rate=args.learning_rate,
                        min_child_samples=args.min_child_samples, subsample=args.subsample, subsample_freq=1,
                        colsample_bytree=args.colsample_bytree, reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda,
                        random_state=R.BASE_SEED, verbose=-1).fit(x_all, np.log(ctx.gwp_per_ksi.to_numpy()))
    s_ext = reg.predict(x_ext)
    rows.append({"where": "External corpus (fitted on full U.S. sample)", "model": "regression score", **evaluate(ci_ext, s_ext)})
    # bootstrap interval for that one, by producer
    rng = np.random.default_rng(R.BASE_SEED); groups = ext.owner.fillna("na").to_numpy()
    keys = pd.unique(groups); idx = {k: np.flatnonzero(groups == k) for k in keys}
    aucs = []
    for _ in range(1000):
        pick = np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        y = (ci_ext[pick] >= np.quantile(ci_ext, 0.9)).astype(int)
        if y.sum() and (1 - y).sum():
            aucs.append(roc_auc_score(y, s_ext[pick]))
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    print(f"\nregression score on the external corpus: AUC 95% producer-resampled interval [{lo:.3f}, {hi:.3f}]")
    print(f"predicted vs observed carbon intensity on the external corpus: Spearman {stats.spearmanr(s_ext, ci_ext).correlation:.3f}; "
          f"median predicted {np.exp(np.median(s_ext)):.1f} vs observed {np.median(ci_ext):.1f} kg CO2 eq per ksi")

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    R.write_table(out, f"external_{X.TAG}_regime_controls.csv")
    pd.DataFrame({"uuid": ext.uuid, "owner": ext.owner, "gwp_per_ksi": ci_ext, "regression_log_ci_pred": s_ext}).to_csv(
        R.RESULTS_DIR / f"external_{X.TAG}_regression_predictions.csv", index=False)


if __name__ == "__main__":
    main()
