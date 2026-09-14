"""Checks on the external result: learner, feature mapping, and learnability.

Three questions the external validation raises, answered in turn:

1. Learner. Is the transfer result specific to the LightGBM learner? The
   XGBoost baseline and a strength-only logistic model are fitted on the same
   U.S. data and scored on the external corpus.
2. Feature mapping. Does the U.S.-fitted model reduce to a strength rule on
   the external corpus? The rank agreement between its score and declared
   strength is measured, with and without the vocabulary bridge.
3. Learnability. Can the external top decile be predicted at all from features
   that follow the market's own declaration conventions (EN 197-1 cement type
   and class, exposure classes, consistency, product type, country)? A
   right-sized learner is evaluated by leave-producers-out cross-validation
   inside the external corpus, against strength alone.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pipeline_common as R
import run_acrm_model as A
import concrete_epd_pipeline as M
import run_external_validation as X
from lightgbm import LGBMClassifier


def market_features(ext: pd.DataFrame) -> pd.DataFrame:
    """Features written in the external market's own conventions."""
    t = (ext.name.fillna("") + " " + ext.text_product.fillna("")).str.replace(" ", " ")
    f = pd.DataFrame(index=ext.index)
    f["strength_mpa"] = ext.strength_mpa
    f["log_strength"] = np.log(ext.strength_mpa)
    f["cem_i"] = t.str.contains(r"\bCEM\s?[Il]\b(?![Il])", regex=True).astype(int)
    f["cem_ii_a"] = t.str.contains(r"CEM\s?[Il]{2}\s?/\s?A", regex=True).astype(int)
    f["cem_ii_b"] = t.str.contains(r"CEM\s?[Il]{2}\s?/\s?B", regex=True).astype(int)
    f["cem_iii"] = t.str.contains(r"CEM\s?[Il]{3}", regex=True).astype(int)
    f["cem_52_5"] = t.str.contains(r"52[,.]5", regex=True).astype(int)
    f["cem_rapid"] = t.str.contains(r"52[,.]5\s?R\b|42[,.]5\s?R\b", regex=True).astype(int)
    f["low_carbon_label"] = t.str.contains(r"lavkarbon|low[- ]carbon|klimabeton|\beco\b|futurecem|miljø", case=False, regex=True).astype(int)
    f["fly_ash"] = t.str.contains(r"CEM\s?II\s?/\s?[AB]\s?-\s?V|flyveaske|flygeaske|fly ash", case=False, regex=True).astype(int)
    f["slag"] = t.str.contains(r"CEM\s?II\s?/\s?[AB]\s?-\s?S\b|CEM\s?III|slagg", case=False, regex=True).astype(int)
    f["scc"] = t.str.contains(r"\bSCC\b|vibfri|selvkomprimerende|self[- ]compacting", case=False, regex=True).astype(int)
    f["shotcrete"] = t.str.contains(r"spr[øo]ytebetong|sprutbetong|shotcrete|sprayed", case=False, regex=True).astype(int)
    f["frost_xf"] = t.str.contains(r"\bXF[1-4]\b|frost", case=False, regex=True).astype(int)
    f["aggressive"] = t.str.contains(r"\bX[SDA][1-3]\b|aggressive|marine", case=False, regex=True).astype(int)
    f["passive_x0"] = t.str.contains(r"\bX0\b|passive|indoor|innendørs|indendørs", case=False, regex=True).astype(int)
    f["average_epd"] = t.str.contains(r"average|gennemsnit|gjennomsnitt", case=False, regex=True).astype(int)
    f["year"] = ext.year.fillna(ext.year.median())
    f["country"] = ext.geo.fillna("unknown")
    return f


def market_features_au(ext: pd.DataFrame) -> pd.DataFrame:
    """Features in Australian declaration conventions (AS 1379 grades, SCM %, product lines, state)."""
    t = (ext.name.fillna("") + " " + ext.text_product.fillna(""))
    f = pd.DataFrame(index=ext.index)
    f["strength_mpa"] = ext.strength_mpa
    f["log_strength"] = np.log(ext.strength_mpa)
    f["fly_ash"] = t.str.contains(r"fly ash|\bFA\b", case=False, regex=True).astype(int)
    f["slag"] = t.str.contains(r"GGBFS|GGBS|slag", case=False, regex=True).astype(int)
    f["silica_fume"] = t.str.contains(r"silica fume|\bSF\b|microsilica", case=False, regex=True).astype(int)
    f["gp_cement"] = t.str.contains(r"\bGP\b|general purpose cement", case=False, regex=True).astype(int)
    pct = t.str.extract(r"SCM[^.%]{0,40}?(\d{1,2})\s?%\s?(?:and|to|-)\s?(\d{1,2})\s?%", flags=re.I)
    f["scm_pct_upper"] = pd.to_numeric(pct[1], errors="coerce").fillna(-1)
    f["scm_pct_stated"] = (f["scm_pct_upper"] >= 0).astype(int)
    f["low_carbon_line"] = t.str.contains(r"ECOPact|Greenstar|Green Star|ViroDecs|Envisia|lower carbon|low carbon|climate act|eco", case=False, regex=True).astype(int)
    f["high_strength_line"] = t.str.contains(r"high strength|post[- ]tension", case=False, regex=True).astype(int)
    f["hardscape"] = t.str.contains(r"footpath|kerb|curb|driveway|paving|path", case=False, regex=True).astype(int)
    f["structural"] = t.str.contains(r"structural|footing|slab|column|wall|beam", case=False, regex=True).astype(int)
    f["shotcrete"] = t.str.contains(r"shotcrete|sprayed", case=False, regex=True).astype(int)
    f["project_specific"] = t.str.contains(r"\bproject\b", case=False, regex=True).astype(int)
    f["average_epd"] = t.str.contains(r"average", case=False, regex=True).astype(int)
    for st in ("NSW", "VIC", "QLD", "WA", "SA", "ACT", "TAS", "NT"):
        f[f"state_{st}"] = t.str.contains(rf"\b{st}\b|{ {'NSW':'Sydney|Newcastle','VIC':'Melbourne|Victoria','QLD':'Brisbane|Queensland|Sunshine Coast','WA':'Perth|Western Australia','SA':'Adelaide','ACT':'Canberra','TAS':'Tasmania|Hobart','NT':'Darwin'}[st] }", case=False, regex=True).astype(int)
    f["year"] = ext.year.fillna(ext.year.median())
    f["country"] = ext.geo.fillna("unknown")
    return f


def topk_capture(y, s, frac):
    k = max(int(round(frac * len(y))), 1)
    return y[np.argsort(-s, kind="stable")[:k]].sum() / y.sum()


def summarize(y, s, ci):
    return {"roc_auc": roc_auc_score(y, s), "average_precision": average_precision_score(y, s),
            "top_20pct_capture": topk_capture(y, s, 0.20), "spearman_prob_vs_ci": stats.spearmanr(s, ci).correlation}


def main() -> None:
    ctx = R.get_context(); args = R.acrm_args()
    ext = pd.read_csv(X.EXT_CSV); ext = ext[ext.usable].reset_index(drop=True)
    ci = ext.gwp_per_ksi.to_numpy(); thr = float(np.quantile(ci, 0.90))
    y = (ci >= thr).astype(int); groups = ext.owner.fillna("not_available").to_numpy()
    print(f"external: n={len(ext)}, positives={int(y.sum())}, threshold={thr:.1f}")

    # ---- 1. learner comparison, all fitted on the full U.S. sample -------------
    rows = []
    pre = A.make_preprocessor(ctx.numeric_cols, ctx.categorical_cols, [], max_text_features=args.max_text_features)
    x_us = pre.fit_transform(ctx.model_frame); x_us = sparse.csr_matrix(x_us) if not sparse.issparse(x_us) else x_us
    y_us = np.asarray(ctx.y)
    learners = {
        "ACRM (LightGBM, reported configuration)": lambda: A.fit_model(A.make_model(y_us, args), x_us, y_us, args),
        "XGBoost baseline (reported configuration)": lambda: M.make_model(pd.Series(y_us)).fit(x_us, y_us),
    }
    for variant, bridged in (("no_adaptation", False), ("vocabulary_bridge", True)):
        base = X.external_base(ctx, ext, bridged)
        frame, _, _, _ = A.enriched_feature_frame(base)
        x_ext = pre.transform(frame); x_ext = sparse.csr_matrix(x_ext) if not sparse.issparse(x_ext) else x_ext
        for name, fit in learners.items():
            s = fit().predict_proba(x_ext)[:, 1]
            m = summarize(y, s, ci)
            m["spearman_score_vs_strength"] = stats.spearmanr(s, -ext.strength_psi).correlation
            rows.append({"question": "learner", "variant": variant, "model": name, **m})
        # strength-only logistic fitted on the U.S. sample
        lr = LogisticRegression(max_iter=1000).fit(np.log(ctx.strength.to_numpy()).reshape(-1, 1), y_us)
        s = lr.predict_proba(np.log(ext.strength_psi.to_numpy()).reshape(-1, 1))[:, 1]
        m = summarize(y, s, ci); m["spearman_score_vs_strength"] = stats.spearmanr(s, -ext.strength_psi).correlation
        rows.append({"question": "learner", "variant": variant, "model": "Strength-only logistic (U.S.-fitted)", **m})
    # composition flags actually set on the external frame, both variants
    for variant, bridged in (("no_adaptation", False), ("vocabulary_bridge", True)):
        base = X.external_base(ctx, ext, bridged)
        flags = {c: int(base[c].sum()) for c in ["has_fly_ash", "has_slag", "has_silica_fume", "has_limestone_cement", "has_fiber", "has_recycled"]}
        print(f"  composition flags set [{variant}]: {flags}")

    # ---- 3. learnability with market-convention features ---------------------
    f = market_features_au(ext) if X.TAG == "australia" else market_features(ext)
    num = [c for c in f.columns if c != "country"]
    print("  market features present (share of records):",
          {c: round(float(f[c].mean()), 2) for c in f.columns if c not in ("country", "year", "strength_mpa", "log_strength", "scm_pct_upper")})
    ct = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), ["country"])])
    models = {
        "Market-feature logistic (within corpus)": lambda: Pipeline([("prep", ct), ("lr", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))]),
        "Market-feature LightGBM, small (within corpus)": lambda: Pipeline([("prep", ct), ("gbm", LGBMClassifier(n_estimators=150, num_leaves=7, learning_rate=0.05, min_child_samples=8, subsample=0.9, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced", verbose=-1, random_state=0))]),
        "Strength-only logistic (within corpus)": lambda: Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), ["log_strength"])])), ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    }
    gkf = GroupKFold(n_splits=5)
    for name, make in models.items():
        res = []
        for seed in range(5):
            rng = np.random.default_rng(seed); order = rng.permutation(len(y)); pred = np.zeros(len(y))
            for tr, te in gkf.split(order, y[order], groups[order]):
                tr, te = order[tr], order[te]
                pred[te] = make().fit(f.iloc[tr], y[tr]).predict_proba(f.iloc[te])[:, 1]
            res.append(summarize(y, pred, ci))
        d = pd.DataFrame(res).mean().to_dict()
        rows.append({"question": "learnability", "variant": "leave-producers-out CV", "model": name, **d})

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n" + out.round(3).to_string(index=False))
    R.write_table(out, f"external_{X.TAG}_checks.csv")


if __name__ == "__main__":
    main()
