"""External validation on an independently compiled corpus.

The development data are one compilation of U.S. declarations. This scores a
corpus that was collected by a different organization, under a different
program operator and product category rule, in other countries, and extracted
by a different process: ready-mixed concrete EPDs published in the EPD-Norge
digital registry (Nordic and other European producers). The model is fitted on
U.S. records only and applied unchanged.

Two facts shape the design. First, no external declaration reaches the U.S.
high-carbon threshold, so the absolute label has no positives there; the
threshold transfer is therefore reported as the flag rate and probability
distribution, and discrimination is evaluated against the external corpus's
own top decile. Second, the free text is Norwegian, Danish, and Swedish, so
the U.S. TF-IDF vocabulary matches little of it; the corpus is scored both
with no adaptation at all and with a documented vocabulary bridge that maps
cement-type notation and Nordic constituent words to the English terms the
composition flags look for.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import average_precision_score, roc_auc_score

import pipeline_common as R
import run_acrm_model as A
import concrete_epd_pipeline as M

import os
TAG = os.environ.get("EXTERNAL_TAG", "nordic")
EXT_CSV = Path(os.environ.get("EXTERNAL_CSV", Path(__file__).resolve().parents[1] / "data" / "external" / "epdnorge_ready_mixed.csv"))
US_THRESHOLD_KEY = "gwp_per_ksi_q90"
N_BOOT = 1000
SEED = 42

# Vocabulary bridge: cement-type notation (EN 197-1) and Nordic constituent
# words mapped to the English terms the U.S. composition flags detect.
BRIDGE = [
    # supplementary cementitious materials (EN 197-1 notation and Nordic words)
    (r"CEM\s?II\s?/\s?[AB]\s?-\s?V\b|flyveaske|flygeaske|flyaske|flyveask", "fly ash"),
    (r"CEM\s?II\s?/\s?[AB]\s?-\s?S\b|CEM\s?III|slagg\b|slagge", "slag"),
    (r"CEM\s?II\s?/\s?[AB]\s?-\s?L{1,2}\b|kalkstein|kalksten|kalkfiller", "type il limestone cement"),
    (r"CEM\s?II\s?/\s?[AB]\s?-\s?M\b|futurecem", "blended cement limestone"),
    (r"CEM\s?II\s?/\s?[AB]\s?-\s?D\b|mikrosilika|silika\b|silikast", "silica fume"),
    # pure Portland cement: the U.S. text says so explicitly
    (r"\bCEM\s?I\b(?!I)|\bCEM\s?l\b(?!l)", "portland cement type i cement"),
    (r"52[,.]5\s?R\b|high early|hurtig", "high early strength"),
    # placement and product type
    (r"\bSCC\b|vibfri|selvkomprimerende|självkompakterande|self[- ]compacting", "self consolidating scc"),
    (r"spr[øo]ytebetong|sprutbetong|shotcrete|sprayed", "shotcrete"),
    (r"\bXF[1-4]\b|frost|luftinnhold|air content|luftporer", "air entrained freeze thaw"),
    (r"\bXS[1-3]\b|\bXD[1-3]\b|\bXA[1-3]\b|aggressive|marine|chloride", "marine chloride sulfate exposure"),
    (r"\bfiber|\bfibre|stålfiber|polypropylenfiber", "fiber"),
    (r"resirkulert|genbrug|återvunn", "recycled"),
    (r"lettbetong|letbeton|lättbetong", "lightweight"),
    (r"lavkarbon|low[- ]carbon|klimabeton|eco\b|miljøsement|miljösement", "low carbon"),
    # Australian conventions
    (r"GGBFS|GGBS|blast furnace slag", "slag"),
    (r"GP cement|general purpose cement|\bGP\b", "portland cement type i cement"),
    (r"ECOPact|Greenstar|Green Star|ViroDecs|Envisia|lower carbon|climate act", "low carbon"),
    (r"post[- ]tension|precast|prestress|structural", "structural"),
    (r"footpath|kerb|curb|driveway|paving", "hardscape paving"),
]


def bridge_text(text: str) -> str:
    extra = [eng for pat, eng in BRIDGE if re.search(pat, text, re.I)]
    return " ".join(extra)


NEGATED = re.compile(r"[^.]*\bnot (covered|included|part of|applicable)\b[^.]*\.?", re.I)

# raw categorical columns the U.S. pipeline one-hot encodes; unknown values
# must stay unknown rather than be imputed with the most frequent U.S. value
RAW_CATEGORICAL = ["Company", "Company Location - State", "Plant", "Plant Location - City",
                   "Plant Location - State", "U.S. Region of Plant", "EPD Program Operator",
                   "Concrete Curation Time"]


def external_base(ctx, ext: pd.DataFrame, bridged: bool) -> pd.DataFrame:
    """A frame with the development data's columns, filled from the external records.

    Product text (name, technology description, applicability) fills the
    description field; the registry's LCA data-quality boilerplate is left out,
    as the U.S. description field holds product text only.
    """
    base = pd.DataFrame(index=range(len(ext)), columns=ctx.base.columns, dtype=object)
    for c in RAW_CATEGORICAL:
        base[c] = "not_available"
    base["Company"] = ext.owner.fillna("not_available").to_numpy()
    base["EPD Program Operator"] = "EPD-Norge"
    base["EPD Source Link"] = ext.uuid.to_numpy()
    base["Concrete Compressive Strength (psi)"] = ext.strength_psi.to_numpy()
    base["strength_psi"] = ext.strength_psi.astype(float).to_numpy()
    base["issue_year"] = ext.year.astype(float).to_numpy()
    base["EPD Date of Issue"] = [f"{int(y)}-01-01" if pd.notna(y) else None for y in ext.year]
    base["issue_date"] = pd.to_datetime(base["EPD Date of Issue"], errors="coerce")
    base["Concrete Curation Time"] = "28 days"
    base["curing_days"] = 28.0
    product = ext.text_product.fillna("")
    if bridged:
        product = product.map(lambda t: NEGATED.sub(" ", t))
    base["Mix Label"] = ext.name.to_numpy()
    base["Mix Description"] = product.to_numpy()
    comp = ext.name.fillna("")
    if bridged:
        comp = comp + " " + product.map(bridge_text)
    base["Product Components"] = comp.to_numpy()
    base[M.TARGET_COL] = ext.gwp_a1a3.astype(float).to_numpy()
    base["gwp_per_ksi"] = ext.gwp_per_ksi.astype(float).to_numpy()
    base["strength_bin_500"] = ((base["strength_psi"].astype(float) / 500).round() * 500).astype(int)
    base = M.make_flags(base)
    return base


def topk(y: np.ndarray, s: np.ndarray, frac: float) -> dict[str, float]:
    n = len(y); k = max(int(round(frac * n)), 1)
    order = np.argsort(-s, kind="stable")[:k]
    tp = y[order].sum(); P = y.sum(); prev = P / n
    return {"capture": tp / P if P else np.nan, "precision": tp / k, "lift": (tp / k) / prev if prev else np.nan}


def metrics(y: np.ndarray, s: np.ndarray, ci: np.ndarray) -> dict[str, float]:
    out = {"roc_auc": roc_auc_score(y, s), "average_precision": average_precision_score(y, s),
           "spearman_prob_vs_ci": stats.spearmanr(s, ci).correlation}
    for f in (0.10, 0.20):
        m = topk(y, s, f)
        out[f"top_{int(f*100)}pct_capture"] = m["capture"]
        out[f"top_{int(f*100)}pct_precision"] = m["precision"]
        out[f"top_{int(f*100)}pct_lift"] = m["lift"]
    return out


def bootstrap(y, s, ci, groups, fn, reps=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    rows_row, rows_cl = [], []
    keys = pd.unique(groups); index = {k: np.flatnonzero(groups == k) for k in keys}
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            rows_row.append(fn(y[idx], s[idx], ci[idx]))
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([index[keys[i]] for i in pick])
        if len(np.unique(y[idx])) == 2:
            rows_cl.append(fn(y[idx], s[idx], ci[idx]))
    def summ(rows):
        d = pd.DataFrame(rows)
        return {c: (d[c].quantile(0.025), d[c].quantile(0.975)) for c in d.columns}
    return summ(rows_row), summ(rows_cl)


def main() -> None:
    ctx = R.get_context()
    args = R.acrm_args()
    ext = pd.read_csv(EXT_CSV)
    ext = ext[ext.usable].reset_index(drop=True)
    us_thr = ctx.summary[US_THRESHOLD_KEY]
    ext_thr = float(ext.gwp_per_ksi.quantile(0.90))
    y_ext = (ext.gwp_per_ksi >= ext_thr).astype(int).to_numpy()
    y_us_def = (ext.gwp_per_ksi >= us_thr).astype(int).to_numpy()
    ci = ext.gwp_per_ksi.to_numpy()
    groups = ext.owner.fillna("unknown").to_numpy()
    print(f"external corpus: {len(ext)} records, {ext.owner.nunique()} producers, "
          f"countries {ext.geo.value_counts().to_dict()}")
    print(f"U.S. threshold {us_thr:.1f} kg CO2 eq per ksi -> external positives {int(y_us_def.sum())}; "
          f"external 90th percentile {ext_thr:.1f} -> positives {int(y_ext.sum())}")

    # models fitted on U.S. data only
    fits = {}
    train_temporal, _ = ctx.published_splits()["temporal_latest20"]
    for name, idx in (("full_us_sample", np.arange(ctx.n)), ("temporal_training_partition", train_temporal)):
        pre = A.make_preprocessor(ctx.numeric_cols, ctx.categorical_cols, [], max_text_features=args.max_text_features)
        x = pre.fit_transform(ctx.model_frame.iloc[idx])
        if not sparse.issparse(x):
            x = sparse.csr_matrix(x)
        y = np.asarray(ctx.y.iloc[idx])
        model = A.fit_model(A.make_model(y, args), x, y, args)
        fits[name] = (pre, model)
        print(f"  fitted {name}: {len(idx):,} records")

    rows, preds = [], []
    for variant, bridged in (("no_adaptation", False), ("vocabulary_bridge", True)):
        base = external_base(ctx, ext, bridged)
        frame, num_cols, cat_cols, _ = A.enriched_feature_frame(base)
        assert num_cols == ctx.numeric_cols and cat_cols == ctx.categorical_cols, "feature columns differ"
        for fit_name, (pre, model) in fits.items():
            x = pre.transform(frame)
            if not sparse.issparse(x):
                x = sparse.csr_matrix(x)
            s = model.predict_proba(x)[:, 1]
            m = metrics(y_ext, s, ci)
            ci_row, ci_cl = bootstrap(y_ext, s, ci, groups, metrics)
            row = {"variant": variant, "fitted_on": fit_name, "n_external": len(ext),
                   "n_producers": int(ext.owner.nunique()), "external_threshold": ext_thr,
                   "us_threshold": us_thr, "external_positives_us_definition": int(y_us_def.sum()),
                   "flag_rate_at_0.5": float((s >= 0.5).mean()), "prob_median": float(np.median(s)),
                   "prob_p95": float(np.quantile(s, 0.95)), "prob_max": float(s.max())}
            for k, v in m.items():
                row[k] = v
                row[f"{k}_ci_low_row"], row[f"{k}_ci_high_row"] = ci_row[k]
                row[f"{k}_ci_low_producer"], row[f"{k}_ci_high_producer"] = ci_cl[k]
            # within-class rank correlation
            wc = []
            for b, g in pd.DataFrame({"b": frame["strength_bin_500"], "s": s, "ci": ci}).groupby("b"):
                if len(g) >= 20:
                    wc.append((b, len(g), stats.spearmanr(g.s, g.ci).correlation))
            row["within_class_spearman"] = "; ".join(f"{int(b)} psi (n={n}): {r:.2f}" for b, n, r in wc)
            rows.append(row)
            preds.append(pd.DataFrame({"variant": variant, "fitted_on": fit_name, "uuid": ext.uuid,
                                       "owner": ext.owner, "geo": ext.geo, "year": ext.year,
                                       "strength_psi": ext.strength_psi, "gwp_a1a3": ext.gwp_a1a3,
                                       "gwp_per_ksi": ci, "y_external_top_decile": y_ext, "proba": s}))
            print(f"\n[{variant} | {fit_name}] AUC {m['roc_auc']:.3f} "
                  f"[{ci_row['roc_auc'][0]:.3f}, {ci_row['roc_auc'][1]:.3f}] row / "
                  f"[{ci_cl['roc_auc'][0]:.3f}, {ci_cl['roc_auc'][1]:.3f}] producer | AP {m['average_precision']:.3f} | "
                  f"capture@10% {m['top_10pct_capture']:.3f} @20% {m['top_20pct_capture']:.3f} | "
                  f"lift@20% {m['top_20pct_lift']:.2f} | Spearman(prob, CI) {m['spearman_prob_vs_ci']:.3f} | "
                  f"flag rate at 0.5: {row['flag_rate_at_0.5']:.3f}, prob median {row['prob_median']:.3f}, p95 {row['prob_p95']:.3f}")
            print("   within class:", row["within_class_spearman"])

    # practice rules on the external corpus, for context
    rules = {"Unordered review (expected)": None,
             "Lowest declared strength first": -ext.strength_psi.to_numpy(),
             "Highest declared strength first": ext.strength_psi.to_numpy(),
             "Arrival order (oldest first)": -ext.year.fillna(ext.year.median()).to_numpy()}
    rule_rows = []
    for name, score in rules.items():
        if score is None:
            rule_rows.append({"rule": name, "top_10pct_capture": 0.10, "top_20pct_capture": 0.20}); continue
        rng = np.random.default_rng(SEED)
        score = score + rng.normal(0, 1e-6, len(score))    # random tie order
        rule_rows.append({"rule": name, "top_10pct_capture": topk(y_ext, score, 0.10)["capture"],
                          "top_20pct_capture": topk(y_ext, score, 0.20)["capture"]})
    rule_df = pd.DataFrame(rule_rows)
    print("\npractice rules on the external corpus:"); print(rule_df.round(3).to_string(index=False))

    R.write_table(pd.DataFrame(rows), f"external_{TAG}_validation.csv")
    R.write_table(pd.concat(preds, ignore_index=True), f"external_{TAG}_predictions.csv")
    R.write_table(rule_df, f"external_{TAG}_practice_rules.csv")
    json.dump({"source": "EPD-Norge digital registry, class Bygg / Ferdig betong, retrieved 2026-09-12",
                  "endpoint": "https://epdnorway.lca-data.com/resource/processes",
                  "records_retrieved": 277, "records_usable": int(len(ext)),
                  "usable_rule": "declared unit 1 m3, GWP A1 to A3 present, strength class parsed",
                  "strength_conversion": "cylinder MPa x 145.038 = psi; EN 206 C xx/yy -> xx; Norwegian B xx -> xx",
                  "gwp_indicator": "GWP-total (EN 15804+A2) or GWP (EN 15804+A1), modules A1+A2+A3",
                  "bridge": BRIDGE, "bootstrap_reps": N_BOOT, "seed": SEED}, open(R.RESULTS_DIR / f"external_{TAG}_manifest.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__" and __import__("sys").argv[-1] != "--within":
    main()


def within_external_cv() -> None:
    """Refit the same pipeline inside the external corpus, grouped by producer.

    Separates whether the U.S. weights transfer from whether the method does:
    the same feature construction, learner and configuration, trained only on
    external declarations, evaluated by leave-producers-out cross-validation.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    ctx = R.get_context(); args = R.acrm_args()
    ext = pd.read_csv(EXT_CSV); ext = ext[ext.usable].reset_index(drop=True)
    ext_thr = float(ext.gwp_per_ksi.quantile(0.90))
    y = (ext.gwp_per_ksi >= ext_thr).astype(int).to_numpy()
    groups = ext.owner.fillna("not_available").to_numpy()
    base = external_base(ctx, ext, bridged=True)
    frame, num_cols, cat_cols, _ = A.enriched_feature_frame(base)
    gkf = GroupKFold(n_splits=5)
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(seed); order = rng.permutation(len(y))
        pred = np.zeros(len(y)); pred_lr = np.zeros(len(y))
        for tr, te in gkf.split(order, y[order], groups[order]):
            tr, te = order[tr], order[te]
            pre = A.make_preprocessor(num_cols, cat_cols, [], max_text_features=args.max_text_features)
            xtr = pre.fit_transform(frame.iloc[tr]); xte = pre.transform(frame.iloc[te])
            if not sparse.issparse(xtr): xtr = sparse.csr_matrix(xtr); xte = sparse.csr_matrix(xte)
            model = A.fit_model(A.make_model(y[tr], args), xtr, y[tr], args)
            pred[te] = model.predict_proba(xte)[:, 1]
            s_tr = frame.strength_psi.iloc[tr].to_numpy().reshape(-1, 1); s_te = frame.strength_psi.iloc[te].to_numpy().reshape(-1, 1)
            pred_lr[te] = LogisticRegression(max_iter=1000).fit(np.log(s_tr), y[tr]).predict_proba(np.log(s_te))[:, 1]
        for name, p in (("ACRM pipeline, refit within external corpus", pred), ("Strength-only logistic, within external corpus", pred_lr)):
            m = metrics(y, p, ext.gwp_per_ksi.to_numpy())
            rows.append({"model": name, "seed": seed, **m})
    out = pd.DataFrame(rows)
    summ = out.groupby("model")[["roc_auc", "average_precision", "top_10pct_capture", "top_20pct_capture", "spearman_prob_vs_ci"]].agg(["mean", "std"]).round(3)
    print("\nleave-producers-out CV within the external corpus (5 folds x 5 shuffles):")
    print(summ.to_string())
    R.write_table(out, f"external_{TAG}_within_cv.csv")


if __name__ == "__main__" and __import__("sys").argv[-1] == "--within":
    within_external_cv()
