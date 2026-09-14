"""Generate publication-quality figures for the concrete-EPD procurement-screening manuscript.

All figures are rendered at 300 DPI from the frozen result CSVs and per-row
prediction files in ``results/``. No model re-training is performed here; the
script consumes the validated outputs of the modelling pipeline so that every
figure traces to a source file.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)
RAW_V5 = ROOT / "data" / "raw" / "concrete_epd_mendeley_v5.csv"

# ---- global publication style (Okabe-Ito colour-blind-safe palette) ----------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})

C = {
    "acrm": "#0072B2",     # blue
    "xgb": "#D55E00",      # vermillion
    "carm": "#009E73",     # green
    "dlgbm": "#CC79A7",    # purple
    "gray": "#7A7A7A",
    "company": "#0072B2",
    "source": "#E69F00",
    "temporal": "#009E73",
    "pos": "#D55E00",
    "neg": "#8FB8DE",
}
SPLIT_LABEL = {
    "group_company": "Unseen company",
    "group_epd_source": "Unseen EPD source",
    "temporal_latest20": "Temporal (final 20%)",
}
SPLIT_COLOR = {
    "group_company": C["company"],
    "group_epd_source": C["source"],
    "temporal_latest20": C["temporal"],
}
TARGET_Q90 = 124.0


def save(fig, name):
    path = FIGS / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")


def bottom_caption(ax, text, dy=-38):
    """Place a bold panel caption (a)/(b)/... BELOW the panel (construction-journal style)."""
    ax.annotate(text, xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, dy), textcoords="offset points",
                ha="center", va="top", fontsize=10, fontweight="bold",
                annotation_clip=False)


# =============================================================================
# Fig 1 - Analytical framework
# =============================================================================
def fig_framework():
    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 24); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=9.2, tc="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.18",
                                    linewidth=1.3, edgecolor=ec, facecolor=fc, alpha=0.95))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                     linewidth=1.4, color="#555555"))

    blue, green, orange, purple, gray = "#DCE9F5", "#DCEFE6", "#FCE9D6", "#F1DCEC", "#ECECEC"
    eb, eg, eo, ep = "#0072B2", "#009E73", "#D55E00", "#CC79A7"

    box(2.3, 22.0, 5.4, 1.4, "Mendeley U.S. concrete EPD dataset\n(47,413 records, 2021 to 2025)", blue, eb)
    box(2.3, 20.0, 5.4, 1.3, "Quality filtering and 99.5th pct trimming\nto 46,917 modelling records", blue, eb)
    box(0.3, 17.6, 4.4, 1.7, "Strength normalised target\nCI = GWP(A1 to A3) / (fc / 1000)\ny = 1[CI >= 90th pct] (=124.0)", green, eg)
    box(5.3, 17.6, 4.4, 1.7, "Leakage control\nexclude 25 GWP and LCA outcome\ncolumns and derived target", orange, eo)
    box(2.3, 15.2, 5.4, 1.7, "Feature construction\nsupplier and plant, strength and curing,\nSCM flags, application, TFIDF text", blue, eb)
    box(0.3, 12.6, 4.4, 1.8, "Proposed model: ACRM\napplication aware single\nLightGBM learner", purple, ep)
    box(5.3, 12.6, 4.4, 1.8, "13 established and baseline learners\nXGBoost, CatBoost, RF, MLP,\nSVM, logistic, ...", gray, "#888888")
    box(1.6, 10.2, 6.8, 1.5, "Holdout validation\nunseen company | unseen EPD source | temporal", green, eg)
    box(1.6, 7.8, 6.8, 1.5, "Top k procurement screening\ncapture, precision, lift, review effort savings", blue, eb)
    box(1.6, 5.4, 6.8, 1.5, "Procurement opportunity analysis\ncomparable group lower carbon peers vs p25", orange, eo)
    box(1.6, 3.0, 6.8, 1.5, "Uncertainty and calibration\n1,000x bootstrap CIs, ECE, Brier", green, eg)
    box(2.3, 0.7, 5.4, 1.4, "Decision support for low carbon\nconcrete procurement", purple, ep, fs=9.6)

    arrow(5, 22.0, 5, 21.3)
    arrow(5, 20.0, 5, 19.35); arrow(4.7, 18.45, 5.3, 18.45)
    arrow(2.5, 17.6, 4.0, 16.9); arrow(7.5, 17.6, 6.0, 16.9)
    arrow(3.7, 15.2, 3.0, 14.4); arrow(6.3, 15.2, 7.0, 14.4)
    arrow(2.5, 12.6, 3.8, 11.7); arrow(7.5, 12.6, 6.2, 11.7)
    arrow(5, 10.2, 5, 9.3)
    arrow(5, 7.8, 5, 6.9)
    arrow(5, 5.4, 5, 4.5)
    arrow(5, 3.0, 5, 2.1)
    ax.set_title("Analytical framework for carbon aware EPD procurement screening",
                 fontsize=11.5, pad=6)
    save(fig, "fig01_framework.png")


# =============================================================================
# Fig 2 - Dataset & target profile
# =============================================================================
def load_v5_profile():
    df = pd.read_csv(RAW_V5, low_memory=False)
    gwp = pd.to_numeric(df["A1-A3 Global Warming Potential (kg CO2-eq)"], errors="coerce")
    strength = pd.to_numeric(df["Concrete Compressive Strength (psi)"], errors="coerce")
    date = pd.to_datetime(df["EPD Date of Issue"], errors="coerce")
    m = gwp.notna() & (gwp > 0) & strength.notna() & (strength > 0)
    base = pd.DataFrame({"gwp": gwp[m], "strength": strength[m], "year": date[m].dt.year})
    base["gwp_per_ksi"] = base["gwp"] / (base["strength"] / 1000.0)
    q = base["gwp"].quantile(0.995); qi = base["gwp_per_ksi"].quantile(0.995)
    base = base[(base["gwp"] < q) & (base["gwp_per_ksi"] < qi)].copy()
    base["high"] = (base["gwp_per_ksi"] >= base["gwp_per_ksi"].quantile(0.90)).astype(int)
    return base


def fig_profile():
    base = load_v5_profile()
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.4))

    ax = axes[0, 0]
    ax.hist(base["gwp_per_ksi"], bins=70, color=C["neg"], edgecolor="white", linewidth=0.3)
    ax.axvline(TARGET_Q90, color=C["pos"], linestyle="--", linewidth=1.8)
    ax.text(TARGET_Q90 + 4, ax.get_ylim()[1] * 0.86,
            f"90th pct = {TARGET_Q90:.0f}\n(high carbon)", color=C["pos"], fontsize=8.4)
    ax.set_xlabel("Carbon intensity, GWP A1 to A3 per ksi (kg CO2 eq)")
    ax.set_ylabel("Number of EPDs")
    ax.set_xlim(0, base["gwp_per_ksi"].quantile(0.995))
    bottom_caption(ax, "(a) Strength normalised carbon intensity")

    ax = axes[0, 1]
    ax.hist(base["strength"], bins=60, color="#8FB8DE", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Compressive strength (psi)"); ax.set_ylabel("Number of EPDs")
    ax.set_xlim(0, base["strength"].quantile(0.995))
    bottom_caption(ax, "(b) Declared compressive strength")

    ax = axes[1, 0]
    yr = base["year"].value_counts().sort_index()
    yr = yr[yr.index.notna()]
    ax.bar([int(i) for i in yr.index], yr.values, color=C["acrm"], edgecolor="white")
    ax.set_xlabel("EPD issue year"); ax.set_ylabel("Number of EPDs")
    for x, v in zip(yr.index, yr.values):
        ax.text(int(x), v + max(yr.values) * 0.01, f"{v:,}", ha="center", va="bottom", fontsize=7.5)
    bottom_caption(ax, "(c) EPD records by issue year")

    ax = axes[1, 1]
    hb = ax.hexbin(base["strength"], base["gwp_per_ksi"], gridsize=45, cmap="Blues",
                   bins="log", mincnt=1)
    ax.axhline(TARGET_Q90, color=C["pos"], linestyle="--", linewidth=1.6)
    ax.set_xlabel("Compressive strength (psi)")
    ax.set_ylabel("GWP A1 to A3 per ksi (kg CO2 eq)")
    ax.set_xlim(0, base["strength"].quantile(0.99))
    ax.set_ylim(0, base["gwp_per_ksi"].quantile(0.99))
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02); cb.set_label("log10 count", fontsize=8)
    bottom_caption(ax, "(d) Carbon intensity vs. strength")
    for a in axes.ravel():
        a.grid(alpha=0.2)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.96, bottom=0.10, hspace=0.62, wspace=0.28)
    save(fig, "fig02_data_target_profile.png")


# =============================================================================
# Fig 3 - Algorithm benchmark (ROC-AUC by algorithm across holdouts)
# =============================================================================
def fig_benchmark():
    cons = RESULTS / "manuscript_tables" / "algorithm_benchmark_full.csv"
    df = pd.read_csv(cons if cons.exists()
                     else RESULTS / "final_all_algorithm_metrics_with_acrm.csv")
    core = df[df["algorithm"].isin([
        "ACRM application-aware single model", "CARM-Boost fixed single model",
        "XGBoost GPU", "LightGBM GPU", "CatBoost CPU SVD", "HistGradientBoosting SVD",
        "Random Forest", "Extra Trees", "MLP SVD", "SGD logistic regression",
        "Linear SVM SGD", "Passive-Aggressive", "Ridge classifier", "Complement Naive Bayes",
    ])].copy()
    rename = {
        "ACRM application-aware single model": "ACRM (proposed)",
        "CARM-Boost fixed single model": "CARM-Boost",
        "XGBoost GPU": "XGBoost", "LightGBM GPU": "LightGBM", "CatBoost CPU SVD": "CatBoost",
        "HistGradientBoosting SVD": "HistGB", "Random Forest": "Random Forest",
        "Extra Trees": "Extra Trees", "MLP SVD": "MLP", "SGD logistic regression": "Logistic (SGD)",
        "Linear SVM SGD": "Linear SVM", "Passive-Aggressive": "Passive-Aggr.",
        "Ridge classifier": "Ridge", "Complement Naive Bayes": "Naive Bayes",
    }
    core["name"] = core["algorithm"].map(rename)
    # The broad benchmark of all learners was run on the unseen-company and
    # temporal holdouts; only the developed models add the unseen-source split.
    bench_splits = ["group_company", "temporal_latest20"]
    piv = core.pivot_table(index="name", columns="split", values="roc_auc", aggfunc="mean")
    piv = piv.dropna(subset=bench_splits)
    piv["order"] = piv[bench_splits].mean(axis=1)
    piv = piv.sort_values("order", ascending=True)
    algos = piv.index.tolist()
    y = np.arange(len(algos)); h = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    for i, sp in enumerate(bench_splits):
        ax.barh(y + (i - 0.5) * h, piv[sp].values, height=h, color=SPLIT_COLOR[sp],
                label=SPLIT_LABEL[sp], edgecolor="white", linewidth=0.3)
    ax.set_yticks(y); ax.set_yticklabels(algos)
    ax.set_xlim(0.78, 0.975); ax.set_xlabel("ROC-AUC")
    ax.set_title("Algorithm benchmark: ROC-AUC across leakage-resistant holdouts")
    for lab in ax.get_yticklabels():
        if "ACRM" in lab.get_text():
            lab.set_fontweight("bold"); lab.set_color(C["acrm"])
    ax.axvline(0.90, color=C["gray"], linestyle=":", linewidth=0.9, alpha=0.7)
    ax.legend(loc="lower right", ncol=1)
    ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    save(fig, "fig03_algorithm_benchmark.png")


# =============================================================================
# Prediction loaders
# =============================================================================
def acrm_preds():
    return pd.read_csv(RESULTS / "acrm_single_model_predictions.csv")


def val_preds():
    return pd.read_csv(RESULTS / "validation_predictions.csv")


# =============================================================================
# Fig 4 - ROC curves (ACRM across splits)
# =============================================================================
def fig_roc():
    ap = acrm_preds()
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for sp in ["group_company", "group_epd_source", "temporal_latest20"]:
        d = ap[ap["split"] == sp]
        fpr, tpr, _ = roc_curve(d["y_true"], d["acrm_probability"])
        auc = roc_auc_score(d["y_true"], d["acrm_probability"])
        ax.plot(fpr, tpr, color=SPLIT_COLOR[sp], linewidth=2.0,
                label=f"{SPLIT_LABEL[sp]} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=C["gray"], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ACRM ROC curves by validation holdout")
    ax.legend(loc="lower right"); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    save(fig, "fig04_acrm_roc.png")


# =============================================================================
# Fig 5 - Precision-Recall curves (ACRM across splits)
# =============================================================================
def fig_pr():
    ap = acrm_preds()
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for sp in ["group_company", "group_epd_source", "temporal_latest20"]:
        d = ap[ap["split"] == sp]
        prec, rec, _ = precision_recall_curve(d["y_true"], d["acrm_probability"])
        apr = average_precision_score(d["y_true"], d["acrm_probability"])
        ax.plot(rec, prec, color=SPLIT_COLOR[sp], linewidth=2.0,
                label=f"{SPLIT_LABEL[sp]} (AP = {apr:.3f})")
        base = d["y_true"].mean()
        ax.axhline(base, color=SPLIT_COLOR[sp], linestyle=":", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Recall (capture of high-carbon EPDs)"); ax.set_ylabel("Precision")
    ax.set_title("ACRM precision-recall curves by holdout")
    ax.legend(loc="upper right"); ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1.02)
    ax.text(0.02, 0.05, "dotted = class prevalence", color=C["gray"], fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig05_acrm_pr.png")


def fig_roc_pr():
    """Combined two-panel ROC and precision-recall figure (replaces separate ROC/PR)."""
    ap = acrm_preds()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9))
    ax = axes[0]
    for sp in ["group_company", "group_epd_source", "temporal_latest20"]:
        d = ap[ap["split"] == sp]
        fpr, tpr, _ = roc_curve(d["y_true"], d["acrm_probability"])
        auc = roc_auc_score(d["y_true"], d["acrm_probability"])
        ax.plot(fpr, tpr, color=SPLIT_COLOR[sp], linewidth=2.0,
                label=f"{SPLIT_LABEL[sp]} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=C["gray"], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right"); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    bottom_caption(ax, "(a) Receiver operating characteristic")

    ax = axes[1]
    for sp in ["group_company", "group_epd_source", "temporal_latest20"]:
        d = ap[ap["split"] == sp]
        prec, rec, _ = precision_recall_curve(d["y_true"], d["acrm_probability"])
        apr = average_precision_score(d["y_true"], d["acrm_probability"])
        ax.plot(rec, prec, color=SPLIT_COLOR[sp], linewidth=2.0,
                label=f"{SPLIT_LABEL[sp]} (AP = {apr:.3f})")
        ax.axhline(d["y_true"].mean(), color=SPLIT_COLOR[sp], linestyle=":", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="none")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1.02)
    bottom_caption(ax, "(b) Precision recall (dotted = class prevalence)")
    fig.tight_layout(); fig.subplots_adjust(bottom=0.20)
    save(fig, "fig_roc_pr.png")


# =============================================================================
# Fig 6 - Cumulative gains & lift (top-k screening)
# =============================================================================
def cumulative_gains(y_true, score):
    order = np.argsort(score)[::-1]
    y = np.asarray(y_true)[order]
    frac = np.arange(1, len(y) + 1) / len(y)
    capture = np.cumsum(y) / y.sum()
    return frac, capture


def fig_gains():
    ap = acrm_preds()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9))
    ax = axes[0]
    for sp in ["group_company", "group_epd_source", "temporal_latest20"]:
        d = ap[ap["split"] == sp]
        frac, cap = cumulative_gains(d["y_true"].values, d["acrm_probability"].values)
        ax.plot(frac * 100, cap * 100, color=SPLIT_COLOR[sp], linewidth=2.0,
                label=SPLIT_LABEL[sp])
    ax.plot([0, 100], [0, 100], color=C["gray"], linestyle="--", linewidth=1, label="Random review")
    ax.axvline(20, color="#444", linestyle=":", linewidth=1)
    ax.set_xlabel("EPDs reviewed (% of pool)")
    ax.set_ylabel("High carbon EPDs captured (%)")
    ax.legend(loc="lower right"); ax.set_xlim(0, 100); ax.set_ylim(0, 101)
    # annotate temporal at 20%
    d = ap[ap["split"] == "temporal_latest20"]
    frac, cap = cumulative_gains(d["y_true"].values, d["acrm_probability"].values)
    i20 = np.searchsorted(frac, 0.20)
    ax.scatter([20], [cap[i20] * 100], color=C["temporal"], zorder=5, s=30)
    ax.annotate(f"{cap[i20]*100:.1f}% at 20%", (20, cap[i20] * 100),
                textcoords="offset points", xytext=(6, -14), fontsize=8.2, color=C["temporal"])
    bottom_caption(ax, "(a) Cumulative gains (capture) curve")

    ax = axes[1]
    op = pd.read_csv(RESULTS / "acrm_single_model_operational_metrics.csv")
    ks = [5, 10, 20, 30]
    for sp in ["group_company", "group_epd_source", "temporal_latest20"]:
        r = op[op["split"] == sp].iloc[0]
        lifts = [r[f"top_{k}pct_lift"] for k in ks]
        ax.plot(ks, lifts, marker="o", color=SPLIT_COLOR[sp], linewidth=2.0,
                label=SPLIT_LABEL[sp])
    ax.axhline(1.0, color=C["gray"], linestyle="--", linewidth=1, label="Random (lift = 1)")
    ax.set_xlabel("Review workload (top k %)"); ax.set_ylabel("Lift over random review")
    ax.set_xticks(ks); ax.legend(loc="upper right")
    bottom_caption(ax, "(b) Screening lift by review workload")
    fig.tight_layout(); fig.subplots_adjust(bottom=0.20)
    save(fig, "fig06_topk_screening.png")


# =============================================================================
# Fig 7 - Calibration / reliability (ACRM vs XGBoost, temporal)
# =============================================================================
def fig_calibration():
    ap = acrm_preds(); vp = val_preds()
    da = ap[ap["split"] == "temporal_latest20"]
    dx = vp[vp["split"] == "temporal_latest20"]
    op = pd.read_csv(RESULTS / "operational_metrics_all_models_with_acrm.csv")

    def ece(row_name):
        r = op[(op["split"] == "temporal_latest20") & (op["algorithm"] == row_name)]
        return float(r["calibration_ece_10_bins"].iloc[0]) if len(r) else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for y, p, col, lab, e in [
        (da["y_true"], da["acrm_probability"], C["acrm"], "ACRM", ece("ACRM application-aware single model")),
        (dx["y_true"], dx["xgb_probability"], C["xgb"], "XGBoost baseline", ece("XGBoost GPU SOTA baseline")),
    ]:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", color=col, linewidth=1.8,
                label=f"{lab} (ECE = {e:.3f})")
    ax.plot([0, 1], [0, 1], color=C["gray"], linestyle="--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed high carbon frequency")
    ax.legend(loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    bottom_caption(ax, "(a) Reliability diagram (temporal holdout)")

    ax = axes[1]
    ax.hist(da["acrm_probability"], bins=40, color=C["acrm"], alpha=0.8, edgecolor="white",
            linewidth=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("ACRM predicted probability"); ax.set_ylabel("Count (log scale)")
    bottom_caption(ax, "(b) Predicted probability distribution")
    fig.tight_layout(); fig.subplots_adjust(bottom=0.20)
    save(fig, "fig07_calibration.png")


# =============================================================================
# Fig 8 - Procurement opportunity
# =============================================================================
def fig_opportunity():
    det = pd.read_csv(RESULTS / "procurement_opportunity_acrm_temporal_latest20_top20_details.csv")
    high = det[(det["y_true"] == 1) & (det["has_comparable_alternative_group"])].copy()
    grp = pd.read_csv(RESULTS / "procurement_opportunity_acrm_temporal_latest20_top20_by_group.csv")
    grp = grp[grp["flagged_true_high_carbon_count"] >= 5].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9))
    ax = axes[0]
    vals = high["relative_reduction_to_p25"] * 100
    ax.hist(vals, bins=40, color=C["temporal"], edgecolor="white", linewidth=0.3, alpha=0.9)
    med = vals.median()
    ax.axvline(med, color=C["pos"], linestyle="--", linewidth=1.8)
    ax.text(med + 1.5, ax.get_ylim()[1] * 0.85, f"median = {med:.1f}%", color=C["pos"], fontsize=8.6)
    ax.axvline(20, color="#444", linestyle=":", linewidth=1)
    ax.set_xlabel("Relative lower carbon opportunity vs. comparable group p25 (%)")
    ax.set_ylabel("Captured high carbon EPDs")
    bottom_caption(ax, "(a) Opportunity distribution (temporal, top 20%)")

    ax = axes[1]
    grp = grp.sort_values("flagged_true_high_carbon_count", ascending=True)
    names = [g.replace("_", " ").title() for g in grp["application_family"]]
    yy = np.arange(len(grp))
    ax.barh(yy, grp["median_relative_reduction_to_p25_true_high_carbon"] * 100,
            color=C["acrm"], edgecolor="white", alpha=0.9)
    ax.set_yticks(yy); ax.set_yticklabels(names)
    for i, (_, r) in enumerate(grp.iterrows()):
        ax.text(r["median_relative_reduction_to_p25_true_high_carbon"] * 100 + 0.6, i,
                f"n={int(r['flagged_true_high_carbon_count'])}", va="center", fontsize=7.6,
                color="#444")
    ax.set_xlabel("Median opportunity vs. p25 (%)")
    bottom_caption(ax, "(b) Opportunity by application family")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.24)
    save(fig, "fig08_procurement_opportunity.png")


# =============================================================================
# Fig 9 - Bootstrap uncertainty (forest plot, temporal)
# =============================================================================
def fig_uncertainty():
    unc = pd.read_csv(RESULTS / "acrm_uncertainty_analysis.csv")
    t = unc[unc["split"] == "temporal_latest20"]
    rows = []
    def add(rf, metric, label, atype="topk_prediction", scale=100):
        r = t[(t["review_fraction"] == rf) & (t["analysis_type"] == atype) & (t["metric"] == metric)]
        if len(r):
            r = r.iloc[0]
            rows.append((label, r["point"] * scale, r["ci_low"] * scale, r["ci_high"] * scale))
    add(0.20, "topk_recall_capture", "Capture @ top-20%")
    add(0.10, "topk_recall_capture", "Capture @ top-10%")
    add(0.20, "topk_precision", "Precision @ top-20%")
    add(0.10, "topk_precision", "Precision @ top-10%")
    add(0.20, "review_savings_fraction_vs_random", "Review-effort saving @ top-20%")
    add(0.20, "median_relative_reduction_to_p25_true_high_carbon",
        "Median opportunity vs p25 @ top-20%", atype="procurement_opportunity")
    add(0.20, "share_true_high_carbon_with_10pct_opportunity",
        "Share >=10% opportunity @ top-20%", atype="procurement_opportunity")

    labels = [r[0] for r in rows]
    pts = [r[1] for r in rows]
    lo = [r[1] - r[2] for r in rows]
    hi = [r[3] - r[1] for r in rows]
    yy = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.errorbar(pts, yy, xerr=[lo, hi], fmt="o", color=C["acrm"], ecolor=C["temporal"],
                elinewidth=2, capsize=4, markersize=7)
    for x, y in zip(pts, yy):
        ax.text(x, y + 0.16, f"{x:.1f}%", ha="center", fontsize=8.2, color="#333")
    ax.set_yticks(yy); ax.set_yticklabels(labels)
    ax.set_xlabel("Value (%) with 95% bootstrap confidence interval (1,000 resamples)")
    ax.set_title("ACRM temporal screening: point estimates and uncertainty")
    ax.set_xlim(0, 100); ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    save(fig, "fig09_uncertainty_forest.png")


# =============================================================================
# Fig 10 - Feature importance
# =============================================================================
def fig_importance():
    fi = pd.read_csv(RESULTS / "feature_importance_top50.csv").head(20)
    def clean(n):
        n = n.replace("num__", "").replace("cat__", "").replace("text__", "'")
        if n.startswith("'"):
            n = n + "' (text)"
        n = (n.replace("strength_psi", "compressive strength")
              .replace("strength_bin_500", "strength bin (500 psi)")
              .replace("Plant Location - State_", "plant state = ")
              .replace("Company_", "company = "))
        n = n.replace("_", " ")
        n = re.sub(r"(?<=\w)-(?=\w)", " ", n)   # no residual hyphens
        return n
    fi["label"] = fi["feature"].map(clean)
    fi = fi.sort_values("gain", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    colors = [C["acrm"] if ("strength" in l or "state" in l or "company" in l)
              else "#8FB8DE" for l in fi["label"]]
    ax.barh(np.arange(len(fi)), fi["gain"], color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(np.arange(len(fi))); ax.set_yticklabels(fi["label"], fontsize=8)
    ax.set_xlabel("Gain importance (XGBoost booster)")
    ax.set_title("Top 20 predictors of high carbon concrete risk")
    ax.grid(axis="y", alpha=0)
    handles = [mpatches.Patch(color=C["acrm"], label="structured metadata"),
               mpatches.Patch(color="#8FB8DE", label="EPD text (TFIDF token)")]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    save(fig, "fig10_feature_importance.png")


def main():
    print("Generating manuscript figures ...")
    fig_framework()        # Fig 1
    fig_profile()          # Fig 2
    fig_roc_pr()           # Fig 3 (merged ROC + PR)
    fig_gains()            # Fig 4 (top-k screening)
    fig_calibration()      # Fig 5
    fig_opportunity()      # Fig 6
    fig_importance()       # Fig 7
    print("Done.")


if __name__ == "__main__":
    main()
