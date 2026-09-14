"""Figures added for the JCCE revision.

Style matches make_figures.py so the new panels sit alongside the
existing ones without a visible change of hand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

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
    "acrm": "#0072B2",
    "xgb": "#D55E00",
    "gray": "#7A7A7A",
    "company": "#0072B2",
    "source": "#E69F00",
    "temporal": "#009E73",
}
RATIO_COLOR = {5: "#8FB8DE", 10: "#0072B2", 20: "#009E73", 50: "#CC79A7"}


def save(fig, name: str) -> None:
    path = FIGS / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")


def fig_decision(split: str = "temporal_latest20") -> None:
    dca = pd.read_csv(RESULTS / "decision_curve.csv")
    cost = pd.read_csv(RESULTS / "cost_sensitive_budget.csv")
    d = dca[dca["split"] == split]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9))

    # (a) decision curve
    ax = axes[0]
    acrm = d[d["strategy"] == "ACRM"].sort_values("threshold_probability")
    ax.plot(acrm["threshold_probability"], acrm["net_benefit"],
            color=C["acrm"], linewidth=2.2, label="ACRM")
    xgb = d[d["strategy"] == "XGBoost baseline"].sort_values("threshold_probability")
    if not xgb.empty:
        ax.plot(xgb["threshold_probability"], xgb["net_benefit"],
                color=C["xgb"], linewidth=2.0, linestyle="--", label="XGBoost baseline")
    ax.plot(acrm["threshold_probability"], acrm["net_benefit_review_all"],
            color=C["gray"], linewidth=1.4, linestyle="-.", label="Review all")
    ax.axhline(0.0, color=C["gray"], linewidth=1.2, linestyle=":", label="Review none")
    ax.set_xlim(0.02, 0.60)
    ax.set_ylim(-0.20, 0.09)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("(a) Decision curve", loc="left")
    ax.legend(loc="lower left")

    # (b) expected review cost against budget, by cost ratio
    ax = axes[1]
    c = cost[(cost["split"] == split) & (cost.get("is_optimum").isna()
             if "is_optimum" in cost.columns else True)]
    n_test = None
    for ratio in [5, 10, 20, 50]:
        sub = c[c["cost_ratio_missed_to_review"] == ratio].sort_values("review_fraction")
        if sub.empty:
            continue
        scale = sub["cost_review_all"].iloc[0]
        n_test = scale
        y = sub["cost_model"] / scale * 100
        ax.plot(sub["review_fraction"] * 100, y, color=RATIO_COLOR[ratio],
                linewidth=2.0, label=f"cost ratio {ratio}:1")
        j = int(np.argmin(y.to_numpy()))
        ax.plot(sub["review_fraction"].iloc[j] * 100, y.iloc[j], marker="o",
                color=RATIO_COLOR[ratio], markersize=5.5, zorder=5)
    ax.set_xlabel("Review budget (% of declarations reviewed)")
    ax.set_ylabel("Expected cost per 100 declarations\n(review = 1 unit)")
    ax.set_title("(b) Cost-sensitive review budget", loc="left")
    ax.legend(loc="upper right", title="missed mix : one review", title_fontsize=8)
    if n_test:
        ax.text(0.97, 0.46, "markers = cost-optimal budget", transform=ax.transAxes,
                fontsize=7.5, color=C["gray"], ha="right")

    fig.tight_layout()
    save(fig, "fig08_decision_analysis.png")


def fig_uncertainty() -> None:
    """Interval width under row-level versus cluster resampling."""
    ci = pd.read_csv(RESULTS / "clustered_bootstrap_ci.csv")
    metrics = {
        "top_20pct_capture": "Capture @20%",
        "top_20pct_precision": "Precision @20%",
        "top_20pct_review_saving": "Review saving @20%",
    }
    schemes = ["row_level", "cluster_plant", "cluster_source", "cluster_company"]
    labels = ["Row level", "Cluster: plant", "Cluster: source", "Cluster: company"]
    splits = ["group_company", "group_epd_source", "temporal_latest20"]
    split_label = {"group_company": "Unseen company",
                   "group_epd_source": "Unseen EPD source",
                   "temporal_latest20": "Temporal"}
    split_color = {"group_company": C["company"], "group_epd_source": C["source"],
                   "temporal_latest20": C["temporal"]}

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.2), sharey=True)
    for ax, (metric, title) in zip(axes, metrics.items()):
        for si, split in enumerate(splits):
            sub = ci[(ci["split"] == split) & (ci["metric"] == metric)]
            xs, ys = [], []
            for k, scheme in enumerate(schemes):
                r = sub[sub["resampling"] == scheme]
                if r.empty:
                    continue
                xs.append(k + (si - 1) * 0.22)
                ys.append(float(r["ci_width"].iloc[0]) * 100)
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.6,
                    color=split_color[split], label=split_label[split])
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Width of 95% interval (percentage points)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig09_interval_width.png")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    print("building revision figures")
    fig_decision()
    fig_uncertainty()


if __name__ == "__main__":
    main()
