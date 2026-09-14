"""Effort-matched comparison against the achievable ceiling.

A reviewer who opens every declaration and computes its strength-normalized
carbon intensity identifies every high-carbon mix, because that quantity is the
screening target itself. The relevant question is therefore not whether a model
beats a human on judgement, but how much of that ceiling a screen reaches
without opening any declaration at all. This script computes the ceiling, the
screen, and the rules available before a declaration is opened, on one scale.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import pipeline_common as R

PRED = R.RESULTS_DIR / "acrm_predictions_enriched.csv"
HEUR = R.RESULTS_DIR / "practice_heuristic_baselines.csv"
FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50]
SPLIT = {"group_company": "Unseen company", "group_epd_source": "Unseen EPD source",
         "temporal_latest20": "Temporal"}


def capture_at(y: np.ndarray, score: np.ndarray, frac: float) -> float:
    n = len(y)
    k = max(int(round(n * frac)), 1)
    order = np.argsort(-score, kind="stable")[:k]
    pos = float(y.sum())
    return float(y[order].sum() / pos) if pos else float("nan")


def main() -> None:
    pred = pd.read_csv(PRED)
    rows: list[dict[str, Any]] = []
    for split, g in pred.groupby("split"):
        y = g["y_true"].to_numpy()
        prev = float(y.mean())
        for frac in FRACTIONS:
            # Perfect ordering: every positive precedes every negative.
            ceiling = min(1.0, frac / prev) if prev else float("nan")
            model = capture_at(y, g["proba"].to_numpy(), frac)
            rows.append({
                "split": split, "review_fraction": frac, "prevalence": prev,
                "ceiling_capture": ceiling,
                "model_capture": model,
                "share_of_ceiling_realized": model / ceiling if ceiling else float("nan"),
            })
    frame = pd.DataFrame(rows)
    R.write_table(frame.round(4), "ceiling_comparison.csv")

    print("=== capture against the achievable ceiling ===")
    for split in ["temporal_latest20", "group_company", "group_epd_source"]:
        s = frame[frame.split == split]
        print(f"\n{SPLIT[split]} (prevalence {s.prevalence.iloc[0]:.4f})")
        print(f"  {'budget':>7s} {'ceiling':>8s} {'model':>8s} {'% of ceiling':>13s}")
        for _, r in s.iterrows():
            print(f"  {r.review_fraction:7.0%} {r.ceiling_capture:8.3f} {r.model_capture:8.3f} "
                  f"{r.share_of_ceiling_realized:12.1%}")

    # One table combining the ceiling, the screen, and the pre-award rules.
    if HEUR.exists():
        h = pd.read_csv(HEUR)
        out: list[dict[str, Any]] = []
        for split in ["temporal_latest20", "group_company", "group_epd_source"]:
            s = frame[(frame.split == split) & (frame.review_fraction == 0.20)].iloc[0]
            out.append({"split": SPLIT[split], "strategy":
                        "Complete manual review (every declaration opened)",
                        "information_needed": "Full declaration",
                        "review_effort": "100%", "capture_at_20pct": 1.0})
            out.append({"split": SPLIT[split], "strategy":
                        "Perfect ordering at a 20% budget (upper bound)",
                        "information_needed": "Outcome known",
                        "review_effort": "20%", "capture_at_20pct": s.ceiling_capture})
            best = h[(h.split == split) & (h.strategy != "ACRM (proposed)")]
            best = best.loc[best["top_20pct_capture"].idxmax()]
            out.append({"split": SPLIT[split], "strategy": f"Best pre-award rule ({best.strategy})",
                        "information_needed": "Pre-award metadata",
                        "review_effort": "20%", "capture_at_20pct": float(best["top_20pct_capture"])})
            acrm = h[(h.split == split) & (h.strategy == "ACRM (proposed)")].iloc[0]
            out.append({"split": SPLIT[split], "strategy": "ACRM (proposed)",
                        "information_needed": "Pre-award metadata",
                        "review_effort": "20%", "capture_at_20pct": float(acrm["top_20pct_capture"])})
        R.write_table(pd.DataFrame(out).round(4), "ceiling_vs_practice.csv")
        print("\n=== effort-matched summary (20% budget) ===")
        for _, r in pd.DataFrame(out).iterrows():
            print(f"  {r.split:20s} {r.strategy[:52]:52s} {r.review_effort:>5s} "
                  f"{r.capture_at_20pct:6.3f}")

    R.write_manifest({
        "analysis": "effort-matched comparison against the achievable ceiling",
        "note": "The ceiling at a budget above the prevalence is complete capture, because a "
                "perfect ordering places every positive first.",
    }, "ceiling_manifest.json")


if __name__ == "__main__":
    main()
