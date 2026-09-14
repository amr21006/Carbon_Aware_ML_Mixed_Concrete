"""Export a compact whole-sample key table for the revision analyses.

The decision-analytic work needs training-side information (supplier carbon
history, declared GWP, mixture flags) for rows that are not in any test
partition, so the full modelling sample is exported once in a small file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import pipeline_common as R
import run_acrm_model as A


def main() -> None:
    ctx = R.get_context()
    b = ctx.base
    flags = [c for c in b.columns if c.startswith("has_")]
    frame = pd.DataFrame(
        {
            "row_index": np.arange(ctx.n),
            "y_true": ctx.y.to_numpy(),
            "company": ctx.company.to_numpy(),
            "plant": ctx.plant.to_numpy(),
            "source": ctx.source.to_numpy(),
            "operator": ctx.operator.to_numpy(),
            "region": ctx.region.to_numpy(),
            "strength_psi": ctx.strength.to_numpy(),
            "strength_bin_500": ctx.strength_bin.to_numpy(),
            "curing_days": pd.to_numeric(b["curing_days"], errors="coerce").to_numpy(),
            "gwp": ctx.gwp.to_numpy(),
            "gwp_per_ksi": ctx.gwp_per_ksi.to_numpy(),
            "issue_date": ctx.issue_date.to_numpy(),
            "application_family": A.make_application_family(b).to_numpy(),
            "curing_family": A.make_curing_family(b["curing_days"]).to_numpy(),
        }
    )
    for col in flags:
        frame[col] = pd.to_numeric(b[col], errors="coerce").fillna(0).astype(int).to_numpy()
    frame["scm_any"] = frame[[c for c in flags if c in
                              ("has_fly_ash", "has_slag", "has_silica_fume",
                               "has_limestone_cement", "has_carbon_cure", "has_recycled")]].max(axis=1)

    R.write_table(frame, "sample_keys.csv")

    splits = ctx.published_splits()
    rows = []
    for name, (train_idx, test_idx) in splits.items():
        rows.append({
            "split": name,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "test_prevalence": float(ctx.y.iloc[test_idx].mean()),
            "n_company_train": int(ctx.company.iloc[train_idx].nunique()),
            "n_company_test": int(ctx.company.iloc[test_idx].nunique()),
            "n_plant_test": int(ctx.plant.iloc[test_idx].nunique()),
            "n_source_test": int(ctx.source.iloc[test_idx].nunique()),
            "company_overlap": int(
                len(set(ctx.company.iloc[train_idx]) & set(ctx.company.iloc[test_idx]))
            ),
            "source_overlap": int(
                len(set(ctx.source.iloc[train_idx]) & set(ctx.source.iloc[test_idx]))
            ),
        })
        np.save(R.RESULTS_DIR / f"rev_split_test_{name}.npy", np.asarray(test_idx))
    R.write_table(pd.DataFrame(rows), "split_profile.csv")


if __name__ == "__main__":
    main()
