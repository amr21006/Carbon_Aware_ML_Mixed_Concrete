"""Gain-based attribution from the reported model.

The attribution table previously shipped with the figures came from an earlier
XGBoost pipeline on an older dataset version, so it did not describe the model
the manuscript reports. This refits the reported configuration on the temporal
training partition and extracts gain importance from that booster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

import pipeline_common as R
import run_acrm_model as A


def main() -> None:
    ctx = R.get_context()
    train_idx, _ = ctx.published_splits()["temporal_latest20"]
    args = R.acrm_args()

    pre = A.make_preprocessor(ctx.numeric_cols, ctx.categorical_cols, [],
                              max_text_features=args.max_text_features)
    x = pre.fit_transform(ctx.model_frame.iloc[train_idx])
    if not sparse.issparse(x):
        x = sparse.csr_matrix(x)
    y = np.asarray(ctx.y.iloc[train_idx])

    model = A.make_model(y, args)
    model = A.fit_model(model, x, y, args)

    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(x.shape[1])]
    gain = model.booster_.feature_importance(importance_type="gain")
    n = min(len(names), len(gain))
    fi = (pd.DataFrame({"feature": names[:n], "gain": gain[:n]})
            .sort_values("gain", ascending=False)
            .head(50)
            .reset_index(drop=True))
    R.write_table(fi, "acrm_feature_importance_top50.csv")

    print(f"\nlearner: {type(model).__name__}   features: {x.shape[1]:,}   "
          f"training declarations: {len(train_idx):,}")
    print("top 12 by gain:")
    for _, r in fi.head(12).iterrows():
        print(f"  {r.gain:12,.1f}  {r.feature}")


if __name__ == "__main__":
    main()
