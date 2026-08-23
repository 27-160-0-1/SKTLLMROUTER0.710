# SPDX-License-Identifier: Apache-2.0
"""a06 step 14: apples-to-apples -- train a within-family model on TRAIN, score it
on DEV, and compare with the deployed E43 held-out dev predictions in the same
family, same split, same metric."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from a06_features import build  # noqa: E402


def corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


tr, dv = build("train"), build("dev")
sptr, spdv = tr["split"], dv["split"]
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
S = P["score_premium"]

print("dev correlation with the realised score, per family")
print(f"{'family':16s} {'n_dv':>5s} | " +
      " | ".join(f"{m:>5s} dense  E43  blend.5" for m in ("light", "mid", "k1")))
blend_all = np.zeros_like(S)
for f in sorted(set(dv["fam"])):
    mt, md = tr["fam"] == f, dv["fam"] == f
    row = f"{f:16s} {md.sum():5d} |"
    for j in range(3):
        y = sptr.score[mt, j]
        if np.std(y) < 1e-9:
            blend_all[md, j] = S[md, j]
            row += "   const  ---     --- |"
            continue
        g = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=8,
                                          min_samples_leaf=20, learning_rate=0.08,
                                          random_state=0).fit(tr["X"][mt], y)
        p = g.predict(dv["X"][md])
        bl = 0.5 * p + 0.5 * S[md, j]
        blend_all[md, j] = bl
        row += (f" {corr(p, spdv.score[md,j]):8.3f} {corr(S[md,j], spdv.score[md,j]):5.3f}"
                f" {corr(bl, spdv.score[md,j]):7.3f} |")
    print(row)

print("\npooled dev corr")
print("  dense-within-family blend .5 :",
      [round(corr(blend_all[:, j], spdv.score[:, j]), 3) for j in range(3)])
print("  deployed E43                 :",
      [round(corr(S[:, j], spdv.score[:, j]), 3) for j in range(3)])
