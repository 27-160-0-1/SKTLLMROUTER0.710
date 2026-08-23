# SPDX-License-Identifier: Apache-2.0
"""a06 step 7: the decisive question -- how much WITHIN-family score signal exists?

For every family, fit a small GBM on the 35 interpretable features with 5-fold
OOF on train, report OOF corr with the realised label, and compare to the
noise ceiling sqrt(var_p/var_obs).  Then repeat with the deployed E43
predictions on dev for the same families (already in a06_deployed_gap.py) and
report the residual k1 log-cost bias per family / sub-family.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from a06_features import build  # noqa: E402
from a06_counterfactual import subfam  # noqa: E402


def cmax(y, g):
    mu, v = y.mean(), y.var()
    if v < 1e-12 or g <= 1:
        return 0.0
    vp = (v - (mu - mu * mu) / g) / (1 - 1 / g)
    return float(np.sqrt(max(vp, 0.0) / v))


def corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    tr, dv = build("train"), build("dev")
    sptr, spdv = tr["split"], dv["split"]
    print("A. within-family OOF (5-fold, train) prompt-feature score model vs noise ceiling")
    print("   'OOF r' = corr(pred, realised);  'ceil' = sqrt(var_p/var_obs)")
    print(f"{'family':16s} {'n':>4s} | " +
          " | ".join(f"{m:>5s} OOFr  ceil  r/ceil" for m in ("light", "mid", "k1")))
    for f in sorted(set(tr["fam"])):
        m = tr["fam"] == f
        if m.sum() < 40:
            continue
        Xf = tr["X"][m]
        row = f"{f:16s} {m.sum():4d} |"
        for j in range(3):
            yf = sptr.score[m, j]
            if np.std(yf) < 1e-9:
                row += "   const   ---    --- |"
                continue
            oof = np.zeros_like(yf)
            for a, b in KFold(5, shuffle=True, random_state=0).split(Xf):
                g = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=8,
                                                  min_samples_leaf=20, learning_rate=0.08,
                                                  random_state=0).fit(Xf[a], yf[a])
                oof[b] = g.predict(Xf[b])
            r = corr(oof, yf)
            c = cmax(yf, sptr.ngen[m, j].mean())
            row += f" {r:8.3f} {c:5.2f} {r/max(c,1e-6):6.2f} |"
        print(row)

    print("\nB. deployed E43 residual log-cost bias on dev  (mean log(pred/true))")
    P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
    C = P["cost_premium"]
    gdv = subfam(dv["fam"], dv["X"], dv["names"])
    print(f"{'group':16s} {'n':>4s} {'light':>8s} {'mid':>8s} {'k1':>8s} "
          f"{'sd log k1':>10s} {'true k1 share of light-budget':>30s}")
    lb = spdv.cost[:, 0].sum()
    for g in sorted(set(gdv)) + ["ALL"]:
        m = np.ones(len(spdv), bool) if g == "ALL" else (gdv == g)
        lr = np.log(C[m] / spdv.cost[m])
        print(f"{g:16s} {m.sum():4d} {lr[:,0].mean():8.3f} {lr[:,1].mean():8.3f} "
              f"{lr[:,2].mean():8.3f} {lr[:,2].std():10.3f} "
              f"{spdv.cost[m,2].sum()/lb:30.2f}")

    print("\nC. k1 cost decile vs realised gain (dev): is the expensive tail worth buying?")
    c2 = spdv.cost[:, 2]
    q = np.argsort(c2)
    print(f"{'decile':>7s} {'mean c2/light-budget*880':>24s} {'gain k1-mid':>12s} "
          f"{'gain k1-light':>13s} {'s_light':>8s} {'s_k1':>7s} {'top fams':>28s}")
    for d in range(10):
        idx = q[d * 88:(d + 1) * 88]
        fams = dict(zip(*np.unique(dv["fam"][idx], return_counts=True)))
        top = sorted(fams.items(), key=lambda kv: -kv[1])[:2]
        print(f"{d:7d} {c2[idx].sum()/lb:24.3f} "
              f"{(spdv.score[idx,2]-spdv.score[idx,1]).mean():12.3f} "
              f"{(spdv.score[idx,2]-spdv.score[idx,0]).mean():13.3f} "
              f"{spdv.score[idx,0].mean():8.3f} {spdv.score[idx,2].mean():7.3f} "
              f"{str(top):>28s}")

    print("\nD. same decile table on TRAIN (replication)")
    c2t = sptr.cost[:, 2]
    qt = np.argsort(c2t)
    lbt = sptr.cost[:, 0].sum()
    for d in range(10):
        idx = qt[d * 176:(d + 1) * 176]
        fams = dict(zip(*np.unique(tr["fam"][idx], return_counts=True)))
        top = sorted(fams.items(), key=lambda kv: -kv[1])[:2]
        print(f"{d:7d} {c2t[idx].sum()/lbt:24.3f} "
              f"{(sptr.score[idx,2]-sptr.score[idx,1]).mean():12.3f} "
              f"{(sptr.score[idx,2]-sptr.score[idx,0]).mean():13.3f} "
              f"{sptr.score[idx,0].mean():8.3f} {sptr.score[idx,2].mean():7.3f} "
              f"{str(top):>28s}")


if __name__ == "__main__":
    main()
