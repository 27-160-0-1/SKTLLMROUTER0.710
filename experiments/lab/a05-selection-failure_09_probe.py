# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a05 step 9: is the mid-vs-light decision doing ANY work?
(a) monotonicity of realised gain across predicted-gain deciles among the picks
(b) counterfactual: keep the deployed per-family mid budget, reorder within family
(c) is item difficulty one-dimensional (does g2 predict g1)?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
ROOT = Path(__file__).resolve().parents[2]

dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
Cc = np.load(ROOT / "experiments/lab/a05-selection-failure_cache.npz", allow_pickle=True)
N = len(dv); IDX = np.arange(N); FAM = Cc["fam"]; PHAT = Cc["phat"]
L = dv.cost[:, 0].sum()

print("=== (a) among the items the deployed router upgraded to MID, does the predicted")
print("        gain order the realised gain?  (premium tier, quintiles of predicted gain) ===")
for t in TIERS:
    sd = Cc[f"sel_d_{t}"]; S = P[f"score_{t}"]
    m = np.where(sd == 1)[0]
    g = S[m, 1] - S[m, 0]
    o = np.argsort(g)
    print(f"  {t}: n={len(m)}")
    for q in range(5):
        blk = o[q * len(m) // 5:(q + 1) * len(m) // 5]
        ii = m[blk]
        print(f"    Q{q+1} pred gain {g[blk].mean():+.3f}  realised gain {(dv.score[ii,1]-dv.score[ii,0]).mean():+.4f}"
              f"  EB gain {(PHAT[ii,1]-PHAT[ii,0]).mean():+.4f}  n={len(ii)}")

print("\n  ... same for the K1 picks (premium only):")
sd = Cc["sel_d_premium"]; S = P["score_premium"]
m = np.where(sd == 2)[0]
g = S[m, 2] - S[m, 1]
o = np.argsort(g)
for q in range(5):
    blk = o[q * len(m) // 5:(q + 1) * len(m) // 5]
    ii = m[blk]
    print(f"    Q{q+1} pred gain(k1-mid) {g[blk].mean():+.3f}  realised gain(k1-light) "
          f"{(dv.score[ii,2]-dv.score[ii,0]).mean():+.4f}  EB {(PHAT[ii,2]-PHAT[ii,0]).mean():+.4f} n={len(ii)}")

print("\n=== (b) keep the deployed per-family upgrade counts, reorder WITHIN family ===")
print("    (identical family mix and near-identical budget; only the item choice changes)")
rng = np.random.default_rng(7)
for t in TIERS:
    sd = Cc[f"sel_d_{t}"]; S = P[f"score_{t}"]
    base_sc = dv.score[IDX, sd].mean()
    base_c = dv.cost[IDX, sd].sum() / L
    out = {}
    for key, mk in (
        ("deployed order (pred gain/cost)", None),
        ("reverse of deployed key", None),
        ("cheapest true cost first", None),
        ("random", None),
    ):
        pass
    keys = {}
    eff1 = (S[:, 1] - S[:, 0]) / np.maximum(P[f"cost_{t}"][:, 1] - P[f"cost_{t}"][:, 0], 1e-9)
    eff2 = (S[:, 2] - S[:, 1]) / np.maximum(P[f"cost_{t}"][:, 2] - P[f"cost_{t}"][:, 1], 1e-9)
    keys["deployed efficiency"] = (eff1, eff2)
    keys["reversed efficiency"] = (-eff1, -eff2)
    keys["random"] = (rng.normal(size=N), rng.normal(size=N))
    keys["cheapest-first (true cost)"] = (-dv.cost[:, 1], -dv.cost[:, 2])
    print(f"  --- {t}: deployed score {base_sc:.4f} (EB {PHAT[IDX,sd].mean():.4f}) ratio {base_c:.3f}")
    for knm, (k1, k2) in keys.items():
        sel = np.zeros(N, int)
        for f_ in sorted(set(FAM.tolist())):
            mf = np.where(FAM == f_)[0]
            n1 = int((sd[mf] == 1).sum()); n2 = int((sd[mf] == 2).sum())
            # assign k1 to the top-n2 by k2, then mid to the next-n1 by k1
            ordk = mf[np.argsort(-k2[mf])]
            pick2 = ordk[:n2]
            rest = np.array([i for i in mf if i not in set(pick2.tolist())], dtype=int)
            ordm = rest[np.argsort(-k1[rest])] if len(rest) else rest
            pick1 = ordm[:n1]
            sel[pick2] = 2; sel[pick1] = 1
        sc = dv.score[IDX, sel].mean(); eb = PHAT[IDX, sel].mean()
        rr = dv.cost[IDX, sel].sum() / L
        print(f"    {knm:28s} score={sc:.4f} EB={eb:.4f} ratio={rr:.3f} "
              f"{'(OVER BUDGET)' if rr > TIER_MULT[t] else ''}")

print("\n=== (c) is item difficulty one-dimensional? ===")
g1 = dv.score[:, 1] - dv.score[:, 0]
g2 = dv.score[:, 2] - dv.score[:, 1]
gk = dv.score[:, 2] - dv.score[:, 0]
e1 = PHAT[:, 1] - PHAT[:, 0]; e2 = PHAT[:, 2] - PHAT[:, 1]
print(f"  corr(realised g1, realised g2) = {np.corrcoef(g1, g2)[0,1]:+.3f}   "
      f"corr(EB g1, EB g2) = {np.corrcoef(e1, e2)[0,1]:+.3f}")
print(f"  corr(realised g1, s_light)     = {np.corrcoef(g1, dv.score[:,0])[0,1]:+.3f}   "
      f"corr(EB g1, EB p_light) = {np.corrcoef(e1, PHAT[:,0])[0,1]:+.3f}")
print("\n  among the 335 items where light scores 0, who rescues?")
m0 = dv.score[:, 0] <= 0
print(f"    n(light=0) = {m0.sum()}")
print(f"    mid rescues (s_M>0)          : {int((dv.score[m0,1]>0).sum())}")
print(f"    k1  rescues (s_K>0)          : {int((dv.score[m0,2]>0).sum())}")
print(f"    both                         : {int(((dv.score[m0,1]>0)&(dv.score[m0,2]>0)).sum())}")
print(f"    mid only                     : {int(((dv.score[m0,1]>0)&(dv.score[m0,2]<=0)).sum())}")
print(f"    P(mid rescues | k1 rescues)  = {((dv.score[m0,1]>0)&(dv.score[m0,2]>0)).sum()/max((dv.score[m0,2]>0).sum(),1):.3f}")
print(f"    P(mid rescues | k1 fails)    = {((dv.score[m0,1]>0)&(dv.score[m0,2]<=0)).sum()/max((dv.score[m0,2]<=0).sum(),1):.3f}")
print("\n  per family: P(mid rescues | light=0)  and  P(k1 rescues | light=0)")
for f_ in sorted(set(FAM.tolist())):
    mm = m0 & (FAM == f_)
    if mm.sum() == 0:
        continue
    print(f"    {f_:15s} n={mm.sum():4d}  P(mid)={np.mean(dv.score[mm,1]>0):.3f}  "
          f"P(k1)={np.mean(dv.score[mm,2]>0):.3f}")
