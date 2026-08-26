# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a09: oracle ladder for cost calibration granularity + selection-bias decomposition."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.similarity import classify_family

dv = load_split("dev"); n = len(dv)
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
fam = np.array([classify_family(t) for t in dv.texts]); fams = sorted(set(fam))
famid = np.array([fams.index(f) for f in fam])

def cal_permodel(C):
    return C * (dv.cost.sum(0) / C.sum(0))[None, :]

def cal_perfamily(C):
    C = C.copy()
    for k in range(len(fams)):
        m = famid == k
        C[m] *= (dv.cost[m].sum(0) / C[m].sum(0))[None, :]
    return C

def cal_duan_permodel(C):
    r = np.log(dv.cost) - np.log(C)
    return C * np.exp(r).mean(0)[None, :]

VAR = {
    "base":            lambda C: C,
    "per-model sum":   cal_permodel,
    "per-fam x model": cal_perfamily,
    "true cost":       lambda C: dv.cost,
}

def sweep(name, f):
    dep = 0.0; best = 0.0; rows=[]
    for t in TIERS:
        S = P[f"score_{t}"]; C = f(P[f"cost_{t}"])
        r = tier_result(S, C, dv, t, SAFE[t]); dep += TIER_WEIGHT[t]*r["tier_score"]
        b = None
        for sf in np.arange(0.30, 1.601, 0.005):
            rr = tier_result(S, C, dv, t, float(sf))
            if rr["passed"] and (b is None or rr["score"] > b[1]): b = (float(sf), rr["score"])
        best += TIER_WEIGHT[t]*b[1]
        rows.append(f"{t[:4]}:dep {r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else '!BUST'} best {b[1]:.4f}@{b[0]:.3f}")
    print(f"{name:18s} dep-safety={dep:.4f}  best-safety={best:.4f}   " + "  ".join(rows))

print("=== calibration granularity ladder (dev, oracle factors) ===")
for k,v in VAR.items(): sweep(k, v)

print()
print("=== selection-induced cost bias decomposition (premium, deployed safety) ===")
for t in TIERS:
    S=P[f"score_{t}"]; C=P[f"cost_{t}"]
    res = tier_result(S,C,dv,t,SAFE[t]); sel=res["sel"]; idx=np.arange(n)
    print(f" {t}:")
    for j,m in enumerate(MODEL_IDS):
        msk = sel==j
        if msk.sum()==0: continue
        Rall = dv.cost[:,j].sum()/C[:,j].sum()
        Rsel = dv.cost[msk,j].sum()/C[msk,j].sum()
        print(f"   {m:11s} nsel={msk.sum():4d}  R(all items)={Rall:.4f}  R(selected)={Rsel:.4f}  "
              f"share of pred spend={C[msk,j].sum()/C[idx,sel].sum():.3f}")

print()
print("=== gain-calibration diagnostic: E[true delta | pred delta decile] ===")
t="premium"; S=P[f"score_{t}"]
for (a,b,lab) in ((0,1,"mid-light"),(1,2,"k1-mid")):
    pg = S[:,b]-S[:,a]; tg = dv.score[:,b]-dv.score[:,a]
    q = np.quantile(pg, np.linspace(0,1,11)); q[0]-=1e-9
    print(f"  {lab}:")
    for i in range(10):
        m = (pg>q[i])&(pg<=q[i+1])
        print(f"    d{i+1}: n={m.sum():3d} pred={pg[m].mean():+.4f} true={tg[m].mean():+.4f} diff={tg[m].mean()-pg[m].mean():+.4f}")
    neg = pg<0
    print(f"    pred<0: n={neg.sum()} pred={pg[neg].mean():+.4f} true={tg[neg].mean():+.4f}")
