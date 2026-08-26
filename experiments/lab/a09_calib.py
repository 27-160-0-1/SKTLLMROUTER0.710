# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a09: score reliability diagrams, ordinal-reconstruction shape, budget-variance sources."""
from __future__ import annotations
import sys
from pathlib import Path as FP
import numpy as np
sys.path.insert(0, str(FP(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, MODEL_IDS
from a09_harness import LPath, eval_cached, headroom_cached, make_W
ROOT = FP(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.similarity import classify_family
dv = load_split("dev"); n = len(dv)
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
fam = np.array([classify_family(t) for t in dv.texts])

print("=== score reliability (dev, tier=premium), 8 equal-count bins ===")
S = P["score_premium"]
for j, m in enumerate(MODEL_IDS):
    q = np.quantile(S[:, j], np.linspace(0, 1, 9)); q[0] -= 1e-9
    print(f"  {m}:")
    tot = 0.0
    for i in range(8):
        msk = (S[:, j] > q[i]) & (S[:, j] <= q[i + 1])
        d = dv.score[msk, j].mean() - S[msk, j].mean()
        tot += msk.sum() * d
        print(f"    bin{i+1} n={msk.sum():3d} pred={S[msk,j].mean():.3f} true={dv.score[msk,j].mean():.3f} bias={d:+.3f}")
    print(f"    overall bias = {tot/n:+.4f}")

print("\n=== ordinal reconstruction shape: E[s]=0.25*sum(sigmoid) ===")
for j, m in enumerate(MODEL_IDS):
    v = S[:, j]
    print(f"  {m:11s} frac<0.05 {np.mean(v<0.05):.3f} frac>0.95 {np.mean(v>0.95):.3f} "
          f"frac in [0.2,0.8] {np.mean((v>=0.2)&(v<=0.8)):.3f}  true: frac==0 {np.mean(dv.score[:,j]==0):.3f} "
          f"frac==1 {np.mean(dv.score[:,j]==1):.3f}")

print("\n=== where the budget-ratio variance comes from (premium, deployed-ish safety) ===")
from labdata import tier_result
for t in TIERS:
    C = P[f"cost_{t}"]; Sc = P[f"score_{t}"]
    sf = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}[t]
    r = tier_result(Sc, C, dv, t, sf); sel = r["sel"]; idx = np.arange(n)
    err = dv.cost[idx, sel] - C[idx, sel]          # per-item cost surprise (in cost units)
    tot_var = err.var() * n
    print(f"  {t:9s} sum surprise={err.sum():.3f} ({100*err.sum()/dv.cost[:,0].sum():.1f}% of light total)")
    for j, mm in enumerate(MODEL_IDS):
        msk = sel == j
        if msk.sum() == 0: continue
        print(f"      {mm:11s} n={msk.sum():4d} var share={100*err[msk].var()*msk.sum()/tot_var:5.1f}%  "
              f"mean surprise={err[msk].mean():+.5f} sd={err[msk].std():.5f}")
    # top-20 contributors
    o = np.argsort(-np.abs(err))[:20]
    print(f"      top-20 |surprise| items carry {100*np.abs(err[o]).sum()/np.abs(err).sum():.1f}% of total |surprise|; "
          f"families: {sorted(set(fam[o]))}")

print("\n=== how many k1 upgrades are there to buy? marginal value of budget (premium) ===")
Sc = P["score_premium"]; C = P["cost_premium"]
pth = LPath(Sc, C, dv.cost, dv.score)
W = np.ones((n, 1)); b = pth.batch(W)
for sf in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10):
    sc, ra, pa = eval_cached(b, 4.0, sf)
    g = (b["pred"] <= b["pl"] * max(1.0, 4.0 * sf)).argmax(axis=0)[0]
    cnt = np.bincount(pth.sel[g], minlength=3)
    print(f"  safety={sf:.2f} true ratio={ra[0]:.3f} score={sc[0]:.4f} sel={cnt}")
