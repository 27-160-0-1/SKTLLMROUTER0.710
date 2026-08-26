# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a09: bust probability / EV at the deployed safety values, and the safety exchange rate."""
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
fam = np.array([classify_family(t) for t in dv.texts]); fams = sorted(set(fam))
famid = np.array([fams.index(f) for f in fam])
SEEDS = (7, 17, 23); Ws = {s: make_W(n, 400, s) for s in SEEDS}
half = np.random.default_rng(0).integers(0, 2, size=n)

def sig2_perfam(C, shrink=20.0):
    r = np.log(dv.cost)-np.log(C); V = np.zeros_like(C)
    for h in (0, 1):
        a = half != h; b = half == h
        for j in range(3):
            g = r[a, j].var()
            for f in range(len(fams)):
                aa = a & (famid == f); bb = b & (famid == f)
                if bb.sum() == 0: continue
                if aa.sum() < 8: V[bb, j] = g; continue
                w = aa.sum()/(aa.sum()+shrink); V[bb, j] = w*r[aa, j].var() + (1-w)*g
    return V

def sig2_permodel(C):
    r = np.log(dv.cost)-np.log(C); V = np.zeros_like(C)
    for h in (0, 1):
        a = half != h; b = half == h
        for j in range(3): V[b, j] = r[a, j].var()
    return V

VAR = {"base": lambda C: C,
       "permodel k=1": lambda C: C*np.exp(sig2_permodel(C)),
       "perfam k=1":   lambda C: C*np.exp(sig2_perfam(C))}

print("=== EV / bust probability curve (3 seeds x 400 bootstrap of held-out dev preds) ===")
for nm, f in VAR.items():
    print(f"-- {nm}")
    for t in TIERS:
        S = P[f"score_{t}"]; C = f(P[f"cost_{t}"].copy())
        pth = LPath(S, C, dv.cost, dv.score); mult = TIER_MULT[t]
        bs = [pth.batch(Ws[s]) for s in SEEDS]
        out = []
        for sf in (0.70, 0.75, 0.80, 0.85, 0.87, 0.89, 0.90, 0.95, 0.98, 1.00, 1.05):
            ev = np.mean([(lambda r: (r[0]*r[2]).mean())(eval_cached(b, mult, sf)) for b in bs])
            bu = np.mean([1-eval_cached(b, mult, sf)[2].mean() for b in bs])
            pas = np.mean([ (lambda r: r[0][r[2]].mean() if r[2].any() else 0)(eval_cached(b, mult, sf)) for b in bs])
            out.append(f"{sf:.2f}:EV{ev:.4f}/bust{bu*100:4.1f}%")
        print(f"   {t:9s} " + "  ".join(out))

print("\n=== safety exchange rate: d(weighted final) per +0.01 safety, no bust ===")
for t in TIERS:
    S = P[f"score_{t}"]; C = P[f"cost_{t}"]
    pth = LPath(S, C, dv.cost, dv.score); b = pth.batch(np.ones((n,1)))
    lo = eval_cached(b, TIER_MULT[t], 0.80)[0][0]; hi = eval_cached(b, TIER_MULT[t], 1.00)[0][0]
    print(f"  {t:9s} score(0.80)={lo:.4f} score(1.00)={hi:.4f}  "
          f"d(weighted final)/+0.01 safety = {TIER_WEIGHT[t]*(hi-lo)/20:.5f}")

print("\n=== headroom H distribution, deployed vs smeared (all seeds pooled) ===")
for nm, f in VAR.items():
    row=[]
    for t in TIERS:
        S = P[f"score_{t}"]; C = f(P[f"cost_{t}"].copy())
        pth = LPath(S, C, dv.cost, dv.score)
        H = np.concatenate([headroom_cached(pth.batch(Ws[s]), TIER_MULT[t]) for s in SEEDS])
        row.append(f"{t[:4]} mean{H.mean():.3f} sd{H.std():.4f} cv{H.std()/H.mean():.4f} q02 {np.quantile(H,0.02):.3f}")
    print(f"  {nm:14s} " + " | ".join(row))
