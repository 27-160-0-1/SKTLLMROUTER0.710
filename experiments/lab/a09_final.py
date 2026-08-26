# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a09: (1) cost-side exchange rate, (2) split-robust comparison of the smearing family,
   (3) per-tier kappa recommendation."""
from __future__ import annotations
import sys, time
from pathlib import Path as FP
import numpy as np
sys.path.insert(0, str(FP(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, MODEL_IDS
from a09_harness import LPath, eval_cached, headroom_cached, make_W
ROOT = FP(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.similarity import classify_family
from sklearn.ensemble import HistGradientBoostingRegressor

dv = load_split("dev"); n = len(dv)
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
fam = np.array([classify_family(t) for t in dv.texts]); fams = sorted(set(fam))
famid = np.array([fams.index(f) for f in fam]); TL = np.array([len(t) for t in dv.texts], float)
GRID = np.round(np.arange(0.55, 1.601, 0.005), 4)
SEEDS = (7, 17, 23); Ws = {s: make_W(n, 400, s) for s in SEEDS}
HS = {h: np.random.default_rng(h).integers(0, 2, size=n) for h in range(5)}

def tier_eval(t, S, C):
    pth = LPath(S, C, dv.cost, dv.score); mult = TIER_MULT[t]; cur = {}; Hs = []
    for s in SEEDS:
        b = pth.batch(Ws[s]); Hs.append(headroom_cached(b, mult)); ev = np.zeros(len(GRID))
        for gi, sf in enumerate(GRID):
            sc, ra, pa = eval_cached(b, mult, float(sf)); ev[gi] = (sc*pa).mean()
        cur[s] = ev
    ev = np.mean([cur[s] for s in SEEDS], axis=0); bi = int(ev.argmax())
    lo = np.mean([cur[h][int(np.mean([cur[s] for s in SEEDS if s != h], axis=0).argmax())] for h in SEEDS])
    H = np.concatenate(Hs)
    return dict(best=float(ev[bi]), sf=float(GRID[bi]), loso=float(lo), cvH=float(H.std()/H.mean()), mH=float(H.mean()))

def run(mk_c, mk_s=None):
    r = {}
    for t in TIERS:
        S = P[f"score_{t}"] if mk_s is None else mk_s(P[f"score_{t}"])
        r[t] = tier_eval(t, S, mk_c(P[f"cost_{t}"].copy(), S, t))
    r["best"] = sum(TIER_WEIGHT[t]*r[t]["best"] for t in TIERS)
    r["loso"] = sum(TIER_WEIGHT[t]*r[t]["loso"] for t in TIERS)
    return r

print("=== 1. COST-SIDE EXCHANGE RATE  log-cost blended toward truth ===")
print(f"{'lam':>5s} {'logRMSE k1':>10s} {'EVbest':>8s} {'loso':>8s}  per-tier (EV@safety, cvH)")
for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
    def mk(C, S, t, l=lam): return np.exp((1-l)*np.log(C) + l*np.log(dv.cost))
    r = run(mk)
    rm = np.sqrt(((np.log(mk(P['cost_premium'],None,'premium')[:,2])-np.log(dv.cost[:,2]))**2).mean())
    print(f"{lam:5.2f} {rm:10.3f} {r['best']:8.4f} {r['loso']:8.4f}  " +
          " | ".join(f"{t[:4]} {r[t]['best']:.4f}@{r[t]['sf']:.3f} cv{r[t]['cvH']:.3f}" for t in TIERS))

def sig2_perfam(C, half, shrink=20.0):
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

def sig2_permodel(C, half):
    r = np.log(dv.cost)-np.log(C); V = np.zeros_like(C)
    for h in (0, 1):
        a = half != h; b = half == h
        for j in range(3): V[b, j] = r[a, j].var()
    return V

def sig2_gbm(C, S, half):
    X = np.column_stack([famid, np.log(C), S, np.log(TL), np.log(C[:,2]/C[:,0])])
    r = np.log(dv.cost)-np.log(C); V = np.zeros_like(C)
    for h in (0, 1):
        a = half != h; b = half == h
        for j in range(3):
            m = HistGradientBoostingRegressor(max_leaf_nodes=8, learning_rate=0.08, max_iter=120,
                    min_samples_leaf=30, l2_regularization=1.0, categorical_features=[0], random_state=7)
            m.fit(X[a], (r[a, j]-r[a, j].mean())**2); V[b, j] = np.clip(m.predict(X[b]), 0.02, 2.0)
    return V

print("\n=== 2. split-robust comparison (5 cross-fit splits, delta vs base) ===")
CAND = {
 "permodel  k=0.5":  lambda h: (lambda C,S,t: C*np.exp(0.5*sig2_permodel(C, HS[h]))),
 "permodel  k=1.0":  lambda h: (lambda C,S,t: C*np.exp(1.0*sig2_permodel(C, HS[h]))),
 "permodel  k=1.5":  lambda h: (lambda C,S,t: C*np.exp(1.5*sig2_permodel(C, HS[h]))),
 "perfam    k=0.5":  lambda h: (lambda C,S,t: C*np.exp(0.5*sig2_perfam(C, HS[h]))),
 "perfam    k=1.0":  lambda h: (lambda C,S,t: C*np.exp(1.0*sig2_perfam(C, HS[h]))),
 "perfam    k=1.5":  lambda h: (lambda C,S,t: C*np.exp(1.5*sig2_perfam(C, HS[h]))),
}
base = run(lambda C,S,t: C)
print(f"  {'base':20s} EVbest {base['best']:.4f} loso {base['loso']:.4f}   " +
      " | ".join(f"{t[:4]} {base[t]['best']:.4f}@{base[t]['sf']:.3f} cv{base[t]['cvH']:.3f}" for t in TIERS))
store = {}
for nm, mkf in CAND.items():
    rs = [run(mkf(h)) for h in range(5)]
    store[nm] = rs
    d = np.array([r["best"] for r in rs]) - base["best"]
    dl = np.array([r["loso"] for r in rs]) - base["loso"]
    per = {t: np.array([r[t]["best"] for r in rs]) - base[t]["best"] for t in TIERS}
    print(f"  {nm:20s} dEVbest {d.mean():+.4f}+-{d.std():.4f} ({np.sum(d>0)}/5 pos)  dloso {dl.mean():+.4f}  " +
          " ".join(f"{t[:4]}{per[t].mean():+.4f}" for t in TIERS))

print("\n=== 3. per-tier recommendation (mean over 5 splits) ===")
for t in TIERS:
    rows = sorted(((nm, np.mean([r[t]["best"] for r in rs]), np.mean([r[t]["sf"] for r in rs]),
                    np.mean([r[t]["cvH"] for r in rs]), np.std([r[t]["best"] for r in rs]))
                   for nm, rs in store.items()), key=lambda x: -x[1])
    print(f"  {t:9s} base {base[t]['best']:.4f}@{base[t]['sf']:.3f} cv{base[t]['cvH']:.3f}")
    for nm, m, sf, cv, sd in rows[:3]:
        print(f"      {nm:20s} {m:.4f}+-{sd:.4f} @safety~{sf:.3f} cvH {cv:.3f}  (delta {m-base[t]['best']:+.4f})")
