# SPDX-License-Identifier: Apache-2.0
"""a09: final package - rule-based safety (bust<=1%), per-tier smearing, 5 splits."""
from __future__ import annotations
import sys
from pathlib import Path as FP
import numpy as np
sys.path.insert(0, str(FP(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT
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

def curve(t, C):
    S = P[f"score_{t}"]; pth = LPath(S, C, dv.cost, dv.score); mult = TIER_MULT[t]
    ev = np.zeros(len(GRID)); bu = np.zeros(len(GRID)); pas = np.zeros(len(GRID))
    bs = [pth.batch(Ws[s]) for s in SEEDS]
    for gi, sf in enumerate(GRID):
        e=[];b_=[];p_=[]
        for b in bs:
            sc, ra, pa = eval_cached(b, mult, float(sf))
            e.append((sc*pa).mean()); b_.append(1-pa.mean()); p_.append(sc[pa].mean() if pa.any() else 0.0)
        ev[gi]=np.mean(e); bu[gi]=np.mean(b_); pas[gi]=np.mean(p_)
    return ev, bu, pas

def rule(ev, bu, cap=0.01):
    ok = np.where(bu <= cap)[0]
    i = ok[int(np.argmax(ev[ok]))]
    return i

PKG = {
 "A base (no smearing)": {t: (lambda C, S, h: C) for t in TIERS},
 "B perfam k=1 all tiers": {t: (lambda C, S, h: C*np.exp(sig2_perfam(C, HS[h]))) for t in TIERS},
 "C per-tier recommended": {
    "fast":     lambda C, S, h: C,
    "balanced": lambda C, S, h: C*np.exp(1.0*sig2_perfam(C, HS[h])),
    "premium":  lambda C, S, h: C*np.exp(0.5*sig2_perfam(C, HS[h]) + 0.5*sig2_gbm(C, S, HS[h])),
 },
}
print("rule: pick the safety maximising bootstrap EV subject to bust<=1%  (3 seeds x 400)")
for nm, spec in PKG.items():
    accs = []
    for h in range(5):
        tot = 0.0; parts = []
        for t in TIERS:
            C = spec[t](P[f"cost_{t}"].copy(), P[f"score_{t}"], h)
            ev, bu, pas = curve(t, C); i = rule(ev, bu)
            tot += TIER_WEIGHT[t]*ev[i]
            parts.append(f"{t[:4]} {ev[i]:.4f}@{GRID[i]:.3f}(pass {pas[i]:.4f},bust{bu[i]*100:.1f}%)")
        accs.append(tot)
        if h == 0: print(f"  {nm:24s} split0 EV={tot:.4f}  " + " | ".join(parts))
    a = np.array(accs)
    print(f"  {nm:24s} 5-split EV = {a.mean():.4f} +- {a.std():.4f}   raw: {np.round(a,4)}")
