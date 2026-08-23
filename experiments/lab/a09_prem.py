# SPDX-License-Identifier: Apache-2.0
"""a09: premium-tier only - can the conditional-variance model beat the family table?"""
from __future__ import annotations
import sys
from pathlib import Path as FP
import numpy as np
sys.path.insert(0, str(FP(__file__).resolve().parents[0]))
from labdata import load_split, TIER_MULT, TIER_WEIGHT
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
T = "premium"; S0 = P["score_premium"]; C0 = P["cost_premium"]

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

def ev(C):
    pth = LPath(S0, C, dv.cost, dv.score); cur = {}
    Hs = []
    for s in SEEDS:
        b = pth.batch(Ws[s]); Hs.append(headroom_cached(b, 4.0)); e = np.zeros(len(GRID))
        for gi, sf in enumerate(GRID):
            sc, ra, pa = eval_cached(b, 4.0, float(sf)); e[gi] = (sc*pa).mean()
        cur[s] = e
    m = np.mean([cur[s] for s in SEEDS], axis=0); bi = int(m.argmax())
    lo = np.mean([cur[h][int(np.mean([cur[s] for s in SEEDS if s != h], axis=0).argmax())] for h in SEEDS])
    H = np.concatenate(Hs)
    return float(m[bi]), float(GRID[bi]), float(lo), float(H.std()/H.mean())

base = ev(C0)
print(f"  {'base':26s} EV {base[0]:.4f}@{base[1]:.3f} loso {base[2]:.4f} cvH {base[3]:.4f}")
CAND = {
 "gbm k=0.5":        lambda h: C0*np.exp(0.5*sig2_gbm(C0, S0, HS[h])),
 "gbm k=0.75":       lambda h: C0*np.exp(0.75*sig2_gbm(C0, S0, HS[h])),
 "perfam1":          lambda h: C0*np.exp(1.0*sig2_perfam(C0, HS[h])),
 "perfam1 + gbm0.25":lambda h: C0*np.exp(1.0*sig2_perfam(C0, HS[h]) + 0.25*sig2_gbm(C0, S0, HS[h])),
 "perfam0.5 + gbm0.5":lambda h: C0*np.exp(0.5*sig2_perfam(C0, HS[h]) + 0.5*sig2_gbm(C0, S0, HS[h])),
}
for nm, f in CAND.items():
    rs = [ev(f(h)) for h in range(5)]
    a = np.array([r[0] for r in rs]); l = np.array([r[2] for r in rs])
    print(f"  {nm:26s} EV {a.mean():.4f}+-{a.std():.4f} ({np.sum(a>base[0])}/5 > base) "
          f"loso {l.mean():.4f} d={a.mean()-base[0]:+.4f}  safety~{np.mean([r[1] for r in rs]):.3f} "
          f"cvH {np.mean([r[3] for r in rs]):.4f}")
