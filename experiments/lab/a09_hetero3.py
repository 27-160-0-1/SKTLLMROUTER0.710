# SPDX-License-Identifier: Apache-2.0
"""a09: family-variance x conditional-variance grid, per-tier kappa, robustness."""
from __future__ import annotations
import sys, time, itertools
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
GRID = np.round(np.arange(0.55, 1.501, 0.005), 4)
SEEDS = (7, 17, 23); Ws = {s: make_W(n, 400, s) for s in SEEDS}
HALVES = {}
def halves(seed):
    if seed not in HALVES:
        HALVES[seed] = np.random.default_rng(seed).integers(0, 2, size=n)
    return HALVES[seed]

def sig2_family(C, shrink=20.0, robust=False, hseed=0):
    half = halves(hseed)
    r = np.log(dv.cost) - np.log(C); V = np.zeros_like(C)
    def scale(x):
        if robust:
            q = np.subtract(*np.quantile(x, [0.75, 0.25])); return (q / 1.349) ** 2
        return x.var()
    for h in (0, 1):
        tr = half != h; te = half == h
        for j in range(3):
            g = scale(r[tr, j])
            for f in range(len(fams)):
                a = tr & (famid == f); b = te & (famid == f)
                if b.sum() == 0: continue
                if a.sum() < 8: V[b, j] = g; continue
                w = a.sum() / (a.sum() + shrink)
                V[b, j] = w * scale(r[a, j]) + (1 - w) * g
    return V

_G = {}
def sig2_gbm(C, S, hseed=0):
    key = (C.tobytes()[:48], hseed)
    if key in _G: return _G[key]
    half = halves(hseed)
    X = np.column_stack([famid, np.log(C), S, np.log(TL), np.log(C[:, 2] / C[:, 0])])
    r = np.log(dv.cost) - np.log(C); V = np.zeros_like(C)
    for h in (0, 1):
        tr = half != h; te = half == h
        for j in range(3):
            m = HistGradientBoostingRegressor(max_leaf_nodes=8, learning_rate=0.08, max_iter=120,
                                              min_samples_leaf=30, l2_regularization=1.0,
                                              categorical_features=[0], random_state=7)
            m.fit(X[tr], (r[tr, j] - r[tr, j].mean()) ** 2)
            V[te, j] = np.clip(m.predict(X[te]), 0.02, 2.0)
    _G[key] = V; return V

def tier_curves(t, mk):
    S = P[f"score_{t}"]; C = mk(P[f"cost_{t}"].copy(), S)
    pth = LPath(S, C, dv.cost, dv.score); mult = TIER_MULT[t]
    out = {}
    for s in SEEDS:
        b = pth.batch(Ws[s]); H = headroom_cached(b, mult)
        ev = np.zeros(len(GRID))
        for gi, sf in enumerate(GRID):
            sc, ra, pa = eval_cached(b, mult, float(sf)); ev[gi] = (sc * pa).mean()
        out[s] = (ev, H)
    return out

def tier_stats(cur):
    ev = np.mean([cur[s][0] for s in SEEDS], axis=0); bi = int(ev.argmax())
    loso = np.mean([cur[h][0][int(np.mean([cur[s][0] for s in SEEDS if s != h], axis=0).argmax())] for h in SEEDS])
    Hc = np.concatenate([cur[s][1] for s in SEEDS])
    return dict(best=ev[bi], sf=GRID[bi], loso=float(loso), cvH=Hc.std() / Hc.mean())

t0 = time.time()
print("=== A. family-variance kappa (extended), shrink=20 ===")
tab = {}
for kf in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    row = {t: tier_stats(tier_curves(t, lambda C, S, k=kf: C * np.exp(k * sig2_family(C)))) for t in TIERS}
    tab[("fam", kf)] = row
    print(f" kf={kf:<5} EVbest={sum(TIER_WEIGHT[t]*row[t]['best'] for t in TIERS):.4f} "
          f"loso={sum(TIER_WEIGHT[t]*row[t]['loso'] for t in TIERS):.4f}  " +
          " | ".join(f"{t[:4]} {row[t]['best']:.4f}@{row[t]['sf']:.3f} cv{row[t]['cvH']:.3f}" for t in TIERS))

print("\n=== B. family-variance x conditional (GBM) variance grid ===")
best = None
for kf in (0.0, 0.5, 1.0, 1.5):
    for kg in (0.0, 0.25, 0.5):
        if kf == 0 and kg == 0: continue
        mk = lambda C, S, a=kf, b=kg: C * np.exp(a * sig2_family(C) + b * sig2_gbm(C, S))
        row = {t: tier_stats(tier_curves(t, mk)) for t in TIERS}
        tab[("mix", kf, kg)] = row
        e = sum(TIER_WEIGHT[t]*row[t]['best'] for t in TIERS); l = sum(TIER_WEIGHT[t]*row[t]['loso'] for t in TIERS)
        print(f" kf={kf:<4} kg={kg:<5} EVbest={e:.4f} loso={l:.4f}  " +
              " | ".join(f"{t[:4]} {row[t]['best']:.4f}@{row[t]['sf']:.3f} cv{row[t]['cvH']:.3f}" for t in TIERS))

print("\n=== C. per-tier best kappa over everything measured ===")
base = {t: tab[("fam", 0.0)][t] for t in TIERS}
tot_b = 0.0; tot_l = 0.0
for t in TIERS:
    cands = sorted(tab.items(), key=lambda kv: -kv[1][t]["loso"])
    k, row = cands[0]
    print(f"  {t:9s} base {base[t]['best']:.4f}/loso {base[t]['loso']:.4f}  ->  {k} "
          f"{row[t]['best']:.4f}/loso {row[t]['loso']:.4f} @safety {row[t]['sf']:.3f}")
    tot_b += TIER_WEIGHT[t]*row[t]["best"]; tot_l += TIER_WEIGHT[t]*row[t]["loso"]
print(f"  per-tier-best   EVbest={tot_b:.4f} loso={tot_l:.4f}   (base {sum(TIER_WEIGHT[t]*base[t]['best'] for t in TIERS):.4f}"
      f"/{sum(TIER_WEIGHT[t]*base[t]['loso'] for t in TIERS):.4f})")

print("\n=== D. robustness of famtable kf=1.0: shrink, robust scale, cross-fit split seed ===")
for lab, mk in [
    ("shrink 5",     lambda C, S: C*np.exp(1.0*sig2_family(C, shrink=5.0))),
    ("shrink 20",    lambda C, S: C*np.exp(1.0*sig2_family(C, shrink=20.0))),
    ("shrink 60",    lambda C, S: C*np.exp(1.0*sig2_family(C, shrink=60.0))),
    ("IQR scale",    lambda C, S: C*np.exp(1.0*sig2_family(C, robust=True))),
    ("split seed 1", lambda C, S: C*np.exp(1.0*sig2_family(C, hseed=1))),
    ("split seed 2", lambda C, S: C*np.exp(1.0*sig2_family(C, hseed=2))),
]:
    row = {t: tier_stats(tier_curves(t, mk)) for t in TIERS}
    print(f"  {lab:14s} EVbest={sum(TIER_WEIGHT[t]*row[t]['best'] for t in TIERS):.4f} "
          f"loso={sum(TIER_WEIGHT[t]*row[t]['loso'] for t in TIERS):.4f}  " +
          " | ".join(f"{t[:4]} {row[t]['best']:.4f}@{row[t]['sf']:.3f}" for t in TIERS))
print(f"\n[{time.time()-t0:.0f}s]")
