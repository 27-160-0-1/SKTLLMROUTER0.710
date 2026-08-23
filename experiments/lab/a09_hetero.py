# SPDX-License-Identifier: Apache-2.0
"""a09: heteroscedastic cost re-transformation - kappa sweep, shape-only, per-column,
   and out-of-sample safety selection."""
from __future__ import annotations
import sys, time
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
rng = np.random.default_rng(0); half = rng.integers(0, 2, size=n)

def feats(C, S):
    return np.column_stack([famid, np.log(C), S, np.log(TL), np.log(C[:, 2] / C[:, 0])])

_cache = {}
def sigma2(C, S, tier):
    """cross-fitted conditional variance of the log cost residual, per model."""
    key = (tier, C.tobytes()[:64], C.shape)
    if key in _cache: return _cache[key]
    X = feats(C, S); r = np.log(dv.cost) - np.log(C)
    V = np.zeros_like(C); M = np.zeros(3)
    for h in (0, 1):
        tr = half != h; te = half == h
        for j in range(3):
            m = HistGradientBoostingRegressor(max_leaf_nodes=8, learning_rate=0.08, max_iter=120,
                                              min_samples_leaf=30, l2_regularization=1.0,
                                              categorical_features=[0], random_state=7)
            m.fit(X[tr], (r[tr, j] - r[tr, j].mean()) ** 2)
            V[te, j] = np.clip(m.predict(X[te]), 0.02, 2.0)
    _cache[key] = V
    return V

def summatch(C):
    out = C.copy()
    for h in (0, 1):
        tr = half != h; te = half == h
        for j in range(3):
            out[te, j] = C[te, j] * (dv.cost[tr, j].sum() / C[tr, j].sum())
    return out

def hetero(C, S, tier, kappa, normalize, cols=(0, 1, 2)):
    V = sigma2(C, S, tier)
    k = np.ones_like(C)
    for j in cols:
        k[:, j] = np.exp(kappa * V[:, j])
    if normalize:                       # keep the per-model predicted SUM unchanged (pure shape)
        for j in range(3):
            k[:, j] /= (C[:, j] * k[:, j]).sum() / C[:, j].sum()
    return C * k

SAFE_DEP = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
GRID = np.round(np.arange(0.55, 1.501, 0.005), 4)
SEEDS = (7, 17, 23)
Ws = {s: make_W(n, 400, s) for s in SEEDS}

def evaluate(name, f, show=True):
    """Returns per-tier dict with EV curves per seed."""
    res = {}
    for t in TIERS:
        S = P[f"score_{t}"]; C = f(P[f"cost_{t}"].copy(), S, t)
        pth = LPath(S, C, dv.cost, dv.score)
        mult = TIER_MULT[t]
        curves = {}
        H = {}
        for s in SEEDS:
            b = pth.batch(Ws[s])
            H[s] = headroom_cached(b, mult)
            ev = np.zeros(len(GRID)); bu = np.zeros(len(GRID))
            for gi, sf in enumerate(GRID):
                sc, ra, pa = eval_cached(b, mult, float(sf))
                ev[gi] = (sc * pa).mean(); bu[gi] = 1 - pa.mean()
            curves[s] = (ev, bu)
        res[t] = dict(curves=curves, H=H)
    # aggregate: argmax of mean EV (in-sample safety choice)
    tot = 0.0; rows = []
    for t in TIERS:
        ev = np.mean([res[t]["curves"][s][0] for s in SEEDS], axis=0)
        bu = np.mean([res[t]["curves"][s][1] for s in SEEDS], axis=0)
        bi = int(ev.argmax()); tot += TIER_WEIGHT[t] * ev[bi]
        Hc = np.concatenate([res[t]["H"][s] for s in SEEDS])
        rows.append(f"{t[:4]} {ev[bi]:.4f}@{GRID[bi]:.3f} bust{bu[bi]*100:.1f}% cvH={Hc.std()/Hc.mean():.4f}")
    # honest: choose safety on 2 seeds, score on the third
    hon = []
    for held in SEEDS:
        tr = [s for s in SEEDS if s != held]; tt = 0.0
        for t in TIERS:
            ev_tr = np.mean([res[t]["curves"][s][0] for s in tr], axis=0)
            bi = int(ev_tr.argmax())
            tt += TIER_WEIGHT[t] * res[t]["curves"][held][0][bi]
        hon.append(tt)
    if show:
        print(f"{name:26s} EVbest={tot:.4f}  EV(loso-safety)={np.mean(hon):.4f}  " + " | ".join(rows))
    return tot, float(np.mean(hon)), res

t0 = time.time()
print("=== kappa sweep, raw (level moves, safety re-tuned) ===")
for kap in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
    evaluate(f"hetero raw k={kap}", lambda C, S, t, k=kap: hetero(C, S, t, k, False))
print()
print("=== kappa sweep, shape-only (per-model sum preserved) ===")
for kap in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    evaluate(f"hetero shape k={kap}", lambda C, S, t, k=kap: hetero(C, S, t, k, True))
print()
print("=== shape-only, k1 column only ===")
for kap in (0.5, 1.0, 1.5, 2.0):
    evaluate(f"shape k1-only k={kap}", lambda C, S, t, k=kap: hetero(C, S, t, k, True, cols=(2,)))
print()
print("=== summatch composed with shape-only hetero ===")
for kap in (0.0, 0.5, 1.0, 1.5):
    evaluate(f"summatch+shape k={kap}", lambda C, S, t, k=kap: summatch(hetero(C, S, t, k, True)))
print(f"\n[{time.time()-t0:.0f}s]")
