# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a09: cost re-transformation variants, evaluated through bootstrap EV + headroom."""
from __future__ import annotations
import sys, time
from pathlib import Path as FP
import numpy as np
sys.path.insert(0, str(FP(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, MODEL_IDS
from a09_harness import LPath, eval_tier, eval_cached, headroom_cached, make_W
ROOT = FP(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.similarity import classify_family
from sklearn.ensemble import HistGradientBoostingRegressor

dv = load_split("dev"); n = len(dv)
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
fam = np.array([classify_family(t) for t in dv.texts]); fams = sorted(set(fam))
famid = np.array([fams.index(f) for f in fam])
TL = np.array([len(t) for t in dv.texts], float)

rng = np.random.default_rng(0)
half = rng.integers(0, 2, size=n)            # cross-fitting split

def feats(C, S):
    return np.column_stack([famid, np.log(C), S, np.log(TL), np.log(C[:,2]/C[:,0])])

def xfit_const(C, weighted):
    """per-model smearing constant, cross-fitted on the other half."""
    out = C.copy()
    r = np.log(dv.cost) - np.log(C)
    for h in (0, 1):
        tr = half != h; te = half == h
        for j in range(3):
            k = (dv.cost[tr, j].sum() / C[tr, j].sum()) if weighted else np.exp(r[tr, j]).mean()
            out[te, j] = C[te, j] * k
    return out

def xfit_family(C, weighted=True, shrink=0.0):
    out = C.copy()
    for h in (0, 1):
        tr = half != h; te = half == h
        for j in range(3):
            kg = dv.cost[tr, j].sum() / C[tr, j].sum()
            for f in range(len(fams)):
                a = tr & (famid == f); b = te & (famid == f)
                if a.sum() < 5 or b.sum() == 0:
                    out[b, j] = C[b, j] * kg; continue
                kf = dv.cost[a, j].sum() / C[a, j].sum()
                w = a.sum() / (a.sum() + shrink)
                out[b, j] = C[b, j] * (w * kf + (1 - w) * kg)
    return out

def xfit_cond(C, S, leaves=8, lr=0.08, iters=120):
    """E[exp(r)|x] by GBM -> conditional-mean re-transformation."""
    out = C.copy()
    X = feats(C, S)
    r = np.log(dv.cost) - np.log(C)
    for h in (0, 1):
        tr = half != h; te = half == h
        for j in range(3):
            m = HistGradientBoostingRegressor(max_leaf_nodes=leaves, learning_rate=lr,
                                              max_iter=iters, min_samples_leaf=30, l2_regularization=1.0,
                                              categorical_features=[0], random_state=7)
            m.fit(X[tr], np.exp(r[tr, j]))
            k = np.clip(m.predict(X[te]), 0.5, 6.0)
            out[te, j] = C[te, j] * k
    return out

def xfit_sigma(C, S, kappa=0.5):
    """lognormal hetero: pred * exp(kappa*sigma_hat^2)."""
    out = C.copy(); X = feats(C, S)
    r = np.log(dv.cost) - np.log(C)
    for h in (0, 1):
        tr = half != h; te = half == h
        for j in range(3):
            m = HistGradientBoostingRegressor(max_leaf_nodes=8, learning_rate=0.08, max_iter=120,
                                              min_samples_leaf=30, l2_regularization=1.0,
                                              categorical_features=[0], random_state=7)
            m.fit(X[tr], (r[tr, j] - r[tr, j].mean()) ** 2)
            v = np.clip(m.predict(X[te]), 0.02, 2.0)
            out[te, j] = C[te, j] * np.exp(kappa * v + r[tr, j].mean())
    return out

VARIANTS = {
    "C0 base":              lambda C, S: C,
    "C1 duan/model":        lambda C, S: xfit_const(C, weighted=False),
    "C2 summatch/model":    lambda C, S: xfit_const(C, weighted=True),
    "C3 summatch/fam*mdl":  lambda C, S: xfit_family(C, shrink=0.0),
    "C3s fam*mdl shrink30": lambda C, S: xfit_family(C, shrink=30.0),
    "C4 cond E[exp r|x]":   lambda C, S: xfit_cond(C, S),
    "C5 lognorm k=.5":      lambda C, S: xfit_sigma(C, S, 0.5),
    "C6 TRUE cost":         lambda C, S: dv.cost.copy(),
}

SAFE_DEP = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
GRID = np.round(np.arange(0.60, 1.601, 0.005), 4)
SEEDS = (7, 17, 23)
B = 400
Ws = [make_W(n, B, s) for s in SEEDS]

def run(name, f, verbose=True):
    tot_dep = 0.0; tot_best = 0.0; lines = []; picks = {}
    for t in TIERS:
        S = P[f"score_{t}"]; C = f(P[f"cost_{t}"].copy(), S)
        pth = LPath(S, C, dv.cost, dv.score)
        mult = TIER_MULT[t]
        bs = [pth.batch(W) for W in Ws]
        hs = np.concatenate([headroom_cached(b, mult) for b in bs])
        evs = np.zeros(len(GRID)); busts = np.zeros(len(GRID))
        for gi, sf in enumerate(GRID):
            e = []; bb = []
            for b in bs:
                sc, ra, pa = eval_cached(b, mult, float(sf))
                e.append((sc * pa).mean()); bb.append(1 - pa.mean())
            evs[gi] = np.mean(e); busts[gi] = np.mean(bb)
        bi = int(evs.argmax())
        e = [ (lambda r: (r[0]*r[2]).mean())(eval_cached(b, mult, SAFE_DEP[t])) for b in bs ]
        ev_dep = float(np.mean(e))
        tot_dep += TIER_WEIGHT[t] * ev_dep
        tot_best += TIER_WEIGHT[t] * evs[bi]
        picks[t] = float(GRID[bi])
        lines.append(f"{t[:4]} dep {ev_dep:.4f} | best {evs[bi]:.4f}@{GRID[bi]:.3f} bust{busts[bi]*100:.1f}% "
                     f"| H mean {hs.mean():.3f} sd {hs.std():.4f} q05 {np.quantile(hs,0.05):.3f} q01 {np.quantile(hs,0.01):.3f}")
    if verbose:
        print(f"{name:22s} EVdep={tot_dep:.4f}  EVbest={tot_best:.4f}")
        for l in lines: print("      " + l)
    return tot_dep, tot_best, picks

t0 = time.time()
print(f"bootstrap {len(SEEDS)}x{B} batches of {n}\n")
for k, v in VARIANTS.items():
    run(k, v)
print(f"\n[{time.time()-t0:.0f}s]")
