# SPDX-License-Identifier: Apache-2.0
"""a09: simpler (stdlib-friendly) heteroscedastic smearing + estimator-noise check."""
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
rng = np.random.default_rng(0); half = rng.integers(0, 2, size=n)
GRID = np.round(np.arange(0.55, 1.501, 0.005), 4)
SEEDS = (7, 17, 23); Ws = {s: make_W(n, 400, s) for s in SEEDS}

def sig2_family(C, xfit=True, shrink=20.0):
    """family x model variance table of the log-cost residual."""
    r = np.log(dv.cost) - np.log(C); V = np.zeros_like(C)
    parts = [(half != h, half == h) for h in (0, 1)] if xfit else [(np.ones(n, bool), np.ones(n, bool))]
    for tr, te in parts:
        for j in range(3):
            g = r[tr, j].var()
            for f in range(len(fams)):
                a = tr & (famid == f); b = te & (famid == f)
                if b.sum() == 0: continue
                if a.sum() < 5: V[b, j] = g; continue
                w = a.sum() / (a.sum() + shrink)
                V[b, j] = w * r[a, j].var() + (1 - w) * g
    return V

def sig2_gbm(C, S, xfit=True):
    X = np.column_stack([famid, np.log(C), S, np.log(TL), np.log(C[:, 2] / C[:, 0])])
    r = np.log(dv.cost) - np.log(C); V = np.zeros_like(C)
    parts = [(half != h, half == h) for h in (0, 1)] if xfit else [(np.ones(n, bool), np.ones(n, bool))]
    for tr, te in parts:
        for j in range(3):
            m = HistGradientBoostingRegressor(max_leaf_nodes=8, learning_rate=0.08, max_iter=120,
                                              min_samples_leaf=30, l2_regularization=1.0,
                                              categorical_features=[0], random_state=7)
            m.fit(X[tr], (r[tr, j] - r[tr, j].mean()) ** 2)
            V[te, j] = np.clip(m.predict(X[te]), 0.02, 2.0)
    return V

def apply_k(C, V, kappa):
    return C * np.exp(kappa * V)

def curves(S, C, t):
    pth = LPath(S, C, dv.cost, dv.score); mult = TIER_MULT[t]
    out = {}
    for s in SEEDS:
        b = pth.batch(Ws[s]); H = headroom_cached(b, mult)
        ev = np.zeros(len(GRID))
        for gi, sf in enumerate(GRID):
            sc, ra, pa = eval_cached(b, mult, float(sf)); ev[gi] = (sc * pa).mean()
        out[s] = (ev, H)
    return out

def summarize(name, mk):
    per = {}
    for t in TIERS:
        S = P[f"score_{t}"]; C = mk(P[f"cost_{t}"].copy(), S, t)
        per[t] = curves(S, C, t)
    tot = 0.0; loso = []; rows = []
    for t in TIERS:
        ev = np.mean([per[t][s][0] for s in SEEDS], axis=0); bi = int(ev.argmax())
        tot += TIER_WEIGHT[t] * ev[bi]
        Hc = np.concatenate([per[t][s][1] for s in SEEDS])
        rows.append(f"{t[:4]} {ev[bi]:.4f}@{GRID[bi]:.3f} cvH={Hc.std()/Hc.mean():.4f}")
    for held in SEEDS:
        tr = [s for s in SEEDS if s != held]; tt = 0.0
        for t in TIERS:
            bi = int(np.mean([per[t][s][0] for s in tr], axis=0).argmax())
            tt += TIER_WEIGHT[t] * per[t][held][0][bi]
        loso.append(tt)
    print(f"{name:28s} EVbest={tot:.4f} loso={np.mean(loso):.4f}  " + " | ".join(rows))
    return per

print("=== A. family-table sigma^2 (9x3 constants, stdlib-friendly) ===")
for kap in (0.0, 0.25, 0.5, 0.75, 1.0):
    summarize(f"famtable k={kap}", lambda C, S, t, k=kap: apply_k(C, sig2_family(C), k))
print()
print("=== B. GBM sigma^2, cross-fit vs ORACLE (full-dev fit) ===")
perX = summarize("gbm xfit k=0.5", lambda C, S, t: apply_k(C, sig2_gbm(C, S, True), 0.5))
perO = summarize("gbm ORACLE k=0.5", lambda C, S, t: apply_k(C, sig2_gbm(C, S, False), 0.5))
perT = summarize("TRUE cost", lambda C, S, t: dv.cost.copy())
print()
print("=== C. multiplier diagnostics (premium tier, gbm xfit k=0.5) ===")
C = P["cost_premium"]; V = sig2_gbm(C, P["score_premium"], True)
K = np.exp(0.5 * V)
for j, m in enumerate(MODEL_IDS):
    print(f"  {m:11s} sigma^2 mean={V[:,j].mean():.3f}  mult k: "
          f"p05={np.quantile(K[:,j],.05):.3f} med={np.median(K[:,j]):.3f} p95={np.quantile(K[:,j],.95):.3f} max={K[:,j].max():.3f}")
print("  per-family mean multiplier (k1):")
for f in range(len(fams)):
    msk = famid == f
    print(f"    {fams[f]:16s} n={msk.sum():3d} k1 mult={K[msk,2].mean():.3f}  true/pred ratio={dv.cost[msk,2].sum()/C[msk,2].sum():.3f}")
print()
print("=== D. split-half stability of the family sigma^2 table (k1 column) ===")
r = np.log(dv.cost) - np.log(C)
for f in range(len(fams)):
    a = (half == 0) & (famid == f); b = (half == 1) & (famid == f)
    if a.sum() < 5 or b.sum() < 5: continue
    print(f"    {fams[f]:16s} nA={a.sum():3d} nB={b.sum():3d}  varA={r[a,2].var():.3f} varB={r[b,2].var():.3f}")
