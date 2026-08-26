# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a09: counterfactual - what if the INPUT-token part of the cost were exact?

BRIEF: input tokens are predictable to dev R2=0.9985 from cheap text stats, and are ~50% of
light/mid cost but only 12.5% of k1 cost.  The deployed head predicts log(total cost) in one shot.
Model: keep the CURRENT multiplicative error but apply it only to the output-token term.
"""
from __future__ import annotations
import sys
from pathlib import Path as FP
import numpy as np
sys.path.insert(0, str(FP(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, MODEL_IDS, RATES, TOKEN_UNIT
from a09_harness import LPath, eval_cached, headroom_cached, make_W
ROOT = FP(__file__).resolve().parents[2]
dv = load_split("dev"); n = len(dv)
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
GRID = np.round(np.arange(0.55, 1.601, 0.005), 4)
SEEDS = (7, 17, 23); Ws = {s: make_W(n, 400, s) for s in SEEDS}
RIN = np.array([RATES[m][1] for m in MODEL_IDS]); ROUT = np.array([RATES[m][2] for m in MODEL_IDS])
IN = dv.itok * RIN[None, :] / TOKEN_UNIT
OUT = dv.otok * ROUT[None, :] / TOKEN_UNIT
print("input share of true cost per model:", np.round(IN.sum(0) / dv.cost.sum(0), 3))

def hybrid(C, w_in=1.0, w_out=0.0):
    """w_in / w_out = fraction of the *log* error removed from that term."""
    e = C / dv.cost                       # current multiplicative error, <1 on average
    li = np.exp((1 - w_in) * np.log(e)); lo = np.exp((1 - w_out) * np.log(e))
    return IN * li + OUT * lo

def ev(mk):
    tot = 0.0; rows = []
    for t in TIERS:
        S = P[f"score_{t}"]; C = mk(P[f"cost_{t}"].copy())
        pth = LPath(S, C, dv.cost, dv.score); cur = {}; Hs = []
        for s in SEEDS:
            b = pth.batch(Ws[s]); Hs.append(headroom_cached(b, TIER_MULT[t])); e = np.zeros(len(GRID))
            for gi, sf in enumerate(GRID):
                sc, ra, pa = eval_cached(b, TIER_MULT[t], float(sf)); e[gi] = (sc * pa).mean()
            cur[s] = e
        m = np.mean([cur[s] for s in SEEDS], axis=0); bi = int(m.argmax())
        H = np.concatenate(Hs); tot += TIER_WEIGHT[t] * m[bi]
        rows.append(f"{t[:4]} {m[bi]:.4f}@{GRID[bi]:.3f} cvH{H.std()/H.mean():.4f}")
    return tot, rows

for lab, mk in [
    ("base (single log-cost head)", lambda C: C),
    ("input term exact",            lambda C: hybrid(C, 1.0, 0.0)),
    ("input term 50% better",       lambda C: hybrid(C, 0.5, 0.0)),
    ("input exact + output 10% bt", lambda C: hybrid(C, 1.0, 0.1)),
    ("input exact + output 25% bt", lambda C: hybrid(C, 1.0, 0.25)),
    ("both exact (= true cost)",    lambda C: hybrid(C, 1.0, 1.0)),
]:
    tot, rows = ev(mk)
    print(f"  {lab:30s} EV={tot:.4f}  " + " | ".join(rows))

print("\nlog-RMSE per model under each hypothesis:")
for lab, mk in [("base", lambda C: C), ("input exact", lambda C: hybrid(C, 1.0, 0.0))]:
    C = mk(P["cost_premium"].copy())
    print(f"  {lab:12s} " + " ".join(f"{m}={np.sqrt(((np.log(C[:,j])-np.log(dv.cost[:,j]))**2).mean()):.3f}"
                                     for j, m in enumerate(MODEL_IDS)))

# ---- does the token split compose with the variance-form smearing?
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.similarity import classify_family
fam = np.array([classify_family(t) for t in dv.texts]); fams = sorted(set(fam))
famid = np.array([fams.index(f) for f in fam])
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
print("\n=== composition with variance-form smearing (mean over 5 cross-fit splits) ===")
for lab, base_mk in [("base", lambda C: C), ("input exact", lambda C: hybrid(C, 1.0, 0.0))]:
    for kap in (0.0, 0.5, 1.0):
        accs = []
        for h in range(5 if kap > 0 else 1):
            hh = np.random.default_rng(h).integers(0, 2, size=n)
            mk = (lambda C, b=base_mk, k=kap, q=hh: (lambda X: X*np.exp(k*sig2_perfam(X, q)))(b(C)))
            accs.append(ev(mk)[0])
        a = np.array(accs)
        print(f"  {lab:12s} kappa={kap:<4} EV={a.mean():.4f}" + (f" +-{a.std():.4f}" if len(a) > 1 else ""))
