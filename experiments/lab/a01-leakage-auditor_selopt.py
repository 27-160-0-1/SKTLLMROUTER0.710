# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""A01 Q4: how much score is manufactured by CHOOSING a constant on the same
880 items you then report?  Split-half selection-optimism measurement on the
honest train-only holdout predictions (reports/lab/dev_preds_e43.npz)."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labdata as L

tr, dv = L.load_all()
z = np.load(ROOT/"reports/lab/dev_preds_e43.npz", allow_pickle=True)
S = {t: z[f"score_{t}"] for t in L.TIERS}
C = {t: z[f"cost_{t}"] for t in L.TIERS}
GRID = np.round(np.arange(0.60, 1.0001, 0.005), 6)

def tier_on(subset, tier, safety, Sm, Cm):
    """exact allocator+scorer restricted to `subset` indices"""
    ps, pc = Sm[subset], Cm[subset]
    mult = L.TIER_MULT[tier]
    lt = pc[:, 0].sum(); cap = lt * max(1.0, mult * safety)
    def choose(pen):
        return (ps - pen * pc / lt).argmax(axis=1)
    sel = choose(0.0); tot = pc[np.arange(len(sel)), sel].sum()
    if tot > cap:
        lo, hi = 0.0, 1.0
        sel = choose(hi); tot = pc[np.arange(len(sel)), sel].sum()
        while tot > cap and hi < 2**60:
            lo, hi = hi, hi*2.0; sel = choose(hi); tot = pc[np.arange(len(sel)), sel].sum()
        for _ in range(40):
            mid = (lo+hi)/2.0; cand = choose(mid); ct = pc[np.arange(len(cand)), cand].sum()
            if ct <= cap: hi, sel, tot = mid, cand, ct
            else: lo = mid
    if tot > cap: sel = np.zeros(len(ps), dtype=int)
    tc = dv.cost[subset]; ts = dv.score[subset]
    ratio = tc[np.arange(len(sel)), sel].sum() / tc[:, 0].sum()
    passed = ratio <= mult + 1e-15
    raw = ts[np.arange(len(sel)), sel].mean()
    return (raw if passed else 0.0), ratio

rng = np.random.default_rng(20260819)
n = len(dv)
NREP = 200
rows = {t: [] for t in L.TIERS}
for rep in range(NREP):
    perm = rng.permutation(n)
    A, B = perm[:n//2], perm[n//2:]
    for t in L.TIERS:
        vA = np.array([tier_on(A, t, s, S[t], C[t])[0] for s in GRID])
        vB = np.array([tier_on(B, t, s, S[t], C[t])[0] for s in GRID])
        sA = GRID[int(vA.argmax())]        # constant chosen on A
        sB = GRID[int(vB.argmax())]        # constant chosen on B (oracle for B)
        rows[t].append((vB[int(vA.argmax())], vB.max(), vB[int(vB.argmax())], sA, sB))
print(f"split-half selection optimism, {NREP} random 440/440 splits of dev")
print("tier      transfer(B|chosen on A)   in-sample(B|chosen on B)   optimism")
opt = {}
for t in L.TIERS:
    a = np.array(rows[t])
    tr_mean, is_mean = a[:, 0].mean(), a[:, 1].mean()
    opt[t] = is_mean - tr_mean
    print(f"{t:9s} {tr_mean:.6f}                  {is_mean:.6f}              {is_mean-tr_mean:+.6f}"
          f"   (chosen safety mean {a[:,3].mean():.3f} sd {a[:,3].std():.3f})")
w = sum(L.TIER_WEIGHT[t]*opt[t] for t in L.TIERS)
print(f"weighted final optimism at n=440 : {w:+.6f}")
print(f"extrapolated to n=880 (~ /sqrt2)  : {w/np.sqrt(2):+.6f}")

print()
print("== how flat is the dev curve? (full 880, per tier best vs deployed) ==")
for t, dep in (("fast",.98),("balanced",.87),("premium",.85)):
    v = np.array([tier_on(np.arange(n), t, s, S[t], C[t])[0] for s in GRID])
    best = v.max(); at = GRID[int(v.argmax())]
    d = tier_on(np.arange(n), t, dep, S[t], C[t])[0]
    npass = int((v > 0).sum())
    print(f"{t:9s} best={best:.6f}@{at:.3f}  deployed({dep})={d:.6f}  spread={best-d:+.6f}  "
          f"grid points that pass budget: {npass}/{len(GRID)}")
