# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""A01 Q4b: decompose the split-half safety-selection optimism into
(bust rate) x (score when passing), and price the back-off."""
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

def run(subset, tier, safety, Sm, Cm):
    ps, pc = Sm[subset], Cm[subset]
    mult = L.TIER_MULT[tier]
    lt = pc[:, 0].sum(); cap = lt * max(1.0, mult * safety)
    ch = lambda pen: (ps - pen * pc / lt).argmax(axis=1)
    sel = ch(0.0); tot = pc[np.arange(len(sel)), sel].sum()
    if tot > cap:
        lo, hi = 0.0, 1.0
        sel = ch(hi); tot = pc[np.arange(len(sel)), sel].sum()
        while tot > cap and hi < 2**60:
            lo, hi = hi, hi*2.0; sel = ch(hi); tot = pc[np.arange(len(sel)), sel].sum()
        for _ in range(40):
            m = (lo+hi)/2.0; cd = ch(m); ct = pc[np.arange(len(cd)), cd].sum()
            if ct <= cap: hi, sel, tot = m, cd, ct
            else: lo = m
    if tot > cap: sel = np.zeros(len(ps), dtype=int)
    tc, ts = dv.cost[subset], dv.score[subset]
    ratio = tc[np.arange(len(sel)), sel].sum()/tc[:, 0].sum()
    passed = ratio <= mult + 1e-15
    return float(ts[np.arange(len(sel)), sel].mean()), float(ratio), bool(passed)

DEP = dict(fast=.98, balanced=.87, premium=.85)
rng = np.random.default_rng(20260819)
n = len(dv); NREP = 100
res = {t: {"greedy": [], "b02": [], "b04": [], "fix": []} for t in L.TIERS}
for rep in range(NREP):
    perm = rng.permutation(n); A, B = perm[:n//2], perm[n//2:]
    for t in L.TIERS:
        gA = [run(A, t, s, S[t], C[t]) for s in GRID]
        vA = np.array([r[0] if r[2] else 0.0 for r in gA])
        sA = float(GRID[int(vA.argmax())])
        for tag, s in (("greedy", sA), ("b02", max(0.6, sA-0.02)), ("b04", max(0.6, sA-0.04)), ("fix", DEP[t])):
            raw, ratio, ok = run(B, t, s, S[t], C[t])
            res[t][tag].append((raw if ok else 0.0, ok, raw))
print(f"transfer to the held-out half, {NREP} random 440/440 dev splits")
print("tier      rule                 E[tier score]  bust%   score|pass")
fin = {}
for t in L.TIERS:
    for tag, lbl in (("greedy","dev-greedy safety"),("b02","dev-greedy - 0.02"),
                     ("b04","dev-greedy - 0.04"),("fix","fixed .98/.87/.85")):
        a = np.array(res[t][tag], dtype=float)
        ev, bust, cond = a[:,0].mean(), 100*(1-a[:,1].mean()), a[:,2].mean()
        fin.setdefault(tag, 0.0)
        fin[tag] += L.TIER_WEIGHT[t]*ev
        print(f"{t:9s} {lbl:20s} {ev:.6f}      {bust:5.1f}   {cond:.6f}")
print()
for tag, lbl in (("greedy","dev-greedy safety"),("b02","dev-greedy - 0.02"),
                 ("b04","dev-greedy - 0.04"),("fix","fixed .98/.87/.85")):
    print(f"weighted final, {lbl:20s} = {fin[tag]:.6f}")
