# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a10 P6 - which of the proposed 'reparameterisations' are no-ops?

The brief asks whether the allocator should be fed gains / cost deltas /
efficiency ranks instead of levels.  Three of these are provably no-ops; this
script verifies that numerically so nobody spends time on them.

  util_im = s_im - lam * c_im / L ,  sel_i = argmax_m util_im
  ledger  = sum_i c_i,sel  <=  L * mult * safety

  (a) s -> s - s_0 (per-row shift): argmax unchanged  => NO-OP
  (b) c -> c - c_0 with cap -> (mult*safety - 1)*L  : selection and feasible
      set identical                                  => NO-OP
  (c) c -> c / c_0 (per-item ratio) with w_i = c_i0/L: this is a per-row
      RESCALE of the utility cost, NOT a shift -> it IS a different allocator
      (it changes the relative penalty across items).  Verified different.

Also prints the risk table for the deployed safety triple measured on the
held-out dev predictions.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_MULT, TIER_WEIGHT, allocate
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv); ar = np.arange(n)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
fam = np.array([classify_family(t) for t in dv.texts])


def alloc_general(score, ucost, cap_abs, lcost):
    a = np.arange(len(score)); denom = ucost[:, 0].sum() if ucost[:, 0].sum() else 1.0

    def choose(pen):
        return (score - pen * ucost / denom).argmax(axis=1)
    sel = choose(0.0); tot = lcost[a, sel].sum()
    if tot > cap_abs:
        low, high = 0.0, 1.0
        sel = choose(high); tot = lcost[a, sel].sum()
        while tot > cap_abs and high < 2 ** 60:
            low, high = high, high * 2.0
            sel = choose(high); tot = lcost[a, sel].sum()
        for _ in range(40):
            mid = (low + high) / 2.0
            cand = choose(mid); ct = lcost[a, cand].sum()
            if ct <= cap_abs:
                high, sel, tot = mid, cand, ct
            else:
                low = mid
    if tot > cap_abs:
        sel = np.zeros(len(score), dtype=int)
    return sel


print("=" * 86)
print("(1) no-op checks  (identical selection vector required)")
for t in TIERS:
    S, C = P[f"score_{t}"], P[f"cost_{t}"]
    L = C[:, 0].sum(); cap = L * max(1.0, TIER_MULT[t] * SAFE[t])
    ref = allocate(S, C, dv.cost, TIER_MULT[t], SAFE[t])
    a = alloc_general(S - S[:, :1], C, cap, C)                       # gains
    b = alloc_general(S, C - C[:, :1], cap - L, C - C[:, :1])        # cost deltas
    r = C / C[:, :1]
    c = alloc_general(S, r, cap, C)                                  # per-item ratio
    print(f"  {t:9s} gains-identical={bool((a==ref).all())}  "
          f"cost-deltas-identical={bool((b==ref).all())}  "
          f"ratio-input-identical={bool((c==ref).all())} "
          f"(ratio differs on {int((c!=ref).sum())} items)")

print()
print("=" * 86)
print("(2) code x k1 economics on the DEV held-out predictions (premium tier)")
S, C = P["score_premium"], P["cost_premium"]
sel = allocate(S, C, dv.cost, 4.0, SAFE["premium"])
Lp, Lt = C[:, 0].sum(), dv.cost[:, 0].sum()
print(f"  {'family':14s} {'#k1':>4s} {'believed dc':>12s} {'true dc':>9s} "
      f"{'believed ds':>12s} {'true ds':>9s} {'believed eff':>13s} {'true eff':>9s}")
for f in sorted(set(fam)):
    m = (fam == f) & (sel == 2)
    if m.sum() == 0:
        continue
    bdc = (C[m, 2] - C[m, 1]).sum() / Lp
    tdc = (dv.cost[m, 2] - dv.cost[m, 1]).sum() / Lt
    bds = (S[m, 2] - S[m, 1]).sum() / n
    tds = (dv.score[m, 2] - dv.score[m, 1]).sum() / n
    print(f"  {f:14s} {int(m.sum()):4d} {bdc:12.4f} {tdc:9.4f} {bds:12.4f} {tds:9.4f} "
          f"{bds/bdc:13.4f} {tds/tdc:9.4f}")

print()
print("=" * 86)
print("(3) risk of the deployed safety triple, measured on held-out dev preds")
print("    880-item bootstrap, 600 draws, seed 7")
rng = np.random.default_rng(7)
B = 600
bat = [rng.integers(0, n, n) for _ in range(B)]
for t in TIERS:
    S, C = P[f"score_{t}"], P[f"cost_{t}"]
    for sf in (SAFE[t], SAFE[t] - 0.03, SAFE[t] - 0.06, SAFE[t] - 0.10, SAFE[t] - 0.15):
        vals = np.empty(B); nb = 0
        for i, rows in enumerate(bat):
            s = allocate(S[rows], C[rows], dv.cost[rows], TIER_MULT[t], float(sf))
            tc = dv.cost[rows]
            ok = tc[ar, s].sum() / tc[:, 0].sum() <= TIER_MULT[t] + 1e-15
            vals[i] = dv.score[rows][ar, s].mean() if ok else 0.0
            nb += 0 if ok else 1
        print(f"    {t:9s} safety={sf:.3f}  bust={nb/B:6.3f}  EV={vals.mean():.4f}  "
              f"E[score|pass]={vals[vals>0].mean() if (vals>0).any() else 0:.4f}")
    print()
