# SPDX-License-Identifier: Apache-2.0
"""a13 / provider view - part 9: why the CV bust curve and the held-out bust
curve disagree, and what a decision under the pessimistic one costs.

Run: .venv/Scripts/python.exe experiments/lab/a13_provider_09_candidates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
sys.path.insert(0, str(ROOT / "src"))

import labdata  # noqa: E402
from labdata import TIERS, TIER_MULT, TIER_WEIGHT, allocate, tier_result  # noqa: E402

dv = labdata.load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}

print("=" * 88)
print("A. cost calibration of the held-out predictions (the Jensen gap)")
print("=" * 88)
for t in TIERS:
    S, C = P[f"score_{t}"], P[f"cost_{t}"]
    sel = allocate(S, C, dv.cost, TIER_MULT[t], SAFE[t])
    idx = np.arange(880)
    predL, trueL = C[:, 0].sum(), dv.cost[:, 0].sum()
    preds, trues = C[idx, sel].sum(), dv.cost[idx, sel].sum()
    print(f"  {t:9s} pred light total / true light total = {predL/trueL:.4f}")
    print(f"            pred selected / true selected      = {preds/trues:.4f}")
    print(f"            pred ratio {preds/predL:.4f} (cap {TIER_MULT[t]*SAFE[t]:.4f})  "
          f"true ratio {trues/trueL:.4f} (cap {TIER_MULT[t]:.2f})")
    for j in range(3):
        m = sel == j
        if m.sum():
            print(f"              model {j}: n={m.sum():4d} pred/true cost sum "
                  f"{C[m,j].sum()/dv.cost[m,j].sum():.4f}")

print()
print("=" * 88)
print("B. candidate safety triples, EV at several assumed private-set sizes")
print("=" * 88)
CANDS = {
    "deployed E43        .98/.87/.85": (0.98, 0.87, 0.85),
    "E39 min-regret      .985/.875/.81": (0.985, 0.875, 0.81),
    "part-3 optimum n880 .95/.85/.82": (0.95, 0.85, 0.82),
    "part-3 optimum n1760.97/.87/.82": (0.97, 0.87, 0.82),
    "uniform -3%         .95/.84/.82": (0.95, 0.84, 0.82),
    "very safe           .93/.82/.78": (0.93, 0.82, 0.78),
    "aggressive          1.00/.90/.88": (1.00, 0.90, 0.88),
}


def ev(triple, n, nboot=600, seeds=(7, 17)):
    tot = 0.0
    detail = {}
    for k, t in enumerate(TIERS):
        S, C = P[f"score_{t}"], P[f"cost_{t}"]
        vals, busts = [], []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            for _ in range(nboot):
                ix = rng.integers(0, 880, n)
                sel = allocate(S[ix], C[ix], dv.cost[ix], TIER_MULT[t], triple[k])
                idx = np.arange(n)
                r = dv.cost[ix][idx, sel].sum() / dv.cost[ix][:, 0].sum()
                ok = r <= TIER_MULT[t] + 1e-15
                vals.append(dv.score[ix][idx, sel].mean() if ok else 0.0)
                busts.append(0 if ok else 1)
        tot += TIER_WEIGHT[t] * np.mean(vals)
        detail[t] = (np.mean(vals), np.mean(busts) * 100)
    return tot, detail


print(f"{'candidate':36s} " + " ".join(f"{'EV@'+str(n):>9s}" for n in (880, 1760, 2640))
      + "   bust% @880 (f/b/p)   dev point score")
for name, tri in CANDS.items():
    row = f"{name:36s}"
    d880 = None
    for n in (880, 1760, 2640):
        v, d = ev(tri, n)
        if n == 880:
            d880 = d
        row += f" {v:9.4f}"
    # deterministic dev point score with this triple
    pt = 0.0
    for k, t in enumerate(TIERS):
        r = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, tri[k])
        pt += TIER_WEIGHT[t] * r["tier_score"]
    row += (f"   {d880['fast'][1]:4.1f}/{d880['balanced'][1]:4.1f}/"
            f"{d880['premium'][1]:4.1f}      {pt:.4f}")
    print(row)
