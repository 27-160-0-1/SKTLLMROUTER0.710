# SPDX-License-Identifier: Apache-2.0
"""a13 / provider view - part 12: does per-model cost calibration reduce the
BUST RISK (not just the dev point score)?

Part 9 measured that at premium the selected set's predicted cost is 0.777 of the
true cost while the light denominator is 0.816 of true -> a 5.1% systematic
under-pricing of exactly the set we spend on, on top of ~10% sampling noise.
A single global safety scalar can absorb the mean but is then paying for a bias
it could have removed.

Calibration constants are estimated ON DEV here (in-sample) => the numbers are an
UPPER BOUND on what an honest out-of-fold calibration could deliver.
Run: .venv/Scripts/python.exe experiments/lab/a13_provider_12_calib.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
sys.path.insert(0, str(ROOT / "src"))

import labdata  # noqa: E402
from labdata import TIERS, TIER_MULT, TIER_WEIGHT, allocate  # noqa: E402

dv = labdata.load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}

# per-model sum-calibration factors (in-sample on dev)
K = {}
for t in TIERS:
    C = P[f"cost_{t}"]
    K[t] = np.array([dv.cost[:, j].sum() / C[:, j].sum() for j in range(3)])
    print(f"  {t:9s} per-model sum calibration factors {np.round(K[t],4)}")
print()


def ev(tier, safety, n, calib, nboot=500, seeds=(7, 17)):
    S = P[f"score_{tier}"]
    C = P[f"cost_{tier}"] * (K[tier] if calib else 1.0)
    mult = TIER_MULT[tier]
    vals, busts = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        for _ in range(nboot):
            ix = rng.integers(0, 880, n)
            sel = allocate(S[ix], C[ix], dv.cost[ix], mult, safety)
            idx = np.arange(n)
            r = dv.cost[ix][idx, sel].sum() / dv.cost[ix][:, 0].sum()
            ok = r <= mult + 1e-15
            vals.append(dv.score[ix][idx, sel].mean() if ok else 0.0)
            busts.append(0 if ok else 1)
    return float(np.mean(vals)), float(np.mean(busts)) * 100


GRID = {
    "fast": [0.90, 0.93, 0.95, 0.97, 0.98, 1.00, 1.02],
    "balanced": [0.78, 0.82, 0.85, 0.87, 0.90, 0.93, 0.97],
    "premium": [0.74, 0.78, 0.82, 0.85, 0.88, 0.92, 0.96],
}
for n in (880, 1760):
    print("=" * 88)
    print(f"n = {n}")
    print("=" * 88)
    tot_raw = tot_cal = 0.0
    for t in TIERS:
        rows = []
        for sf in GRID[t]:
            e0, b0 = ev(t, sf, n, False)
            e1, b1 = ev(t, sf, n, True)
            rows.append((sf, e0, b0, e1, b1))
        print(f"-- {t}   {'safety':>7s} {'raw EV':>8s} {'bust%':>6s}   "
              f"{'calib EV':>9s} {'bust%':>6s}")
        for sf, e0, b0, e1, b1 in rows:
            mark = " <= deployed" if abs(sf - SAFE[t]) < 1e-9 else ""
            print(f"   {'':11s}{sf:7.2f} {e0:8.4f} {b0:6.1f}   {e1:9.4f} {b1:6.1f}{mark}")
        b_raw = max(rows, key=lambda r: r[1])
        b_cal = max(rows, key=lambda r: r[3])
        tot_raw += TIER_WEIGHT[t] * b_raw[1]
        tot_cal += TIER_WEIGHT[t] * b_cal[3]
        print(f"   best raw   safety {b_raw[0]:.2f} EV {b_raw[1]:.4f}")
        print(f"   best calib safety {b_cal[0]:.2f} EV {b_cal[3]:.4f}")
    print(f"  WEIGHTED best-safety EV: raw {tot_raw:.4f}  calibrated {tot_cal:.4f}  "
          f"delta {tot_cal-tot_raw:+.4f}")
