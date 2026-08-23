# SPDX-License-Identifier: Apache-2.0
"""Exchange rate: how much final score does a better score head buy, at several
cost-model qualities?  score_pred(lam) = (1-lam)*current + lam*true."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, tier_result

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)

def cal(t):
    C = P[f"cost_{t}"].copy()
    return C * (dv.cost.sum(0) / C.sum(0))[None, :]

COSTS = {
    "as-is":      lambda t: P[f"cost_{t}"],
    "calibrated": cal,
    "true":       lambda t: dv.cost,
}

print(f"{'cost model':12s} {'lam':>5s}  {'final':>7s}  corr(light/mid/k1)  per-tier")
for cname, cf in COSTS.items():
    for lam in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        tot = 0.0; parts = []
        cors = []
        for t in TIERS:
            S = (1 - lam) * P[f"score_{t}"] + lam * dv.score
            best = None
            for sf in np.arange(0.5, 1.301, 0.005):
                r = tier_result(S, cf(t), dv, t, float(sf))
                if r["passed"] and (best is None or r["score"] > best[0]):
                    best = (r["score"], float(sf), r["ratio"])
            tot += TIER_WEIGHT[t] * best[0]
            parts.append(f"{t[:4]}={best[0]:.4f}@{best[1]:.2f}")
            if t == "fast":
                cors = [np.corrcoef(S[:, j], dv.score[:, j])[0, 1] for j in range(3)]
        print(f"{cname:12s} {lam:5.2f}  {tot:7.4f}  " +
              "/".join(f"{c:.2f}" for c in cors) + "   " + " ".join(parts))
