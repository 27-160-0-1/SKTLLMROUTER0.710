# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Marginal value of perfect knowledge, column by column."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, TIER_MULT, tier_result

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}

def run(name, mk_s, mk_c, safety=None):
    tot = 0.0; parts = []
    for t in TIERS:
        sf = SAFE[t] if safety is None else safety[t]
        r = tier_result(mk_s(t), mk_c(t), dv, t, sf)
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else '!'}")
    print(f"{name:56s} {tot:.4f}  " + " ".join(parts))
    return tot

ps = lambda t: P[f"score_{t}"]
pc = lambda t: P[f"cost_{t}"]
base = run("baseline (E43 holdout)", ps, pc)

print("\n--- perfect score for ONE model at a time (pred costs) ---")
for j, m in enumerate(MODEL_IDS):
    def mk(t, j=j):
        S = P[f"score_{t}"].copy(); S[:, j] = dv.score[:, j]; return S
    run(f"  true score[{m}]", mk, pc)

print("\n--- perfect cost for ONE model at a time (pred scores) ---")
for j, m in enumerate(MODEL_IDS):
    def mk(t, j=j):
        C = P[f"cost_{t}"].copy(); C[:, j] = dv.cost[:, j]; return C
    run(f"  true cost[{m}]", ps, mk)

print("\n--- oracle-safety: allocate with pred cost, safety tuned so TRUE ratio hits the cap ---")
def best_safety(t):
    lo, hi = 0.3, 1.6
    best = None
    for sf in np.arange(0.5, 1.501, 0.005):
        r = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, float(sf))
        if r["passed"]:
            if best is None or r["score"] > best[1]:
                best = (float(sf), r["score"], r["ratio"])
    return best
tot = 0.0
for t in TIERS:
    sf, sc, ra = best_safety(t)
    tot += TIER_WEIGHT[t] * sc
    print(f"  {t:9s} best safety={sf:.3f} score={sc:.4f} true_ratio={ra:.3f}")
print(f"  ORACLE-SAFETY final = {tot:.4f}  (baseline {base:.4f}, +{tot-base:.4f})")

print("\n--- decompose: rescale predicted costs per model to remove the sum bias ---")
for t in TIERS:
    C = P[f"cost_{t}"]
    k = dv.cost.sum(0) / C.sum(0)
    print(f"  {t:9s} per-model calibration factors: {np.round(k,4)}")
def mkc_cal(t):
    C = P[f"cost_{t}"].copy()
    C *= (dv.cost.sum(0) / C.sum(0))[None, :]
    return C
run("calibrated cost (per-model sum matched), safety=dep", ps, mkc_cal)
tot = 0.0
for t in TIERS:
    best = None
    for sf in np.arange(0.5, 1.501, 0.005):
        r = tier_result(P[f"score_{t}"], mkc_cal(t), dv, t, float(sf))
        if r["passed"] and (best is None or r["score"] > best[1]):
            best = (float(sf), r["score"], r["ratio"])
    tot += TIER_WEIGHT[t] * best[1]
    print(f"  calibrated + best safety {t:9s}: safety={best[0]:.3f} score={best[1]:.4f} ratio={best[2]:.3f}")
print(f"  final={tot:.4f}")

print("\n--- what if we knew the TRUE light-baseline total only (denominator)? ---")
for t in TIERS:
    C = P[f"cost_{t}"].copy()
    C[:, 0] *= dv.cost[:, 0].sum() / C[:, 0].sum()
    r = tier_result(P[f"score_{t}"], C, dv, t, SAFE[t])
    print(f"  {t:9s} score={r['score']:.4f} ratio={r['ratio']:.3f} passed={r['passed']}")
