# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Where does the 0.10 gap to the oracle live?  Counterfactual ablations."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result, MODEL_IDS

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
n = len(dv)

def report(name, get_pred):
    tot = 0.0; parts = []
    for t in TIERS:
        ps, pc, sf = get_pred(t)
        r = tier_result(ps, pc, dv, t, sf)
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else ' BUST'}")
    print(f"{name:52s} final={tot:.4f}   " + "  ".join(parts))
    return tot

base = report("deployed (pred score, pred cost, safety)", lambda t: (P[f"score_{t}"], P[f"cost_{t}"], SAFE[t]))
report("pred score, pred cost, safety=1.0", lambda t: (P[f"score_{t}"], P[f"cost_{t}"], 1.0))
report("pred score, TRUE cost, safety=1.0", lambda t: (P[f"score_{t}"], dv.cost, 1.0))
report("TRUE score, pred cost, safety(dep)", lambda t: (dv.score, P[f"cost_{t}"], SAFE[t]))
report("TRUE score, TRUE cost (oracle)", lambda t: (dv.score, dv.cost, 1.0))

print()
print("=== cost prediction quality (dev) ===")
for t in ("premium",):
    for j, m in enumerate(MODEL_IDS):
        pc = P[f"cost_{t}"][:, j]; tc = dv.cost[:, j]
        lr = np.log(pc) - np.log(tc)
        print(f"  {t:8s} {m:11s} pred_sum={pc.sum():.3f} true_sum={tc.sum():.3f} "
              f"ratio={pc.sum()/tc.sum():.4f}  log-err mean={lr.mean():+.3f} sd={lr.std():.3f} "
              f"corr={np.corrcoef(np.log(pc), np.log(tc))[0,1]:.3f}")
print()
print("=== score prediction quality (dev) ===")
for t in TIERS:
    for j, m in enumerate(MODEL_IDS):
        ps = P[f"score_{t}"][:, j]; ts = dv.score[:, j]
        print(f"  {t:8s} {m:11s} pred_mean={ps.mean():.4f} true_mean={ts.mean():.4f} "
              f"corr={np.corrcoef(ps, ts)[0,1]:.3f} mae={np.abs(ps-ts).mean():.3f}")
    break
print()
print("=== gain (delta score) prediction ===")
t = "premium"
for (a_, b_, lab) in ((0,1,"mid-light"), (1,2,"k1-mid"), (0,2,"k1-light")):
    pg = P[f"score_{t}"][:, b_] - P[f"score_{t}"][:, a_]
    tg = dv.score[:, b_] - dv.score[:, a_]
    print(f"  {lab:10s} corr={np.corrcoef(pg,tg)[0,1]:.3f} pred_mean={pg.mean():+.4f} true_mean={tg.mean():+.4f}")

print()
print("=== budget headroom ===")
for t in TIERS:
    r = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, SAFE[t])
    print(f"  {t:9s} used ratio {r['ratio']:.3f} of {TIER_MULT[t]}  ({100*r['ratio']/TIER_MULT[t]:.1f}%)")

print()
print("=== score value distribution ===")
for j, m in enumerate(MODEL_IDS):
    vals, cnt = np.unique(dv.score[:, j], return_counts=True)
    print(f"  {m:11s} " + " ".join(f"{v:.2f}:{c}" for v, c in zip(vals, cnt)))
print("  num_generations:", np.unique(dv.ngen, return_counts=True))
