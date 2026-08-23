# SPDX-License-Identifier: Apache-2.0
"""a11 step 15 - oracle veto frontier: how much is a long-output detector worth,
and does it stack with the kappa2 relative-price correction?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router import similarity

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv)
fam = np.array([similarity.classify_family(x) for x in dv.texts])
opg = dv.otok[:, 2] / dv.ngen[:, 2]


def best_passing(tier, veto, kappa2=1.0, grid=np.arange(0.60, 1.601, 0.005)):
    pc = P[f"cost_{tier}"] * np.array([1.0, 1.0, kappa2])[None, :]
    pc = pc.copy(); pc[veto, 2] = 1e9
    best = None
    for s in grid:
        r = tier_result(P[f"score_{tier}"], pc, dv, tier, float(s))
        if r["passed"] and (best is None or r["score"] > best[1]):
            best = (float(s), r["score"], r["ratio"], int((r["sel"] == 2).sum()))
    return best


print("=== oracle veto frontier (dev, best passing safety per configuration) ===")
VETOS = [("none", np.zeros(n, bool)),
         ("otok/gen >= 32k", opg >= 32000),
         ("otok/gen >= 20k", opg >= 20000),
         ("otok/gen >= 10k", opg >= 10000),
         ("otok/gen >=  5k", opg >= 5000),
         ("all aime", fam == "aime"),
         ("all aime+code", np.isin(fam, ["aime", "code"]))]
for tier in TIERS:
    print(f"-- {tier}")
    print(f"   {'veto':18s} {'n':>4s} {'s*':>6s} {'score':>7s} {'ratio':>7s} {'n_k1':>5s} {'delta':>8s}")
    ref = None
    for label, v in VETOS:
        b = best_passing(tier, v)
        if ref is None:
            ref = b[1]
        print(f"   {label:18s} {int(v.sum()):4d} {b[0]:6.3f} {b[1]:7.4f} {b[2]:7.4f} {b[3]:5d} "
              f"{b[1]-ref:+8.4f}")

print("\n=== does the veto stack with kappa2? (premium) ===")
for k2 in (1.0, 1.24, 1.5, 2.0):
    for label, v in VETOS[:4]:
        b = best_passing("premium", v, kappa2=k2)
        print(f"   kappa2={k2:4.2f} veto={label:18s} s*={b[0]:.3f} score={b[1]:.4f} "
              f"ratio={b[2]:.4f} n_k1={b[3]}")
    print()

print("=== weighted dev final for the stacked configurations ===")
for k2, vlabel, v in ((1.0, "none", np.zeros(n, bool)),
                      (1.5, "none", np.zeros(n, bool)),
                      (1.0, "otok/gen>=32k", opg >= 32000),
                      (1.5, "otok/gen>=32k", opg >= 32000),
                      (1.5, "otok/gen>=10k", opg >= 10000)):
    tot = 0.0; parts = []
    for tier in TIERS:
        b = best_passing(tier, v, kappa2=k2)
        tot += TIER_WEIGHT[tier] * b[1]
        parts.append(f"{tier[:4]}={b[1]:.4f}@s{b[0]:.3f}")
    print(f"   kappa2={k2:4.2f} veto={vlabel:14s} final={tot:.4f}   " + " ".join(parts))
print("   (all numbers are dev-tuned upper bounds: the safety scalar is chosen")
print("    with knowledge of the realised ratio, so treat them as a ceiling.)")
