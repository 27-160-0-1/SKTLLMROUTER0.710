# SPDX-License-Identifier: Apache-2.0
"""a08 probe 16: is the hidden num_generations multiplier still un-modelled by the
deployed pipeline?  Regress the deployed dev log-cost / score residual on ngen.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, tier_result  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
fam = np.array([classify_family(t) for t in dv.texts])
n4 = dv.ngen[:, 0] == 4
print(f"dev: ngen=4 on {n4.sum()}/{len(dv)} items; families "
      f"{sorted(set(fam[n4]))}")

print("\n=== deployed residuals split by ngen (premium tier predictions) ===")
for j, m in enumerate(MODEL_IDS):
    rc = np.log(P["cost_premium"][:, j]) - np.log(dv.cost[:, j])
    rs = P["score_premium"][:, j] - dv.score[:, j]
    print(f"  {m:11s} logcost resid: ngen2={rc[~n4].mean():+.3f} ngen4={rc[n4].mean():+.3f} "
          f"(gap {rc[n4].mean()-rc[~n4].mean():+.3f})   "
          f"score resid: ngen2={rs[~n4].mean():+.3f} ngen4={rs[n4].mean():+.3f}")

print("\n=== same, WITHIN family (controls for the family-level confound) ===")
for f in sorted(set(fam)):
    m = fam == f
    if (m & n4).sum() < 5 or (m & ~n4).sum() < 5:
        continue
    for j, mm in enumerate(MODEL_IDS):
        rc = np.log(P["cost_premium"][:, j]) - np.log(dv.cost[:, j])
        print(f"  {f:16s} {mm:11s} logcost resid ngen2={rc[m & ~n4].mean():+.3f} "
              f"(n={(m&~n4).sum()})  ngen4={rc[m & n4].mean():+.3f} (n={(m&n4).sum()})")

print("\n=== ceiling: multiply the predicted cost by true_ngen / family_modal_ngen ===")
fam_modal = {}
for f in sorted(set(fam)):
    v = dv.ngen[fam == f, 0]
    fam_modal[f] = 4.0 if (v == 4).mean() > 0.5 else 2.0
scale = dv.ngen[:, 0] / np.array([fam_modal[f] for f in fam])
print(f"  items whose ngen differs from the family mode: {(scale != 1).sum()}")


def run(label, mkc):
    tot = 0.0
    parts = []
    for t in TIERS:
        best = None
        for s in np.arange(0.60, 1.401, 0.005):
            r = tier_result(P[f"score_{t}"], mkc(t), dv, t, float(s))
            if r["passed"] and (best is None or r["score"] > best[0]):
                best = (r["score"], float(s))
        tot += TIER_WEIGHT[t] * best[0]
        parts.append(f"{t[:4]}={best[0]:.4f}@{best[1]:.3f}")
    print(f"  {label:44s} {tot:.4f}  " + " ".join(parts))


run("deployed", lambda t: P[f"cost_{t}"])
run("pred cost * (true ngen / family-modal ngen)", lambda t: P[f"cost_{t}"] * scale[:, None])
