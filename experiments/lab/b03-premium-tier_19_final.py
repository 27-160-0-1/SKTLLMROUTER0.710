# SPDX-License-Identifier: Apache-2.0
"""b03 step 19: the shippable package + a cross-tier margin flag."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P

lab = Lab()
cvB, arrB = B.stage(lab, DEPLOYED_EXP, tag="base")
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")

print("=== dev realised ratio vs cap, every tier, at a range of safeties (legoof) ===")
print(f"{'tier':<10}" + "".join(f"{g:>10.2f}" for g in np.arange(0.90, 1.001, 0.01)))
for t in ("fast",):
    ps, pc = lab.compose(arr, DEPLOYED_CFG, t)
    row = []
    for g in np.arange(0.90, 1.001, 0.01):
        pk = P.exact_allocate(ps, pc, MULTS[t], float(g))
        r = np.arange(len(arr["idx"]))
        rt = lab.true_c[arr["idx"]][r, pk].sum() / lab.true_c[arr["idx"]][:, 0].sum()
        row.append(f"{rt:10.4f}")
    print(f"{t:<10}" + "".join(row) + f"   cap={MULTS[t]}")
for nm, a in (("base", arrB), ("legoof", arr)):
    print(f"\n{nm}: dev margin (cap/ratio - 1) at the bench2 EV-optimal safety")
    sf = {"base": {"fast": .960, "balanced": .840, "premium": .735},
          "legoof": {"fast": .960, "balanced": .825, "premium": .840}}[nm]
    for t in TIERS:
        ps, pc = lab.compose(a, DEPLOYED_CFG, t)
        pk = P.exact_allocate(ps, pc, MULTS[t], sf[t])
        r = np.arange(len(a["idx"]))
        rt = lab.true_c[a["idx"]][r, pk].sum() / lab.true_c[a["idx"]][:, 0].sum()
        print(f"   {t:<10} safety={sf[t]:.3f} ratio={rt:.4f} cap={MULTS[t]:.2f}"
              f"  margin={100*(MULTS[t]/rt-1):6.2f}%")

print("\n=== candidate premium safeties, full package (fast/bal held at bench2 optimum) ===")
for pv in (0.76, 0.78, 0.79, 0.80, 0.82, 0.84):
    r = B.run(lab, cv, arr, DEPLOYED_CFG, label=f"legoof, premium safety {pv:.2f}",
              fixed_safety={"fast": 0.960, "balanced": 0.825, "premium": pv})
