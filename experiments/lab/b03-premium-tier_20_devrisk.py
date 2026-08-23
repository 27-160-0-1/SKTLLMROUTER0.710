# SPDX-License-Identifier: Apache-2.0
"""b03 step 20: risk measured on the DEV pool itself (the a09/a11/a12 convention),
so my numbers are comparable to round 1, for both fast and premium under legoof."""
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
di = arr["idx"]; n = len(di)
ts, tc = lab.true_s[di], lab.true_c[di]

def risk(a, tier, safeties, label):
    ps, pc = lab.compose(a, DEPLOYED_CFG, tier)
    ev = np.zeros(len(safeties)); bu = np.zeros(len(safeties))
    for s in (7, 17, 23):
        smp = np.asarray(lab.samples_for(n, s, 400, 880))
        e, b, _ = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MULTS[tier], safeties)
        ev += e / 3; bu += b / 3
    print(f"  {label:<28}" + "  ".join(f"{g:.2f}:{e:.4f}/{100*bb:4.1f}%"
                                       for g, e, bb in zip(safeties, ev, bu)))

print("DEV-pool bootstrap (880 resampled from dev, allocator re-solved) -- EV/bust%")
print("FAST  (cap 1.25)")
risk(arrB, "fast", np.array([0.94, 0.95, 0.96, 0.97, 0.98]), "base (legacy in-sample)")
risk(arr, "fast", np.array([0.94, 0.95, 0.96, 0.97, 0.98]), "legoof (C1)")
print("BALANCED (cap 2.0)")
risk(arr, "balanced", np.array([0.80, 0.825, 0.85, 0.87, 0.89]), "legoof (C1)")
print("PREMIUM (cap 4.0)")
risk(arrB, "premium", np.array([0.735, 0.79, 0.82, 0.84, 0.85, 0.88]), "base (legacy in-sample)")
risk(arr, "premium", np.array([0.735, 0.79, 0.82, 0.84, 0.85, 0.88]), "legoof (C1)")

print("\nWhere the fast cliff is on dev (legoof): ratio by safety")
ps, pc = lab.compose(arr, DEPLOYED_CFG, "fast")
prev = None
for g in np.arange(0.940, 0.9701, 0.002):
    pk = P.exact_allocate(ps, pc, 1.25, float(g))
    r = np.arange(n)
    rt = tc[r, pk].sum() / tc[:, 0].sum()
    print(f"   safety={g:.3f} ratio={rt:.4f} margin={100*(1.25/rt-1):6.2f}% "
          f"picks={np.bincount(pk,minlength=3).tolist()}")
