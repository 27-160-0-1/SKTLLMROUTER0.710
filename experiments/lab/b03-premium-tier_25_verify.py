# SPDX-License-Identifier: Apache-2.0
"""b03 step 25: robustness of the headline risk numbers (different seeds, subsampling
without replacement, leave-one-item-out)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P

lab = Lab()
cvB, arrB = B.stage(lab, DEPLOYED_EXP, tag="base")
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
di = arr["idx"]; n = len(di); ts, tc = lab.true_s[di], lab.true_c[di]

def bust(a, tier, g, size, replace=True, seeds=(31, 37, 41, 43)):
    ps, pc = lab.compose(a, DEPLOYED_CFG, tier)
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        if replace:
            smp = rng.integers(0, n, size=(300, size))
        else:
            smp = np.array([rng.permutation(n)[:size] for _ in range(300)])
        pk = P.exact_allocate(ps[smp], pc[smp], MULTS[tier], g)
        C = np.take_along_axis(tc[smp], pk[:, :, None], axis=2)[:, :, 0]
        R = C.sum(axis=1) / tc[smp][:, :, 0].sum(axis=1)
        out.append(R > MULTS[tier])
    return float(np.concatenate(out).mean())

print("FAST @0.960, dev predictions -- bust probability under different resampling")
for nm, a in (("base", arrB), ("legoof(C1)", arr)):
    print(f"  {nm:<12} boot880={100*bust(a,'fast',0.96,880):5.2f}%"
          f"  boot440={100*bust(a,'fast',0.96,440):5.2f}%"
          f"  subsample440(no repl)={100*bust(a,'fast',0.96,440,False):5.2f}%"
          f"  subsample700(no repl)={100*bust(a,'fast',0.96,700,False):5.2f}%")
print("PREMIUM @0.840 / @0.760, dev predictions")
for g in (0.84, 0.76):
    for nm, a in (("base", arrB), ("legoof(C1)", arr)):
        print(f"  s={g} {nm:<12} boot880={100*bust(a,'premium',g,880):5.2f}%"
              f"  subsample440(no repl)={100*bust(a,'premium',g,440,False):5.2f}%")
# leave-one-item-out sensitivity of the fast dev ratio
ps, pc = lab.compose(arr, DEPLOYED_CFG, "fast")
pk = P.exact_allocate(ps, pc, 1.25, 0.96); r = np.arange(n)
num = tc[r, pk]; den = tc[:, 0]
base_r = num.sum() / den.sum()
lo = (num.sum() - num) / (den.sum() - den)
print(f"\nfast dev ratio {base_r:.4f}; leave-one-out range [{lo.min():.4f}, {lo.max():.4f}]"
      f"; items whose removal alone would push it over 1.25: {int((lo>1.25).sum())}")
