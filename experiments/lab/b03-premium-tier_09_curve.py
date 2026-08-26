# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b03 step 9: premium safety curve, blend_premium sweep, cost-oracle counterfactuals."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
ci = cv["idx"]; di = arr["idx"]
ts_c, tc_c = lab.true_s[ci], lab.true_c[ci]
ts_d, tc_d = lab.true_s[di], lab.true_c[di]
m = len(ci)

grid = np.arange(0.60, 1.101, 0.01)
ps, pc = lab.compose(cv, DEPLOYED_CFG, "premium")
ev = np.zeros(len(grid)); bu = np.zeros(len(grid)); raw = np.zeros(len(grid))
for s in (7, 17, 23):
    smp = np.asarray(lab.samples_for(m, s, 400, 880))
    e, b, r = P.safety_curve(ps[smp], pc[smp], ts_c[smp], tc_c[smp], 4.0, grid)
    ev += e / 3; bu += b / 3; raw += r / 3
psd, pcd = lab.compose(arr, DEPLOYED_CFG, "premium")
print(f"{'safety':>7}{'OOF EV':>9}{'bust%':>8}{'score|pass':>11} | {'dev score':>10}{'dev ratio':>10}{'pass':>6}")
for i, g in enumerate(grid):
    pk = P.exact_allocate(psd, pcd, 4.0, float(g))
    r = np.arange(len(di))
    rt = tc_d[r, pk].sum() / tc_d[:, 0].sum()
    sc = ts_d[r, pk].mean()
    print(f"{g:7.2f}{ev[i]:9.4f}{100*bu[i]:8.2f}{raw[i]:11.4f} | {sc:10.4f}{rt:10.3f}"
          f"{'  ok' if rt <= 4.0 else ' BUST':>6}")

print("\n--- blend_premium sweep (per-tier meta weight; deployed 0.30) ---")
for b in (0.20, 0.30, 0.40, 0.50, 0.60, 0.75):
    r = B.run(lab, cv, arr, dict(DEPLOYED_CFG, blend_premium=b), label=f"blend_premium={b}")
print("\n--- fast/balanced blends held; premium only reported above ---")
