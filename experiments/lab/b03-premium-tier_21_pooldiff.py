# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b03 step 21: why does the train-OOF pool say 0% bust where the dev pool says 10-42%?"""
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

SF = {"fast": 0.960, "balanced": 0.825, "premium": 0.840}
print(f"{'cfg':<8}{'pool':<10}{'tier':<10}{'ratio':>8}{'margin%':>9}{'b_N':>7}{'b_D':>7}"
      f"{'b_N/b_D':>9}{'k_L':>7}{'k_M':>7}{'k_K':>7}{'sdlogR':>8}{'bust%':>7}")
for cfgn, (c, a) in (("base", (cvB, arrB)), ("legoof", (cv, arr))):
    for pooln, x in (("train-OOF", c), ("dev", a)):
        idx = x["idx"]; ts = lab.true_s[idx]; tc = lab.true_c[idx]; n = len(idx)
        for t in TIERS:
            ps, pc = lab.compose(x, DEPLOYED_CFG, t)
            pk = P.exact_allocate(ps, pc, MULTS[t], SF[t])
            r = np.arange(n)
            rt = tc[r, pk].sum() / tc[:, 0].sum()
            bN = tc[r, pk].sum() / pc[r, pk].sum()
            bD = tc[:, 0].sum() / pc[:, 0].sum()
            k = [tc[:, j].sum() / pc[:, j].sum() for j in range(3)]
            ev = np.zeros(1); bu = np.zeros(1)
            for s in (7, 17, 23):
                smp = np.asarray(lab.samples_for(n, s, 400, 880))
                e, b, _ = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MULTS[t],
                                         np.array([SF[t]]))
                ev += e / 3; bu += b / 3
            smp = np.asarray(lab.samples_for(n, 7, 400, 880))
            pk2 = P.exact_allocate(ps[smp], pc[smp], MULTS[t], SF[t])
            C = np.take_along_axis(tc[smp], pk2[:, :, None], axis=2)[:, :, 0]
            R = C.sum(axis=1) / tc[smp][:, :, 0].sum(axis=1)
            print(f"{cfgn:<8}{pooln:<10}{t:<10}{rt:8.4f}{100*(MULTS[t]/rt-1):9.2f}"
                  f"{bN:7.3f}{bD:7.3f}{bN/bD:9.3f}{k[0]:7.3f}{k[1]:7.3f}{k[2]:7.3f}"
                  f"{np.log(R).std():8.4f}{100*bu[0]:7.2f}")
print("\nlight-cost prediction quality (the budget denominator):")
for cfgn, (c, a) in (("base", (cvB, arrB)), ("legoof", (cv, arr))):
    for pooln, x in (("train-OOF", c), ("dev", a)):
        idx = x["idx"]
        ps, pc = lab.compose(x, DEPLOYED_CFG, "fast")
        e = np.log(lab.true_c[idx][:, 0]) - np.log(pc[:, 0])
        print(f"  {cfgn:<8}{pooln:<10} light log-rmse={np.sqrt((e**2).mean()):.3f}"
              f" sum ratio true/pred={lab.true_c[idx][:,0].sum()/pc[:,0].sum():.3f}"
              f" | mid={lab.true_c[idx][:,1].sum()/pc[:,1].sum():.3f}"
              f" k1={lab.true_c[idx][:,2].sum()/pc[:,2].sum():.3f}")
