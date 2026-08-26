# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, MULTS, TIERS, DEPLOYED_CFG
import protocol as P, gainlab as G

lab = Lab(verbose=False)
cv, arr = G.stage1(lab)
m = len(cv["idx"]); smp = np.asarray(lab.samples_for(m, 7, 100, 880))
ts = lab.true_s[cv["idx"]][smp]; tc = lab.true_c[cv["idx"]][smp]
grid = np.arange(0.80, 1.041, 0.02)
for t in ("premium",):
    ps, pc = lab.compose(cv, DEPLOYED_CFG, t)
    PS, PC = ps[smp], pc[smp]
    t0 = time.perf_counter()
    ev1 = []
    for g in grid:
        pick = P.batch_allocate(PS, PC, MULTS[t], float(g))
        real = np.take_along_axis(tc, pick[:, :, None], axis=2)[:, :, 0].sum(axis=1)
        sc = np.take_along_axis(ts, pick[:, :, None], axis=2)[:, :, 0].mean(axis=1)
        light = tc[:, :, 0].sum(axis=1)
        ev1.append(float(np.mean(np.where(real/light > MULTS[t], 0.0, sc))))
    t1 = time.perf_counter()
    ev2, bust, raw = P.safety_curve(PS, PC, ts, tc, MULTS[t], grid)
    t2 = time.perf_counter()
    print(f"{t}: maxabsdiff={np.max(np.abs(np.array(ev1)-ev2)):.3e}  old {t1-t0:.2f}s  new {t2-t1:.3f}s")
    print("  ev:", np.round(ev2, 4).tolist())
    print("  bust:", np.round(bust, 3).tolist())
