# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b08 - the PHB noise floor: how big must a gain be before it means anything?"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG            # noqa: E402
import bench2 as B                                                # noqa: E402
from importlib import import_module                               # noqa: E402
PHB = import_module("b08-chief-strategist_10_phb")

R = 16
KEYS = PHB.KEYS

if __name__ == "__main__":
    t0 = time.perf_counter()
    lab = Lab()
    res = pickle.loads(Path("reports/lab/b08_rot_arr.pkl").read_bytes())
    cvL, arrL = B.stage(lab, dict(legacy_oof_meta=True), tag="legoof")
    arrs = [res["C1"][s] for s in range(R)] + [{k: arrL[k] for k in KEYS}]
    pl = PHB.pool(lab, arrs, DEPLOYED_CFG)

    print("=== A. resample-seed noise of the PHB EV (same pool, different bootstrap seeds) ===")
    evs = []
    for sd in [(11, 29, 47), (53, 71, 89), (101, 103, 107), (211, 223, 227), (307, 311, 313)]:
        c = PHB.curves(pl, B.GRIDS, seeds=sd)
        sfy = {t: float(c[t]["grid"][int(np.argmax(c[t]["ev"]))]) for t in TIERS}
        EV = sum(W[t] * c[t]["ev"].max() for t in TIERS)
        evs.append(EV)
        print(f"  seeds={sd} EV={EV:.6f} s={'/'.join('%.3f' % sfy[t] for t in TIERS)}")
    evs = np.array(evs)
    print(f"  -> sd over seed triples = {evs.std(ddof=1):.6f}")

    print("\n=== B. replicate-pool noise: leave-one-replicate-out jackknife of the PHB EV ===")
    jk = []
    for k in range(len(arrs)):
        sub = [a for i, a in enumerate(arrs) if i != k]
        p2 = PHB.pool(lab, sub, DEPLOYED_CFG)
        _, _, EV = PHB.choose(p2, B.GRIDS)
        jk.append(EV)
    jk = np.array(jk)
    n = len(jk)
    se = np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum())
    print(f"  jackknife mean={jk.mean():.6f} min={jk.min():.6f} max={jk.max():.6f}")
    print(f"  -> jackknife SE of the PHB EV level = {se:.6f}")

    print("\n=== C. PAIRED difference noise: C1 vs C1+kk1.24, same pool, same resamples ===")
    def mk(kk):
        def f(lab, arr, ps, pc, tier):
            pc = pc.copy(); pc[:, 2] *= kk
            pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
            return ps, pc
        return f
    pl2 = PHB.pool(lab, arrs, DEPLOYED_CFG, mk(1.24))
    d = []
    for sd in [(11, 29, 47), (53, 71, 89), (101, 103, 107), (211, 223, 227), (307, 311, 313)]:
        c1 = PHB.curves(pl, B.GRIDS, seeds=sd); c2 = PHB.curves(pl2, B.GRIDS, seeds=sd)
        e1 = sum(W[t] * c1[t]["ev"].max() for t in TIERS)
        e2 = sum(W[t] * c2[t]["ev"].max() for t in TIERS)
        d.append(e2 - e1)
    d = np.array(d)
    print(f"  paired dEV per seed triple: {np.round(d, 6)}")
    print(f"  mean={d.mean():+.6f} sd={d.std(ddof=1):.6f}")
    print(f"[b08] done ({time.perf_counter()-t0:.0f}s)")
