# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b08 chief strategist - verification pass 1.

Reproduce the two BRIEF2 reference stages and the disputed dev fast-tier margin.
Adds a DEV-POOL bust column (b03 P4) computed with the same bootstrap machinery
as bench2's train-OOF bust, so the two are directly comparable.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG            # noqa: E402
import bench2 as B                                                # noqa: E402
import protocol as P                                              # noqa: E402


def dev_pool_risk(lab, arr, cfg, safety, seeds=(101, 103, 107), nboot=400, transform=None):
    """Bootstrap 880-item resamples of the DEV rows at the chosen safety."""
    di = arr["idx"]; ts = lab.true_s[di]; tc = lab.true_c[di]; m = len(di)
    out = {}
    for t in TIERS:
        ps, pc = lab.compose(arr, cfg, t)
        if transform is not None:
            ps, pc = transform(lab, arr, ps, pc, t)
        g = np.array([safety[t]])
        ev = bu = raw = 0.0
        for s in seeds:
            smp = np.asarray(lab.samples_for(m, s, nboot, 880))
            e, b, r = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MULTS[t], g)
            ev += e[0] / len(seeds); bu += b[0] / len(seeds); raw += r[0] / len(seeds)
        out[t] = dict(ev=ev, bust=bu, raw=raw)
    out["EV"] = sum(W[t] * out[t]["ev"] for t in TIERS)
    return out


def show(lab, cv, arr, cfg, label, transform=B.ident, fixed=None):
    r = B.run(lab, cv, arr, cfg, transform=transform, label=label, fixed_safety=fixed)
    dp = dev_pool_risk(lab, arr, cfg or DEPLOYED_CFG, r["safety"],
                       transform=None if transform is B.ident else transform)
    print(f"    {'':32s} devpoolEV={dp['EV']:.6f} devbust%="
          + "/".join(f"{dp[t]['bust']*100:.1f}" for t in TIERS)
          + "  margins=" + "/".join(f"{r['dev_tiers'][t]['margin']*100:+.2f}%" for t in TIERS),
          flush=True)
    r["devpool"] = dp
    return r


if __name__ == "__main__":
    t0 = time.perf_counter()
    lab = Lab()
    cvB, arrB = B.stage(lab, None, tag="base")
    cvL, arrL = B.stage(lab, dict(legacy_oof_meta=True), tag="legoof")
    print(f"[b08] stages ready ({time.perf_counter()-t0:.0f}s)", flush=True)

    print("\n=== 1. reproduce BRIEF2 section 5 ===")
    rb = show(lab, cvB, arrB, None, "base (no C1)")
    rl = show(lab, cvL, arrL, None, "C1 legacy-OOF meta")

    print("\n=== 2. fast-tier dev ratio ladder under C1 (cap 1.25) ===")
    ps, pc = lab.compose(arrL, DEPLOYED_CFG, "fast")
    di = arrL["idx"]
    for s in np.arange(0.90, 0.9801, 0.005):
        pick = lab.allocate(ps, pc, 1.25, float(s))
        r = np.arange(len(di))
        ratio = lab.true_c[di][r, pick].sum() / lab.true_c[di][:, 0].sum()
        sc = lab.true_s[di][r, pick].mean()
        print(f"  fast safety {s:.3f}  dev ratio {ratio:.4f}  margin {1.25/ratio-1:+.3%}  "
              f"score {sc:.4f}{'  BUST' if ratio > 1.25 else ''}")

    print("\n=== 3. train-OOF vs dev-pool bust across the fast grid (C1) ===")
    ci = cvL["idx"]
    psc, pcc = lab.compose(cvL, DEPLOYED_CFG, "fast")
    g = np.arange(0.88, 1.001, 0.01)
    ev = np.zeros(len(g)); bu = np.zeros(len(g))
    for s in (7, 17, 23):
        smp = np.asarray(lab.samples_for(len(ci), s, 400, 880))
        e, b, _ = P.safety_curve(psc[smp], pcc[smp], lab.true_s[ci][smp], lab.true_c[ci][smp], 1.25, g)
        ev += e / 3; bu += b / 3
    ev2 = np.zeros(len(g)); bu2 = np.zeros(len(g))
    psd, pcd = lab.compose(arrL, DEPLOYED_CFG, "fast")
    for s in (101, 103, 107):
        smp = np.asarray(lab.samples_for(len(di), s, 400, 880))
        e, b, _ = P.safety_curve(psd[smp], pcd[smp], lab.true_s[di][smp], lab.true_c[di][smp], 1.25, g)
        ev2 += e / 3; bu2 += b / 3
    print("  safety  oofEV   oofbust%  devEV   devbust%")
    for i, s in enumerate(g):
        print(f"  {s:.3f}  {ev[i]:.4f}  {bu[i]*100:7.2f}  {ev2[i]:.4f}  {bu2[i]*100:7.2f}")
    print(f"[b08] done ({time.perf_counter()-t0:.0f}s)")
