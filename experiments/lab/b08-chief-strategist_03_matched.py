# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b08 - (A) is C1 real at matched risk?  (B) bound the cost-side candidate family.
       (C) resolve b05 P2 vs b06 P2 on the two post-hoc gain constants.
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
from importlib import import_module                               # noqa: E402
S2 = import_module("b08-chief-strategist_02_safety")

OOF_SEEDS = (7, 17, 23)
DEV_SEEDS = (101, 103, 107)


def mk_mult(km, kk, tiers=TIERS):
    def f(lab, arr, ps, pc, tier):
        if tier not in tiers:
            return ps, pc
        pc = pc.copy()
        pc[:, 1] *= km
        pc[:, 2] *= kk
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        return ps, pc
    return f


def run_rule(lab, cv, arr, cfg, label, transform=None, min_margin=0.05, grids=None):
    """Pick safety = argmax OOF EV subject to the OOF POINT margin >= min_margin."""
    grids = grids or B.GRIDS
    safety, oofd, devd = {}, {}, {}
    for t in TIERS:
        o = S2.pool_table(lab, cv, cfg, t, grids[t], OOF_SEEDS, transform=transform)
        d = S2.pool_table(lab, arr, cfg, t, grids[t], DEV_SEEDS, transform=transform)
        m = MULTS[t] / o["ratio"] - 1.0
        ok = m >= min_margin
        if not ok.any():
            ok = m >= m.max() - 1e-12
        gi = int(np.argmax(np.where(ok, o["ev"], -1.0)))
        safety[t] = float(grids[t][gi])
        oofd[t] = {k: float(v[gi]) for k, v in o.items()}
        devd[t] = {k: float(v[gi]) for k, v in d.items()}
    EV = sum(W[t] * oofd[t]["ev"] for t in TIERS)
    devEV = sum(W[t] * devd[t]["ev"] for t in TIERS)
    dev = sum(W[t] * (devd[t]["score"] if devd[t]["ratio"] <= MULTS[t] + 1e-15 else 0.0)
              for t in TIERS)
    j = lambda f: "/".join(f(t) for t in TIERS)
    print(f"{label:40s} EV={EV:.6f} dev={dev:.6f} devpoolEV={devEV:.6f}  "
          f"s={j(lambda t: '%.3f' % safety[t])}  "
          f"devB%={j(lambda t: '%.1f' % (devd[t]['bust'] * 100))}  "
          f"devMarg={j(lambda t: '%+.1f' % ((MULTS[t] / devd[t]['ratio'] - 1) * 100))}", flush=True)
    return dict(EV=EV, dev=dev, devEV=devEV, safety=safety, oof=oofd, devp=devd, label=label)


if __name__ == "__main__":
    t0 = time.perf_counter()
    lab = Lab()
    cvB, arrB = B.stage(lab, None, tag="base")
    cvL, arrL = B.stage(lab, dict(legacy_oof_meta=True), tag="legoof")
    ST = {"base": (cvB, arrB), "C1": (cvL, arrL)}

    print("\n=== A. base vs C1 at a common OOF point-margin floor ===")
    for mm in (0.05, 0.08, 0.10, 0.12, 0.15):
        for tag in ("base", "C1"):
            run_rule(lab, *ST[tag], DEPLOYED_CFG, f"{tag}  margin>={mm:.0%}", min_margin=mm)

    print("\n=== A2. fast tier only: matched realised DEV margin ===")
    print("  cfg    s      oofRatio oofMarg oofEV  oofB%  devRatio devMarg devEV  devB%  devScore")
    for tag in ("base", "C1"):
        cv, arr = ST[tag]
        g = B.GRIDS["fast"]
        o = S2.pool_table(lab, cv, DEPLOYED_CFG, "fast", g, OOF_SEEDS)
        d = S2.pool_table(lab, arr, DEPLOYED_CFG, "fast", g, DEV_SEEDS)
        for gi, s in enumerate(g):
            if gi % 4:
                continue
            print(f"  {tag:5s} {s:.3f}  {o['ratio'][gi]:7.4f} {MULTS['fast']/o['ratio'][gi]-1:+6.1%} "
                  f"{o['ev'][gi]:.4f} {o['bust'][gi]*100:5.1f}  {d['ratio'][gi]:7.4f} "
                  f"{MULTS['fast']/d['ratio'][gi]-1:+6.1%} {d['ev'][gi]:.4f} {d['bust'][gi]*100:5.1f}  "
                  f"{d['score'][gi]:.4f}")

    print("\n=== B. cost re-pricing grid (km on mid, kk on k1), global, C1 stage, margin>=8% ===")
    best = None
    for km in (0.7, 0.85, 1.0, 1.15, 1.3):
        for kk in (1.0, 1.24, 1.5, 2.0):
            r = run_rule(lab, cvL, arrL, DEPLOYED_CFG, f"C1 km={km} kk={kk}",
                         transform=mk_mult(km, kk), min_margin=0.08)
            if best is None or r["EV"] > best["EV"]:
                best = r
    print(f"  best global multiplier: {best['label']} EV={best['EV']:.6f} dev={best['dev']:.6f}")

    print("\n=== B2. per-tier kappa2 on k1 only (b02 B1), C1 stage, margin>=8% ===")
    for tier in TIERS:
        for kk in (1.24, 1.5, 2.0):
            run_rule(lab, cvL, arrL, DEPLOYED_CFG, f"C1 kk={kk} on {tier}",
                     transform=mk_mult(1.0, kk, tiers=(tier,)), min_margin=0.08)

    print("\n=== C. gain_alpha x rank_beta, C1 stage, margin>=8% ===")
    for ga in (0.0, 0.5, 0.7, 1.0):
        for rb in (0.0, 0.4):
            run_rule(lab, cvL, arrL, dict(gain_alpha=ga, rank_beta=rb),
                     f"C1 gain_alpha={ga} rank_beta={rb}", min_margin=0.08)
    print("\n=== C2. same on the base stage ===")
    for ga in (0.0, 0.5, 1.0):
        for rb in (0.0, 0.4):
            run_rule(lab, cvB, arrB, dict(gain_alpha=ga, rank_beta=rb),
                     f"base gain_alpha={ga} rank_beta={rb}", min_margin=0.08)
    print(f"[b08] done ({time.perf_counter()-t0:.0f}s)")
