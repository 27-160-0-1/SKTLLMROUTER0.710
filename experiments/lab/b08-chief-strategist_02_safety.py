# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b08 - why the bench2 safety argmax fails under C1, and a leak-free rule that does not.

For each tier and each safety on the bench2 grid we report, for BOTH the
train-OOF pool (1,760 rows, the only pool the rule may read) and the held-out
dev pool (880 rows, reported as a diagnostic only):

  point ratio    realised budget ratio of the whole pool at that safety
  margin         mult/ratio - 1
  EV, bust       3-seed x 400 x 880-item bootstrap

The point of the table is that the OOF pool's own POINT MARGIN is leak-free and
already discriminates the configurations whose dev-pool bust explodes.
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

OOF_SEEDS = (7, 17, 23)
DEV_SEEDS = (101, 103, 107)


def pool_table(lab, arr, cfg, tier, grid, seeds, nboot=400, transform=None):
    idx = arr["idx"]; ts = lab.true_s[idx]; tc = lab.true_c[idx]; m = len(idx)
    ps, pc = lab.compose(arr, cfg, tier)
    if transform is not None:
        ps, pc = transform(lab, arr, ps, pc, tier)
    ev = np.zeros(len(grid)); bu = np.zeros(len(grid)); raw = np.zeros(len(grid))
    for s in seeds:
        smp = np.asarray(lab.samples_for(m, s, nboot, 880))
        e, b, r = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MULTS[tier], grid)
        ev += e / len(seeds); bu += b / len(seeds); raw += r / len(seeds)
    # point estimate over the whole pool
    pr = np.zeros(len(grid)); sc = np.zeros(len(grid))
    picks = P.exact_allocate(np.repeat(ps[None], len(grid), 0), np.repeat(pc[None], len(grid), 0),
                             MULTS[tier], 1.0) if False else None
    for gi, g in enumerate(grid):
        pick = lab.allocate(ps, pc, MULTS[tier], float(g))
        r = np.arange(m)
        pr[gi] = tc[r, pick].sum() / tc[:, 0].sum()
        sc[gi] = ts[r, pick].mean()
    return dict(ev=ev, bust=bu, raw=raw, ratio=pr, score=sc)


def pick_rule(oof, grid, mult, rule, min_margin=0.05, max_bust=0.005):
    m = mult / oof["ratio"] - 1.0
    if rule == "argmax":
        ok = np.ones(len(grid), bool)
    elif rule == "margin":
        ok = m >= min_margin
    elif rule == "margin+bust":
        ok = (m >= min_margin) & (oof["bust"] <= max_bust)
    else:
        raise ValueError(rule)
    if not ok.any():
        ok = m >= m.max() - 1e-12
    ev = np.where(ok, oof["ev"], -1.0)
    return int(np.argmax(ev))


def evaluate(lab, cv, arr, cfg, grids, rule, label, **kw):
    safety, oofd, devd = {}, {}, {}
    for t in TIERS:
        o = pool_table(lab, cv, cfg, t, grids[t], OOF_SEEDS, **kw)
        d = pool_table(lab, arr, cfg, t, grids[t], DEV_SEEDS, **kw)
        gi = pick_rule(o, grids[t], MULTS[t], rule)
        safety[t] = float(grids[t][gi])
        oofd[t] = {k: float(v[gi]) for k, v in o.items()}
        devd[t] = {k: float(v[gi]) for k, v in d.items()}
        oofd[t]["margin"] = MULTS[t] / oofd[t]["ratio"] - 1.0
        devd[t]["margin"] = MULTS[t] / devd[t]["ratio"] - 1.0
    EV = sum(W[t] * oofd[t]["ev"] for t in TIERS)
    devEV = sum(W[t] * devd[t]["ev"] for t in TIERS)
    dev = sum(W[t] * (devd[t]["score"] if devd[t]["ratio"] <= MULTS[t] + 1e-15 else 0.0)
              for t in TIERS)
    j = lambda f: "/".join(f(t) for t in TIERS)
    print(f"{label:38s} EV={EV:.6f} dev={dev:.6f} devpoolEV={devEV:.6f}")
    print("   safety=" + j(lambda t: "%.3f" % safety[t])
          + "  oofbust%=" + j(lambda t: "%.1f" % (oofd[t]["bust"] * 100))
          + "  devbust%=" + j(lambda t: "%.1f" % (devd[t]["bust"] * 100)))
    print("   oofmargin=" + j(lambda t: "%+.1f%%" % (oofd[t]["margin"] * 100))
          + "  devmargin=" + j(lambda t: "%+.1f%%" % (devd[t]["margin"] * 100)), flush=True)
    return dict(label=label, EV=EV, dev=dev, devEV=devEV, safety=safety, oof=oofd, devp=devd)


if __name__ == "__main__":
    t0 = time.perf_counter()
    lab = Lab()
    cvB, arrB = B.stage(lab, None, tag="base")
    cvL, arrL = B.stage(lab, dict(legacy_oof_meta=True), tag="legoof")
    G = B.GRIDS

    print("\n=== per-tier ladders, C1 stage ===")
    for t in TIERS:
        o = pool_table(lab, cvL, DEPLOYED_CFG, t, G[t], OOF_SEEDS)
        d = pool_table(lab, arrL, DEPLOYED_CFG, t, G[t], DEV_SEEDS)
        print(f"-- {t} (cap {MULTS[t]}) --")
        print("  s      oofRatio oofMarg  oofEV   oofB%   devRatio devMarg  devEV   devB%   devScore")
        for gi, g in enumerate(G[t]):
            if gi % 4:
                continue
            print(f"  {g:.3f}  {o['ratio'][gi]:7.4f} {MULTS[t]/o['ratio'][gi]-1:+6.1%} "
                  f"{o['ev'][gi]:.4f} {o['bust'][gi]*100:6.2f}  {d['ratio'][gi]:7.4f} "
                  f"{MULTS[t]/d['ratio'][gi]-1:+6.1%} {d['ev'][gi]:.4f} {d['bust'][gi]*100:6.2f}  "
                  f"{d['score'][gi]:.4f}")

    print("\n=== safety rules ===")
    res = {}
    for tag, cv, arr in (("base", cvB, arrB), ("C1", cvL, arrL)):
        for rule in ("argmax", "margin", "margin+bust"):
            res[(tag, rule)] = evaluate(lab, cv, arr, DEPLOYED_CFG, G, rule, f"{tag} / {rule}")
    print(f"[b08] done ({time.perf_counter()-t0:.0f}s)")
