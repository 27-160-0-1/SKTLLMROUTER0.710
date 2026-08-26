# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b02 step 5 - does the balanced tier want its own constants?

Balanced-only evaluator using exactly the bench2 rules (10-fold OOF over Train,
safety = argmax of the 3-seed x 400-sample bootstrap EV on those rows, dev
scored once at that safety).  Two questions:
  (a) which balanced-specific constant beats the shared one, and by how much;
  (b) how much of that is selection bias -> nested split-half: choose the
      constant on half of the OOF rows, score it on the other half.
"""
from __future__ import annotations
import importlib.util, sys, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b02lib", HERE / "b02-balanced-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, MULTS, DEPLOYED_CFG  # noqa
import protocol as P  # noqa

MULT = MULTS["balanced"]
GRID = np.arange(0.60, 1.101, 0.005)
SEEDS = (7, 17, 23); NBOOT = 400; SIZE = 880

lab = Lab()
cv, arr = L.load_stage("base")
ci = cv["idx"]; di = arr["idx"]
ts = lab.true_s[ci]; tc = lab.true_c[ci]; m = len(ci)
SMP = {s: np.asarray(lab.samples_for(m, s, NBOOT, SIZE)) for s in SEEDS}
ALL = np.concatenate([SMP[s] for s in SEEDS])


def evaluate(cfg, kappa=None, rows=None, grid=GRID):
    """rows: subset of OOF positions used for the bootstrap (None = all)."""
    ps, pc = lab.compose(cv, cfg, "balanced")
    if kappa is not None:
        pc = pc * np.asarray(kappa)[None, :]
    if rows is None:
        smp = ALL
    else:
        r = np.random.default_rng(99)
        smp = r.choice(rows, size=(NBOOT * len(SEEDS), SIZE), replace=True)
    ev, bu, raw = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MULT, grid)
    gi = int(np.argmax(ev))
    return dict(safety=float(grid[gi]), ev=float(ev[gi]), bust=float(bu[gi]),
                raw=float(raw[gi]), ev_curve=ev, grid=grid)


def dev_at(cfg, safety, kappa=None):
    ps, pc = lab.compose(arr, cfg, "balanced")
    if kappa is not None:
        pc = pc * np.asarray(kappa)[None, :]
    pick = lab.allocate(ps, pc, MULT, safety)
    r = np.arange(len(di))
    sc = lab.true_s[di][r, pick].mean()
    ratio = lab.true_c[di][r, pick].sum() / lab.true_c[di][:, 0].sum()
    return dict(score=float(sc), ratio=float(ratio), passed=bool(ratio <= MULT + 1e-15),
                counts=np.bincount(pick, minlength=3).tolist())


t0 = time.perf_counter()
base = evaluate(DEPLOYED_CFG)
bd = dev_at(DEPLOYED_CFG, base["safety"])
print(f"baseline balanced: EV={base['ev']:.6f} @s={base['safety']:.3f} bust={base['bust']*100:.2f}% "
      f"| dev={bd['score']:.6f} r={bd['ratio']:.3f} pass={bd['passed']} L/M/K={bd['counts']} "
      f"({time.perf_counter()-t0:.1f}s/eval)")

SWEEPS = {
    "blend_balanced": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0],
    "gain_alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
    "rank_beta": [0.0, 0.2, 0.4, 0.6, 0.8],
    "fam_w": [0.0, 0.075, 0.15, 0.25, 0.35],
    "conf_scale": [0.0, 0.125, 0.25, 0.4],
    "legacy_w": [0.7, 0.8, 0.9, 1.0],
}
print("\n=== single-constant sweeps (balanced only), honest protocol ===")
print(f"  {'constant':16s} {'value':>7s} {'EV':>9s} {'dEV':>8s} {'safety':>7s} {'bust%':>6s} "
      f"{'dev':>8s} {'ddev':>8s} {'ratio':>7s} {'L/M/K'}")
best_single = {}
for name, vals in SWEEPS.items():
    for v in vals:
        cfg = dict(DEPLOYED_CFG); cfg[name] = v
        r = evaluate(cfg)
        d = dev_at(cfg, r["safety"])
        mark = " <-dep" if abs(v - DEPLOYED_CFG[name]) < 1e-12 else ""
        print(f"  {name:16s} {v:7.3f} {r['ev']:9.6f} {r['ev']-base['ev']:+8.4f} "
              f"{r['safety']:7.3f} {r['bust']*100:6.2f} {d['score']:8.6f} "
              f"{d['score']-bd['score']:+8.4f} {d['ratio']:7.3f} {d['counts']}{mark}")
        if name not in best_single or r["ev"] > best_single[name][1]["ev"]:
            best_single[name] = (v, r, d)
    print()

print("=== kappa (relative price) sweep on balanced ===")
print(f"  {'k1':>5s} {'k2':>5s} {'EV':>9s} {'dEV':>8s} {'safety':>7s} {'bust%':>6s} "
      f"{'dev':>8s} {'ddev':>8s} {'ratio':>7s} {'L/M/K'}")
for k1 in (0.90, 0.95, 1.00, 1.05):
    for k2 in (0.7, 0.85, 1.0, 1.24, 1.5, 2.0):
        r = evaluate(DEPLOYED_CFG, kappa=(1.0, k1, k2))
        d = dev_at(DEPLOYED_CFG, r["safety"], kappa=(1.0, k1, k2))
        print(f"  {k1:5.2f} {k2:5.2f} {r['ev']:9.6f} {r['ev']-base['ev']:+8.4f} "
              f"{r['safety']:7.3f} {r['bust']*100:6.2f} {d['score']:8.6f} "
              f"{d['score']-bd['score']:+8.4f} {d['ratio']:7.3f} {d['counts']}")

json.dump({k: dict(value=v[0], ev=v[1]["ev"], safety=v[1]["safety"], dev=v[2]["score"])
           for k, v in best_single.items()},
          open("reports/lab/b02_best_single.json", "w"), indent=2)
print(f"\n[b02] total {time.perf_counter()-t0:.0f}s")
