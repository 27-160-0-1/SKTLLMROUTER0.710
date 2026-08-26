# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b02 step 9 - which cost column unlocks k1 concentration, the value of the k1
option, and the shipped balanced policy with its safety curve.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b02lib", HERE / "b02-balanced-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, MULTS, TIERS, W, DEPLOYED_CFG, DEPLOYED_SAFETY  # noqa
import protocol as P  # noqa

MULT = MULTS["balanced"]
lab = Lab()
cv, arr = L.load_stage("base")
ci = cv["idx"]; di = arr["idx"]; m = len(ci)
eb = np.load("reports/lab/b02_eb.npz")["eb"]
GRID = np.arange(0.60, 1.301, 0.005)
SMP = np.concatenate([np.asarray(lab.samples_for(m, s, 400, 880)) for s in (7, 17, 23)])
ps_c, pc_c = lab.compose(cv, DEPLOYED_CFG, "balanced")
ps_d, pc_d = lab.compose(arr, DEPLOYED_CFG, "balanced")
tc_c = lab.true_c[ci]; tc_d = lab.true_c[di]


def evd(ps, pc, grid=GRID):
    ev, bu, raw = P.safety_curve(ps[SMP], pc[SMP], lab.true_s[ci][SMP], tc_c[SMP], MULT, grid)
    gi = int(np.argmax(ev))
    return float(ev[gi]), float(grid[gi]), float(bu[gi])


def dev(ps, pc, s):
    p = lab.allocate(ps, pc, MULT, s); r = np.arange(len(di))
    return (float(lab.true_s[di][r, p].mean()),
            float(tc_d[r, p].sum() / tc_d[:, 0].sum()), np.bincount(p, minlength=3).tolist())


print("=== (a) which cost column changes the k1 appetite at balanced? ===")
variants = {
    "pred cost (deployed)": (pc_c, pc_d),
    "true k1 col only": (np.column_stack([pc_c[:, 0], pc_c[:, 1], tc_c[:, 2]]),
                         np.column_stack([pc_d[:, 0], pc_d[:, 1], tc_d[:, 2]])),
    "true mid col only": (np.column_stack([pc_c[:, 0], tc_c[:, 1], pc_c[:, 2]]),
                          np.column_stack([pc_d[:, 0], tc_d[:, 1], pc_d[:, 2]])),
    "true light col only": (np.column_stack([tc_c[:, 0], pc_c[:, 1], pc_c[:, 2]]),
                            np.column_stack([tc_d[:, 0], pc_d[:, 1], pc_d[:, 2]])),
    "true cost (all)": (tc_c, tc_d),
}
e0 = None
for name, (a, b) in variants.items():
    a = a.copy(); b = b.copy()
    a[:, 1] = np.maximum(a[:, 1], a[:, 0] * 1.000001); a[:, 2] = np.maximum(a[:, 2], a[:, 1] * 1.000001)
    b[:, 1] = np.maximum(b[:, 1], b[:, 0] * 1.000001); b[:, 2] = np.maximum(b[:, 2], b[:, 1] * 1.000001)
    e, s, bu = evd(ps_c, a)
    d = dev(ps_d, b, s)
    if e0 is None:
        e0 = e
    print(f"  {name:22s} EV={e:.6f} ({e-e0:+.4f}) @s{s:.3f} bust={bu*100:5.2f}% "
          f"dev={d[0]:.6f} r={d[1]:.3f} L/M/K={d[2]}")

print("\n=== (b) value of the k1 option at balanced, at every safety ===")
nk_c = ps_c.copy(); nk_c[:, 2] = nk_c[:, 1]
nk_d = ps_d.copy(); nk_d[:, 2] = nk_d[:, 1]
e1, b1, r1 = P.safety_curve(ps_c[SMP], pc_c[SMP], lab.true_s[ci][SMP], tc_c[SMP], MULT, GRID)
e2, b2, r2 = P.safety_curve(nk_c[SMP], pc_c[SMP], lab.true_s[ci][SMP], tc_c[SMP], MULT, GRID)
print(f"  {'safety':>7s} {'EV k1 on':>10s} {'EV k1 off':>10s} {'delta':>8s} "
      f"{'bust on':>8s} {'bust off':>8s}")
for s in (0.78, 0.82, 0.84, 0.87, 0.89, 0.92, 0.95, 1.0):
    i = int(np.argmin(np.abs(GRID - s)))
    print(f"  {GRID[i]:7.3f} {e1[i]:10.6f} {e2[i]:10.6f} {e1[i]-e2[i]:+8.4f} "
          f"{b1[i]*100:8.2f} {b2[i]*100:8.2f}")

print("\n=== (c) shipped candidate: balanced kappa2, safety curve on Train-OOF ===")
for k2 in (1.0, 1.35, 1.5):
    pc = pc_c.copy(); pc[:, 2] *= k2
    pd = pc_d.copy(); pd[:, 2] *= k2
    ev, bu, raw = P.safety_curve(ps_c[SMP], pc[SMP], lab.true_s[ci][SMP], tc_c[SMP], MULT, GRID)
    gi = int(np.argmax(ev))
    print(f"  k2={k2}: EV*={ev[gi]:.6f} @s={GRID[gi]:.3f} bust={bu[gi]*100:.2f}%")
    print("    " + " ".join(f"s{GRID[i]:.2f}:{ev[i]:.4f}/{bu[i]*100:.1f}%"
                            for i in range(0, len(GRID), 10)))
    for s in (0.84, 0.87, 0.89, 0.90, 0.92):
        i = int(np.argmin(np.abs(GRID - s)))
        d = dev(ps_d, pd, float(GRID[i]))
        print(f"    @s={GRID[i]:.3f}: EV={ev[i]:.6f} bust={bu[i]*100:.2f}% | "
              f"dev={d[0]:.6f} r={d[1]:.3f} pass={d[1]<=MULT} L/M/K={d[2]}")

print("\n=== (d) dev-side bootstrap of the shipped candidate (880 resamples of dev) ===")
md = len(di)
SD = np.concatenate([np.asarray(lab.samples_for(md, s, 400, 880)) for s in (7, 17, 23)])
for k2, s in ((1.0, 0.840), (1.0, 0.870), (1.5, 0.890), (1.35, 0.885), (1.5, 0.870)):
    pd = pc_d.copy(); pd[:, 2] *= k2
    ev, bu, raw = P.safety_curve(ps_d[SD], pd[SD], lab.true_s[di][SD], tc_d[SD], MULT,
                                 np.array([s]))
    d = dev(ps_d, pd, s)
    print(f"  k2={k2:4.2f} s={s:.3f}: devEV={ev[0]:.6f} bust={bu[0]*100:.2f}% raw={raw[0]:.6f} "
          f"| dev point={d[0]:.6f} r={d[1]:.3f} L/M/K={d[2]}")
