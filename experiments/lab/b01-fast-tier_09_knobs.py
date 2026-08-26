# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Fast-tier knobs: the tier blend constant, banning k1, and a monster stress test."""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b01lib", HERE / "b01-fast-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP  # noqa
import bench2 as B
import protocol as P

lab = Lab(); MF = 1.25
cv0, arr0 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
new = np.array([L.classify_v3(t) for t in lab.texts])
GRID = np.arange(0.60, 1.201, 0.005)
SETS = (("TRAIN-OOF", cv0), ("DEV", arr0))


def frontier(ps, pc, rows, cap=1.25):
    ts = lab.true_s[rows]; tc = lab.true_c[rows]; r = np.arange(len(rows))
    best = (-1, None, None)
    for g in GRID:
        pk = P.exact_allocate(ps, pc, MF, float(g))
        rt = tc[r, pk].sum() / tc[:, 0].sum()
        sc = ts[r, pk].mean()
        if rt <= cap + 1e-12 and sc > best[0]:
            best = (sc, rt, g)
    return best


def at_safety(ps, pc, rows, g):
    ts = lab.true_s[rows]; tc = lab.true_c[rows]; r = np.arange(len(rows))
    pk = P.exact_allocate(ps, pc, MF, float(g))
    return ts[r, pk].mean(), tc[r, pk].sum() / tc[:, 0].sum(), np.bincount(pk, minlength=3)


print("=== 1. tier blend constant blend_fast (deployed 0.60) ===")
print(f"{'blend':>6s} | " + " | ".join(f"{n}: front@1.25 (ratio,sfty) | s@.94 | s@.92" for n, _ in SETS))
for b in (0.0, 0.2, 0.3, 0.45, 0.6, 0.75, 1.0):
    cfg = dict(DEPLOYED_CFG, blend_fast=b)
    line = f"{b:6.2f} |"
    for nm, a in SETS:
        ps, pc = lab.compose(a, cfg, "fast")
        sc, rt, g = frontier(ps, pc, a["idx"])
        s94 = at_safety(ps, pc, a["idx"], 0.94); s92 = at_safety(ps, pc, a["idx"], 0.92)
        line += (f" {sc:.4f}(r{rt:.3f},{g:.2f}) | {s94[0]:.4f}/r{s94[1]:.3f} |"
                 f" {s92[0]:.4f}/r{s92[1]:.3f} |")
    print(line, flush=True)

print("\n=== 2. banning k1 in the fast tier ===")
for nm, a in SETS:
    ps, pc = lab.compose(a, DEPLOYED_CFG, "fast")
    ps2 = ps.copy(); ps2[:, 2] = -1e9
    f1 = frontier(ps, pc, a["idx"]); f2 = frontier(ps2, pc, a["idx"])
    n1 = at_safety(ps, pc, a["idx"], 0.94)[2]
    print(f"  {nm:10s} with k1 {f1[0]:.4f} (r{f1[1]:.3f}) counts@.94={n1.tolist()}  "
          f"k1 banned {f2[0]:.4f} (r{f2[1]:.3f})  delta={f2[0]-f1[0]:+.4f}")
    # how often does the ALLOCATOR pick k1 over the whole safety grid?
    npick = max(int((P.exact_allocate(ps, pc, MF, float(g)) == 2).sum()) for g in GRID)
    print(f"             max k1 picks over the whole safety grid: {npick}")

print("\n=== 3. monster stress: inject ONE observed-magnitude runaway per batch ===")
# Train's four largest mid-upgrade footprints, rescaled to an 880 batch: 6.79, 6.59,
# 6.53, 3.88 %L.  The stress asks: if the batch contains one such item AND the
# allocator buys it, does the tier still fit?
ts_all, tc_all = lab.true_s, lab.true_c


def stress_bust(a, cfg, sizes=(0.0, 0.02, 0.04, 0.065), grid=np.arange(0.86, 1.001, 0.01),
                seeds=(7, 17, 23), nboot=300):
    rows = a["idx"]; ps, pc = lab.compose(a, cfg, "fast")
    ts = lab.true_s[rows]; tc = lab.true_c[rows]; m = len(rows)
    out = {}
    for z in sizes:
        bust = np.zeros(len(grid)); sc_m = np.zeros(len(grid))
        for s in seeds:
            smp = np.asarray(lab.samples_for(m, s, nboot, 880))
            TC = tc[smp].copy()
            if z > 0:
                # the cheapest-to-upgrade math item in each resample becomes a runaway
                fam = new[rows][smp]
                mathy = np.isin(fam, ["dmmath", "gsm8k_or_other", "code", "aime"])
                base = TC[:, :, 0].sum(axis=1)
                pen = np.where(mathy, (pc[smp][:, :, 1] - pc[smp][:, :, 0]), 1e18)
                j = np.argmin(pen, axis=1)
                bi = np.arange(len(j))
                TC[bi, j, 1] = TC[bi, j, 0] + z * base
            e, b, r = P.safety_curve(ps[smp], pc[smp], ts[smp], TC, MF, grid)
            bust += b / len(seeds); sc_m += e / len(seeds)
        out[z] = (bust, sc_m)
    return grid, out


for nm, a in SETS:
    grid, out = stress_bust(a, DEPLOYED_CFG)
    print(f"-- {nm}")
    print("   safety   " + "".join(f"{g:7.2f}" for g in grid))
    for z, (bu, ev) in out.items():
        print(f"   bust% z={z:<5} " + "".join(f"{x*100:6.1f} " for x in bu))
    for z, (bu, ev) in out.items():
        print(f"   EV    z={z:<5} " + "".join(f"{x:6.4f} " for x in ev))
