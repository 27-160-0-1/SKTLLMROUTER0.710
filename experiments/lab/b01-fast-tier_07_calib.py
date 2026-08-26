# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""How risky is the fast tier really?  Cross-batch calibration of the safety choice.

The project chooses the safety ratio by bootstrap-resampling the rows it already
has.  That measures within-sample uncertainty only.  Here the whole procedure is
tested out-of-batch, using train rows only:

    split the 1,760 train-OOF rows into two disjoint 880-item batches A and B
    choose safety on A exactly as bench2 does (3 seeds x 400 bootstrap resamples)
    apply it to B once and record score / ratio / bust

Repeated over many random splits this is an honest estimate of the *procedure's*
bust probability on a fresh 880-item batch - the quantity the competition pays.
"""
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
GRID = np.arange(0.80, 1.041, 0.005)

# ---------------------------------------------------------------- A. the tail
print("=== A. per-item mid-upgrade footprint (true extra cost / light total of the batch) ===")
for nm, rows in (("train", lab.train_idx), ("dev", lab.dev_idx)):
    tc = lab.true_c[rows]
    e = (tc[:, 1] - tc[:, 0]) / tc[:, 0].sum() * (len(rows) / 880.0)   # rescaled to an 880 batch
    q = np.percentile(e, [50, 90, 99, 99.9])
    print(f"  {nm:6s} n={len(rows):5d} median={q[0]*100:.4f}%L p90={q[1]*100:.4f}%L "
          f"p99={q[2]*100:.3f}%L p99.9={q[3]*100:.3f}%L max={e.max()*100:.3f}%L "
          f"sum={e.sum():.3f}L  (fast headroom = 0.25L)")
    o = np.argsort(-e)[:5]
    print(f"         top5 %L: " + " ".join(f"{e[i]*100:.2f}(ep{rows[i]},{new[rows[i]]})" for i in o))

print("\n=== B. cross-batch calibration of the safety choice (train rows only) ===")


def choose_safety_on(ps, pc, rows, seeds=(7, 17, 23), nboot=400, grid=GRID):
    ts = lab.true_s[rows]; tc = lab.true_c[rows]; m = len(rows)
    ev = np.zeros(len(grid))
    for s in seeds:
        smp = np.asarray(lab.samples_for(m, s, nboot, 880))
        e, b, r = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MF, grid)
        ev += e / len(seeds)
    return float(grid[int(np.argmax(ev))])


def apply_on(ps, pc, rows, sf):
    pk = P.exact_allocate(ps, pc, MF, sf)
    r = np.arange(len(rows))
    tc = lab.true_c[rows]; ts = lab.true_s[rows]
    ratio = tc[r, pk].sum() / tc[:, 0].sum()
    sc = ts[r, pk].mean()
    return sc, ratio, ratio > MF + 1e-15


def crossbatch(ps_all, pc_all, nsplit=60, fixed=None):
    idx = np.arange(len(cv0["idx"]))
    out = []
    for k in range(nsplit):
        rng = np.random.default_rng(1000 + k)
        pm = rng.permutation(idx)
        A, Bx = pm[:880], pm[880:]
        rowsA = cv0["idx"][A]; rowsB = cv0["idx"][Bx]
        sf = fixed if fixed is not None else choose_safety_on(ps_all[A], pc_all[A], rowsA,
                                                              nboot=200)
        sc, ratio, bust = apply_on(ps_all[Bx], pc_all[Bx], rowsB, sf)
        out.append((sf, sc, ratio, bust))
    o = np.array(out)
    return dict(sf=o[:, 0].mean(), sf_sd=o[:, 0].std(), score=o[:, 1].mean(),
                ratio=o[:, 2].mean(), ratio_sd=o[:, 2].std(), ratio_max=o[:, 2].max(),
                bust=o[:, 3].mean(), ev=float(np.mean(np.where(o[:, 3] > 0, 0.0, o[:, 1]))))


ps_cv, pc_cv = lab.compose(cv0, DEPLOYED_CFG, "fast")
ps_dv, pc_dv = lab.compose(arr0, DEPLOYED_CFG, "fast")
cvm, dvm = L.fam_matrix(lab, new, cv0, arr0)
POL = {"item-level": (ps_cv, ps_dv),
       "family-only": (np.clip(cvm[:, :3], 0, 1), np.clip(dvm[:, :3], 0, 1)),
       "shrink w=.5": (0.5 * ps_cv + 0.5 * np.clip(cvm[:, :3], 0, 1),
                       0.5 * ps_dv + 0.5 * np.clip(dvm[:, :3], 0, 1))}

print(f"{'policy':14s} {'mode':16s} {'safety':>7s} {'B score':>8s} {'B ratio':>8s} "
      f"{'sd':>6s} {'max':>7s} {'bust%':>6s} {'EV':>8s}")
for nm, (a, _b) in POL.items():
    r = crossbatch(a, pc_cv)
    print(f"{nm:14s} {'bootstrap-chosen':16s} {r['sf']:7.3f} {r['score']:8.4f} "
          f"{r['ratio']:8.4f} {r['ratio_sd']:6.4f} {r['ratio_max']:7.4f} "
          f"{r['bust']*100:6.1f} {r['ev']:8.4f}", flush=True)
for nm, (a, _b) in POL.items():
    for f in (0.98, 0.96, 0.94, 0.92, 0.90, 0.88):
        r = crossbatch(a, pc_cv, fixed=f)
        print(f"{nm:14s} {'fixed '+str(f):16s} {r['sf']:7.3f} {r['score']:8.4f} "
              f"{r['ratio']:8.4f} {r['ratio_sd']:6.4f} {r['ratio_max']:7.4f} "
              f"{r['bust']*100:6.1f} {r['ev']:8.4f}", flush=True)
