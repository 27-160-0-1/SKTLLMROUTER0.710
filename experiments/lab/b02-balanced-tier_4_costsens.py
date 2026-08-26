# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b02 step 4 - how much is cost prediction worth, per tier, under the honest
protocol (bench2: OOF safety choice, dev scored once)?

Each row replaces part of the predicted cost vector by the truth (or degrades
it) for ONE tier at a time, so the per-tier value is isolated.  EV is the
3-seed x 400-sample bootstrap expectation with the bust priced in; dev is the
single held-out sample at the safety the bootstrap chose.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b02lib", HERE / "b02-balanced-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG  # noqa
import bench2 as B  # noqa

lab = Lab()
cv, arr = L.load_stage("base")
eb = np.load("reports/lab/b02_eb.npz")["eb"]
rng = np.random.default_rng(4242)


def mk(cols=(), tier_only=None, score_eb=False, fam_cost=False, noise=0.0, kappa=None):
    def tf(_lab, a, ps, pc, tier):
        if tier_only is not None and tier != tier_only:
            return ps, pc
        idx = a["idx"]
        pc = pc.copy(); ps = ps.copy()
        if cols:
            tcv = _lab.true_c[idx]
            for c in cols:
                pc[:, c] = tcv[:, c]
        if fam_cost:
            # keep the light cost, replace the mid/k1 multipliers by the family mean
            fam = _lab.fam_arr[idx]
            tr = _lab.train_idx
            for f in np.unique(fam):
                rows = np.where(fam == f)[0]
                trf = tr[_lab.fam_arr[tr] == f]
                if len(trf) < 8:
                    trf = tr
                m1 = _lab.true_c[trf, 1].sum() / _lab.true_c[trf, 0].sum()
                m2 = _lab.true_c[trf, 2].sum() / _lab.true_c[trf, 0].sum()
                pc[rows, 1] = pc[rows, 0] * m1
                pc[rows, 2] = pc[rows, 0] * m2
        if noise:
            pc = pc * np.exp(rng.normal(0.0, noise, size=pc.shape))
        if kappa is not None:
            pc = pc * np.asarray(kappa)[None, :]
        if score_eb:
            ps = eb[idx]
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        return np.clip(ps, 0, 1), pc
    return tf


base = B.run(lab, cv, arr, DEPLOYED_CFG, label="baseline (deployed cfg)")
print()

ROWS = [
    ("true cost ALL cols", dict(cols=(0, 1, 2))),
    ("true cost light col", dict(cols=(0,))),
    ("true cost mid col", dict(cols=(1,))),
    ("true cost k1 col", dict(cols=(2,))),
    ("true cost mid+k1", dict(cols=(1, 2))),
    ("family-mean cost mult", dict(fam_cost=True)),
    ("EB score oracle", dict(score_eb=True)),
    ("EB score + true cost", dict(score_eb=True, cols=(0, 1, 2))),
    ("cost noise +0.20 log", dict(noise=0.20)),
    ("cost noise +0.40 log", dict(noise=0.40)),
]

for tier in TIERS:
    print(f"--- perturbation applied to {tier.upper()} only ---")
    for name, kw in ROWS:
        r = B.run(lab, cv, arr, DEPLOYED_CFG, transform=mk(tier_only=tier, **kw),
                  label=f"[{tier[:4]}] {name}")
        dt = r["dev_tiers"][tier]; dd = r["det"][tier]
        b0 = base["dev_tiers"][tier]; e0 = base["det"][tier]
        print(f"      -> tier EV {e0['ev']:.4f} -> {dd['ev']:.4f} ({dd['ev']-e0['ev']:+.4f}) | "
              f"tier dev {b0['score']:.4f} -> {dt['score']:.4f} ({dt['score']-b0['score']:+.4f}) | "
              f"safety {base['safety'][tier]:.3f} -> {r['safety'][tier]:.3f}")
    print()
