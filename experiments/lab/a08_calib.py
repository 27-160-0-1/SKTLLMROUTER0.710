# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a08: fold-pure per-family residual calibration of the stacked prediction.

The full-stack test (a08_family.py) shows the meta stack already recovers the
repaired family split from the n-grams (OOF RMSE unchanged).  What it does NOT
do is remove the per-bucket *bias* (deployed dev: light score +-0.08 and light
log-cost +0.41 between the two halves of `gsm8k_or_other`).  This script adds a
fold-pure mean-residual correction per family, applied to the stacked
score/log-cost just before allocation, under both label sets.

Usage: python a08_calib.py [NBOOT] [SEEDS...]
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

import numpy as np

import a08_family as F  # noqa: E402  (loads the cached ridge/legacy/kNN blocks)


def stacked(meta, fm, tier):
    prod = F.CFG["legacy_w"] * F.legacy + (1 - F.CFG["legacy_w"]) * F.linear
    prod = (1 - F.CFG["fam_w"]) * prod + F.CFG["fam_w"] * fm
    conf = np.clip(F.knn_hold[:, 6], 0, 1)[:, None] * F.CFG["conf_scale"]
    prod = (1 - conf) * prod + conf * F.knn_hold[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0, 1)
    m = meta[:, :6].copy()
    recon = np.column_stack([m[:, 0], m[:, 0] + meta[:, 6], m[:, 0] + meta[:, 6] + meta[:, 7]])
    m[:, :3] = (1 - F.CFG["gain_alpha"]) * m[:, :3] + F.CFG["gain_alpha"] * recon
    b = F.CFG[f"blend_{tier}"]
    return (1 - b) * prod + b * m


def calibrate(st, names, which):
    """fold-pure: subtract the fit-fold mean residual of each family bucket."""
    out = st.copy()
    for fold in range(5):
        hold = F.fold_of == fold
        fit = ~hold
        by = defaultdict(list)
        for i in np.where(fit)[0]:
            by[names[i]].append(i)
        for name, rows in by.items():
            if len(rows) < 20:
                continue
            rows = np.asarray(rows)
            bias = (st[rows] - np.hstack([F.true_s, np.log(F.true_c)])[rows]).mean(axis=0)
            k = hold & np.asarray([nm == name for nm in names])
            if not k.any():
                continue
            if which in ("score", "both"):
                out[k, :3] -= bias[:3]
            if which in ("cost", "both"):
                out[k, 3:] -= bias[3:]
    return out


def evaluate(meta, fm, names, which, seed, nboot):
    r = np.random.default_rng(seed)
    samples = [r.integers(0, F.n, size=880) for _ in range(nboot)]
    tot = 0.0
    det = {}
    for tier in ("fast", "balanced", "premium"):
        st = stacked(meta, fm, tier)
        if which != "none":
            st = calibrate(st, names, which)
        ps = np.clip(st[:, :3], 0, 1)
        pc = np.exp(np.clip(st[:, 3:], -50, 50))
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        best = None
        for s in F.GRIDS[tier]:
            evs = []
            for smp in samples:
                p = F.alloc(ps[smp], pc[smp], F.MULTS[tier], float(s))
                ar = np.arange(len(smp))
                ratio = F.true_c[smp][ar, p].sum() / F.true_c[smp][:, 0].sum()
                evs.append(0.0 if ratio > F.MULTS[tier] else F.true_s[smp][ar, p].mean())
            ev = float(np.mean(evs))
            if best is None or ev > best[0]:
                best = (ev, float(s))
        tot += F.WEIGHT[tier] * best[0]
        det[tier] = best
    return tot, det


def main():
    nboot = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seeds = [int(x) for x in sys.argv[2:]] or [7, 17, 23]
    labels = {"rt_dm": [F.classify2(t) for t in F.texts]}
    for lname, names in labels.items():
        t0 = time.perf_counter()
        f1h = F.one_hot(names)
        fm = F.fam_means(names)
        meta = F.run_meta(f1h)
        print(f"[a08] labels={lname} meta done {time.perf_counter()-t0:.0f}s", flush=True)
        for which in ("none", "score", "cost", "both"):
            evs = []
            for sd in seeds:
                ev, det = evaluate(meta, fm, names, which, sd, nboot)
                evs.append(ev)
            print(f"[a08] labels={lname:4s} calib={which:5s} MEAN EV={np.mean(evs):.4f}  "
                  f"seeds={np.round(evs,4)}  " +
                  " ".join(f"{t[:4]}={v[0]:.4f}@{v[1]:.2f}" for t, v in det.items()),
                  flush=True)


if __name__ == "__main__":
    main()
