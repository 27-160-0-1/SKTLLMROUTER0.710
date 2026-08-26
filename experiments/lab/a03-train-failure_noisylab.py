# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a03: how many NOISY labels are worth one clean label?

E41 produced 5,212 self-labelled rows whose per-item score correlation with the
official labels was 0.726 and gained nothing.  This measures the exchange rate
directly: corrupt the train score labels down to a target correlation, refit,
and read off the equivalent clean-sample size from the clean learning curve.

Corruption model: with probability (1-a) replace the label by an independent
draw from the same family's empirical label distribution (this reproduces the
observed structure: the wrong labels are family-plausible, not uniform noise).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lab"))

_spec = importlib.util.spec_from_file_location(
    "a03curve", Path(__file__).with_name("a03-train-failure_curve.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)


def corrupt(Y, fam, a, rng):
    out = Y.copy()
    for f in set(fam.tolist()):
        k = np.where(fam == f)[0]
        for j in range(3):
            pool = Y[k, j]
            swap = rng.random(len(k)) > a
            out[k[swap], j] = rng.choice(pool, size=int(swap.sum()))
    return out


def main():
    Xtr, Xdv, m = _c.load()
    G, Gd = _c.grams(Xtr, Xdv)
    Y, Ydv, ftr = m["Ytr"], m["Ydv"], m["ftr"]
    ntr = len(Y)
    idx_all = np.arange(ntr)
    print("== clean reference curve (alpha 10, score columns only)")
    ref = {}
    for n in (200, 300, 440, 600, 880, 1200, 1500, 1760):
        v = []
        for sd in (0, 1, 2, 3, 4):
            rng = np.random.default_rng(1000 + sd)
            idx = rng.permutation(ntr)[:n] if n < ntr else idx_all
            p = _c.ridge_dual(G, Gd, Y, idx, 10.0)[0]
            v.append([np.corrcoef(p[:, j], Ydv[:, j])[0, 1] for j in range(3)])
        ref[n] = np.mean(v, 0)
        print(f"  n={n:5d}  {ref[n][0]:.4f} {ref[n][1]:.4f} {ref[n][2]:.4f} "
              f"(mean {ref[n].mean():.4f})")
    print("\n== full 1,760 rows with label corruption")
    print("  keep-prob a  corr(label,truth) l/m/k1 | dev corr l/m/k1        equivalent clean n")
    ns = np.array(sorted(ref))
    means = np.array([ref[n].mean() for n in ns])
    for a in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.5):
        cl, dv_ = [], []
        for sd in (0, 1, 2):
            rng = np.random.default_rng(50 + sd)
            Yc = corrupt(Y, ftr, a, rng)
            cl.append([np.corrcoef(Yc[:, j], Y[:, j])[0, 1] for j in range(3)])
            p = _c.ridge_dual(G, Gd, np.column_stack([Yc[:, :3], Y[:, 3:]]), idx_all, 10.0)[0]
            dv_.append([np.corrcoef(p[:, j], Ydv[:, j])[0, 1] for j in range(3)])
        c, d = np.mean(cl, 0), np.mean(dv_, 0)
        eq = np.interp(d.mean(), means, ns)
        print(f"  a={a:4.2f}       {c[0]:.3f}/{c[1]:.3f}/{c[2]:.3f}       | "
              f"{d[0]:.3f}/{d[1]:.3f}/{d[2]:.3f}   {eq:8.0f}"
              f"  ({eq/ntr*100:.0f}% of clean)")


if __name__ == "__main__":
    main()
