# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a03: noise ceiling and learning curve for the quantity the allocator uses -
the upgrade GAIN (s_mid - s_light and s_k1 - s_mid).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lab"))

_spec = importlib.util.spec_from_file_location(
    "a03curve", Path(__file__).with_name("a03-train-failure_curve.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)


def noise_var(s, n):
    return s * (1 - s) * n / (n - 1) / n


def main():
    Xtr, Xdv, m = _c.load()
    G, Gd = _c.grams(Xtr, Xdv)
    Y, Ydv, ng, fdv = m["Ytr"], m["Ydv"], m["ngdv"], m["fdv"]
    ntr = len(Y)
    s = Ydv[:, :3]
    print("== noise ceiling for the upgrade gains (dev)")
    for a, b, lab in ((0, 1, "mid - light"), (1, 2, "k1 - mid"), (0, 2, "k1 - light")):
        g = s[:, b] - s[:, a]
        nv = float(np.mean(noise_var(s[:, a], ng[:, a]) + noise_var(s[:, b], ng[:, b])))
        vg = float(g.var())
        print(f"  {lab:12s} var(g)={vg:.5f}  binomial noise={nv:.5f}  "
              f"latent var={max(vg-nv,0):.5f}  ceiling corr={np.sqrt(max(vg-nv,0)/vg):.3f}")
        # within family
        gw = g.copy()
        for f in set(fdv.tolist()):
            k = fdv == f
            gw[k] -= gw[k].mean()
        vw = float(np.mean(gw ** 2))
        print(f"               within-family var(g)={vw:.5f}  => latent {max(vw-nv,0):.5f}"
              f"  ceiling corr={np.sqrt(max(vw-nv,0)/vw):.3f}  "
              f"(between-family share of var {1-vw/vg:.2f})")
    print("\n== learning curve for the GAIN correlation (alpha 10)")
    print("    n     corr(g1) corr(g2) | within-family corr(g1) corr(g2)")
    g1t = s[:, 1] - s[:, 0]
    g2t = s[:, 2] - s[:, 1]

    def dm(v):
        o = v.copy()
        for f in set(fdv.tolist()):
            k = fdv == f
            o[k] -= o[k].mean()
        return o
    g1w, g2w = dm(g1t), dm(g2t)
    for n in (200, 440, 880, 1200, 1760):
        seeds = (0, 1, 2, 3, 4) if n < ntr else (0,)
        v = []
        for sd in seeds:
            rng = np.random.default_rng(1000 + sd)
            idx = rng.permutation(ntr)[:n] if n < ntr else np.arange(ntr)
            p = _c.ridge_dual(G, Gd, Y, idx, 10.0)[0]
            p = np.clip(p[:, :3], 0, 1)
            a1, a2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 1]
            v.append([np.corrcoef(a1, g1t)[0, 1], np.corrcoef(a2, g2t)[0, 1],
                      np.corrcoef(dm(a1), g1w)[0, 1], np.corrcoef(dm(a2), g2w)[0, 1]])
        r = np.mean(v, 0)
        print(f"  {n:5d}    {r[0]:7.3f}  {r[1]:7.3f} | {r[2]:15.3f} {r[3]:9.3f}")
    print("\n== what a family-mean-only predictor achieves on the gains")
    ftr = m["ftr"]
    fam = np.zeros_like(Ydv)
    for f in set(ftr.tolist()):
        fam[fdv == f] = Y[ftr == f].mean(axis=0)
    print(f"   corr(g1) {np.corrcoef(fam[:,1]-fam[:,0], g1t)[0,1]:.3f}   "
          f"corr(g2) {np.corrcoef(fam[:,2]-fam[:,1], g2t)[0,1]:.3f}")
    print("\n== how the score variance splits into the allocator-invariant LEVEL")
    print("   channel (mean over the 3 models) and the 2 GAIN channels")
    d = np.load(ROOT / "reports/lab/dev_preds_e43.npz")
    for nm, v in (("true dev scores", s), ("deployed E43 preds", d["score_fast"])):
        L = v.mean(1, keepdims=True)
        D = v - L
        tot = v.var(0).sum()
        print(f"   {nm:20s} level {300*L.var()/tot:4.0f}%   gain {100*D.var(0).sum()/tot:4.0f}%")
    Lp = d["score_fast"].mean(1)
    print("   corr(deployed LEVEL channel alone, realised score) =",
          np.round([np.corrcoef(Lp, s[:, j])[0, 1] for j in range(3)], 3))

    print("\n== rank agreement across the 8 predictors measured in _gain.py")
    lvl = [0.484, 0.479, 0.426, 0.380, 0.409, 0.442, 0.423, 0.444]
    g1 = [0.056, 0.048, 0.028, 0.041, 0.088, 0.089, 0.089, 0.110]
    e1 = [0.062, 0.057, 0.027, 0.031, 0.103, 0.114, 0.115, 0.130]
    fin = [0.6872, 0.6933, 0.6916, 0.6960, 0.6951, 0.7010, 0.6978, 0.7035]
    print(f"   spearman(level corr, final)      = {spearmanr(lvl, fin).statistic:+.3f}")
    print(f"   spearman(gain corr 1-0, final)   = {spearmanr(g1, fin).statistic:+.3f}")
    print(f"   spearman(eff rank corr 1, final) = {spearmanr(e1, fin).statistic:+.3f}")


if __name__ == "__main__":
    main()
