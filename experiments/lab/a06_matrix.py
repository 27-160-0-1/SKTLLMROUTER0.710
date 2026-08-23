# SPDX-License-Identifier: Apache-2.0
"""a06 step 4: structure of the 3-column score matrix, per family and pooled.

Monotonicity, inversions, cross-model correlation, conditional probabilities,
and the same quantities on binarised labels (s>0 vs s==0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a06_features import build  # noqa: E402


def block(nm):
    d = build(nm)
    sp, fam = d["split"], d["fam"]
    s = sp.score
    fams = sorted(set(fam)) + ["ALL"]
    print(f"\n########## {nm} ##########")
    print("A. monotone pattern of the realised triple (ties allowed)")
    print(f"{'family':16s} {'n':>4s} {'mono<=':>7s} {'strict<':>8s} {'allequal':>9s} "
          f"{'k1<mid':>7s} {'k1<light':>9s} {'mid<light':>10s} {'lightbest':>10s}")
    for f in fams:
        m = np.ones(len(sp), bool) if f == "ALL" else (fam == f)
        a, b, c = s[m, 0], s[m, 1], s[m, 2]
        print(f"{f:16s} {m.sum():4d} {np.mean((a<=b)&(b<=c)):7.3f} {np.mean((a<b)&(b<c)):8.3f} "
              f"{np.mean((a==b)&(b==c)):9.3f} {np.mean(c<b):7.3f} {np.mean(c<a):9.3f} "
              f"{np.mean(b<a):10.3f} {np.mean((a>=b)&(a>=c)):10.3f}")

    print("\nB. cross-model Pearson corr of realised scores, and of the LATENT part")
    print("   (latent corr = obs corr / sqrt(rel_i * rel_j), rel = var_p/var_obs)")
    print(f"{'family':16s} {'r(l,m)':>7s} {'r(l,k)':>7s} {'r(m,k)':>7s} | "
          f"{'lat(l,m)':>9s} {'lat(l,k)':>9s} {'lat(m,k)':>9s}")
    for f in fams:
        m = np.ones(len(sp), bool) if f == "ALL" else (fam == f)
        rel = []
        for j in range(3):
            y = s[m, j]
            g = sp.ngen[m, j].mean()
            mu, v = y.mean(), y.var()
            vp = (v - (mu - mu * mu) / g) / (1 - 1 / g) if g > 1 else v
            rel.append(max(vp, 0) / max(v, 1e-12))
        rs, ls = [], []
        for (i, j) in ((0, 1), (0, 2), (1, 2)):
            if np.std(s[m, i]) < 1e-9 or np.std(s[m, j]) < 1e-9:
                rs.append(np.nan); ls.append(np.nan); continue
            r = np.corrcoef(s[m, i], s[m, j])[0, 1]
            rs.append(r)
            den = np.sqrt(rel[i] * rel[j])
            ls.append(r / den if den > 1e-6 else np.nan)
        print(f"{f:16s} " + " ".join(f"{x:7.3f}" for x in rs) + " | " +
              " ".join(f"{x:9.3f}" for x in ls))

    print("\nC. conditional probabilities on the binarised labels (b_j = s_j > 0)")
    print(f"{'family':16s} {'P(k1|~l)':>9s} {'P(k1|l)':>8s} {'P(m|~l)':>8s} "
          f"{'P(~k1|l)':>9s} {'P(~l)':>6s} {'lift_k1':>8s}")
    for f in fams:
        m = np.ones(len(sp), bool) if f == "ALL" else (fam == f)
        b = s[m] > 0
        nl = ~b[:, 0]
        def pr(num, den):
            return float(num.sum() / den.sum()) if den.sum() else np.nan
        pk_nl = pr(b[:, 2] & nl, nl)
        pk_l = pr(b[:, 2] & b[:, 0], b[:, 0])
        pm_nl = pr(b[:, 1] & nl, nl)
        pnk_l = pr((~b[:, 2]) & b[:, 0], b[:, 0])
        print(f"{f:16s} {pk_nl:9.3f} {pk_l:8.3f} {pm_nl:8.3f} {pnk_l:9.3f} "
              f"{nl.mean():6.3f} {pk_nl - b[:,2].mean():8.3f}")

    print("\nD. how often does the cheaper model strictly win?  (upgrade is WASTED)")
    lightsum = sp.cost[:, 0].sum()
    for f in fams:
        m = np.ones(len(sp), bool) if f == "ALL" else (fam == f)
        a, b, c = s[m, 0], s[m, 1], s[m, 2]
        wasted_k1 = np.mean(c <= b)
        wasted_mid = np.mean(b <= a)
        # cost of the wasted upgrades, as a share of the whole light budget
        cw_k1 = sp.cost[m][c <= b, 2].sum() / lightsum
        cw_mid = sp.cost[m][b <= a, 1].sum() / lightsum
        print(f"{f:16s} k1<=mid {wasted_k1:.3f} (that k1 spend = {cw_k1:6.3f} light-budgets) | "
              f"mid<=light {wasted_mid:.3f} ({cw_mid:6.3f})")


for nm in ("train", "dev"):
    block(nm)
