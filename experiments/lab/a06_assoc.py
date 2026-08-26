# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a06 step 1: association of interpretable prompt features with targets.

Pooled and within-family Spearman + mutual information, plus a variance
decomposition that separates between-family from within-family variance and
splits the within-family part into binomial label noise vs latent signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a06_features import build  # noqa: E402

np.set_printoptions(suppress=True)

SCORE_T = ["s_light", "s_mid", "s_k1", "g_mid_light", "g_k1_mid"]
COST_T = ["logc_light", "logc_k1", "log_otok_k1"]
ALL_T = SCORE_T + COST_T + ["eff_k1"]


def rho(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(spearmanr(a, b).statistic)


def main():
    tr = build("train")
    dv = build("dev")
    X, names, fam, T = tr["X"], tr["names"], tr["fam"], tr["targets"]
    Xf = np.hstack([X, tr["extra"]])
    namesf = names + tr["extra_names"]
    fams = sorted(set(fam))
    sp = tr["split"]

    print("=" * 100)
    print("A. POOLED rank correlation |rho| (train, n=1760) -- top 12 features per target")
    print("=" * 100)
    R = np.zeros((len(namesf), len(ALL_T)))
    for j, t in enumerate(ALL_T):
        y = T[t]
        for i in range(Xf.shape[1]):
            R[i, j] = rho(Xf[:, i], y)
    for j, t in enumerate(ALL_T):
        order = np.argsort(-np.abs(R[:, j]))[:12]
        print(f"\n {t}:")
        print("   " + "  ".join(f"{namesf[i]}={R[i,j]:+.3f}" for i in order))

    print("\n" + "=" * 100)
    print("B. Mutual information (pooled, train) -- top 10, nats; family one-hot MI for scale")
    print("=" * 100)
    fam_code = np.array([fams.index(f) for f in fam], dtype=float)
    Xmi = np.hstack([Xf, fam_code[:, None]])
    nm_mi = namesf + ["FAMILY(id)"]
    for t in ALL_T:
        mi = mutual_info_regression(Xmi, T[t], discrete_features=[False] * Xf.shape[1] + [True],
                                    random_state=0)
        order = np.argsort(-mi)[:10]
        fam_mi = mi[-1]
        print(f" {t:14s} FAMILY={fam_mi:.3f} | " +
              " ".join(f"{nm_mi[i]}={mi[i]:.3f}" for i in order if i != len(nm_mi) - 1)[:150])

    print("\n" + "=" * 100)
    print("C. Variance decomposition of each target (train): eta^2 = between-family share")
    print("=" * 100)
    print(f" {'target':14s} {'var_total':>10s} {'eta2_fam':>9s} {'var_within':>11s}")
    for t in ALL_T:
        y = T[t]
        vt = y.var()
        mu = np.array([y[fam == f].mean() for f in fams])
        w = np.array([(fam == f).sum() for f in fams]) / len(y)
        vb = (w * (mu - y.mean()) ** 2).sum()
        print(f" {t:14s} {vt:10.4f} {vb/max(vt,1e-12):9.3f} {vt-vb:11.4f}")

    print("\n" + "=" * 100)
    print("D. Within-family LATENT signal in the score labels (train)")
    print("   var(obs) = var(p) + E[p(1-p)]/ngen ; corr_max = sqrt(var_p/var_obs)")
    print("=" * 100)
    hdr = f" {'family':16s} {'n':>4s} {'ngen':>4s}"
    for m in ("light", "mid", "k1"):
        hdr += f" | {m:>5s} mean  sd   var_p  cmax"
    print(hdr)
    latent = {}
    for f in fams:
        m = fam == f
        n = int(m.sum())
        row = f" {f:16s} {n:4d} {sp.ngen[m,0].mean():4.1f}"
        for j, mn in enumerate(("light", "mid", "k1")):
            y = sp.score[m, j]
            g = sp.ngen[m, j].mean()
            mu, v = y.mean(), y.var()
            vp = (v - (mu - mu * mu) / g) / (1.0 - 1.0 / g) if g > 1 else v
            vp = max(vp, 0.0)
            cmax = math_sqrt(vp / v) if v > 1e-12 else 0.0
            latent[(f, mn)] = (mu, v, vp, cmax)
            row += f" | {mu:5.3f} {np.sqrt(v):5.3f} {vp:6.4f} {cmax:5.2f}"
        print(row)

    print("\n" + "=" * 100)
    print("E. Within-family max |rho| over the 35 prompt features (train) vs permutation null")
    print("   null = 95th pct of max|rho| over 35 random permutations of y (200 draws)")
    print("=" * 100)
    rng = np.random.default_rng(0)
    for t in SCORE_T + ["log_otok_k1", "logc_k1"]:
        print(f"\n target {t}")
        print(f" {'family':16s} {'n':>4s} {'maxrho':>7s} {'null95':>7s}  best features")
        for f in fams:
            m = fam == f
            n = int(m.sum())
            y = T[t][m]
            if np.std(y) < 1e-9:
                print(f" {f:16s} {n:4d}   ---     ---   (target constant)")
                continue
            r = np.array([rho(X[m, i], y) for i in range(X.shape[1])])
            # permutation null
            nulls = []
            for _ in range(200):
                yp = rng.permutation(y)
                nulls.append(max(abs(rho(X[m, i], yp)) for i in range(0, X.shape[1], 3)))
            null95 = float(np.percentile(nulls, 95))
            order = np.argsort(-np.abs(r))[:3]
            best = " ".join(f"{names[i]}={r[i]:+.3f}" for i in order)
            flag = "*" if np.abs(r).max() > null95 else " "
            print(f" {f:16s} {n:4d} {np.abs(r).max():7.3f} {null95:7.3f}{flag} {best}")

    print("\nDEV replication of the strongest within-family score associations")
    print(" (feature chosen on train, rho recomputed on dev)")
    for t in SCORE_T:
        for f in fams:
            m = fam == f
            y = T[t][m]
            if np.std(y) < 1e-9:
                continue
            r = np.array([rho(X[m, i], y) for i in range(X.shape[1])])
            i = int(np.argmax(np.abs(r)))
            if abs(r[i]) < 0.20:
                continue
            md = dv["fam"] == f
            rd = rho(dv["X"][md, i], dv["targets"][t][md])
            print(f"  {t:12s} {f:16s} {names[i]:14s} train={r[i]:+.3f} dev={rd:+.3f} "
                  f"(n_tr={m.sum()}, n_dv={md.sum()})")


def math_sqrt(x):
    return float(np.sqrt(max(x, 0.0)))


if __name__ == "__main__":
    main()
