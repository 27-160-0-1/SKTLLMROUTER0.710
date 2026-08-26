# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a12 P3b -- clean, fit-free latent structure of the 3x3 score matrix.

Because the generations of different models are independent, for m != m'
    Cov_obs(x_m, x_m') = Cov_latent(p_m, p_m')      (exactly unbiased)
and
    Var_obs(x_m) = Var_lat(p_m) + E[p_m(1-p_m)]/n
                 = Var_lat + (E[x] - E[x]^2 - Var_lat)/n
  => Var_lat = (Var_obs - (E[x]-E[x]^2)/n) / (1 - 1/n).
So the latent correlation matrix of the three success probabilities is
identifiable with NO model fit.  Done per num_generations group.
Also: bounded 2PL fit, and the per-family logit ability table.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from labdata import load_split  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

tr, dv = load_split("train"), load_split("dev")
fam = np.concatenate([[classify_family(t) for t in tr.texts], [classify_family(t) for t in dv.texts]])
S = np.vstack([tr.score, dv.score])
NG = np.vstack([tr.ngen, dv.ngen]).astype(int)
K = np.rint(S * NG).astype(int)
fams = sorted(set(fam))
FI = np.array([fams.index(f) for f in fam])
MN = ("light", "mid", "k1")


def hdr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def latent_cov(sel):
    x = S[sel]
    n = NG[sel][:, 0]          # same across models within an item
    assert (NG[sel] == n[:, None]).all()
    n = float(n[0])
    mu = x.mean(0)
    Cobs = np.cov(x.T)
    Clat = Cobs.copy()
    for j in range(3):
        v = (Cobs[j, j] - (mu[j] - mu[j] ** 2) / n) / (1 - 1 / n)
        Clat[j, j] = v
    return mu, Cobs, Clat


hdr("P3b.1  latent (noise-free) variance and correlation of the three success probabilities")
for lab, sel in (("ngen=2  (n=%d)" % int((NG[:, 0] == 2).sum()), NG[:, 0] == 2),
                 ("ngen=4  (n=%d)" % int((NG[:, 0] == 4).sum()), NG[:, 0] == 4)):
    mu, Cobs, Clat = latent_cov(sel)
    sd = np.sqrt(np.maximum(np.diag(Clat), 1e-12))
    R = Clat / np.outer(sd, sd)
    print(f"\n  {lab}")
    print(f"    mean p            : " + " ".join(f"{m:8s}={v:.4f}" for m, v in zip(MN, mu)))
    print(f"    observed sd       : " + " ".join(f"{m:8s}={v:.4f}" for m, v in zip(MN, np.sqrt(np.diag(Cobs)))))
    print(f"    latent sd of p    : " + " ".join(f"{m:8s}={v:.4f}" for m, v in zip(MN, sd)))
    print(f"    reliability (lat/obs var): " +
          " ".join(f"{m:8s}={np.diag(Clat)[j]/np.diag(Cobs)[j]:.3f}" for j, m in enumerate(MN)))
    print("    latent correlation matrix:")
    for j in range(3):
        print("      " + f"{MN[j]:8s}" + " ".join(f"{R[j,k]:+7.3f}" for k in range(3)))
    # single-factor decomposition (3 vars -> just identified)
    l01, l02, l12 = R[0, 1], R[0, 2], R[1, 2]
    a2 = l01 * l02 / l12
    b2 = l01 * l12 / l02
    c2 = l02 * l12 / l01
    print(f"    1-factor loadings (just-identified): "
          f"light={np.sign(a2)*np.sqrt(abs(a2)):.3f} mid={np.sign(b2)*np.sqrt(abs(b2)):.3f} "
          f"k1={np.sign(c2)*np.sqrt(abs(c2)):.3f}")
    print(f"    -> share of each model's latent variance explained by the common factor: "
          f"{abs(a2):.3f} / {abs(b2):.3f} / {abs(c2):.3f}")

hdr("P3b.2  same within family (does the common factor survive conditioning on family?)")
print(f"{'family':16s} {'n':>5s} {'ngen':>4s} | {'lat sd l/m/k':>22s} | "
      f"{'r(l,m)':>7s} {'r(l,k)':>7s} {'r(m,k)':>7s} | {'common-factor share k1':>22s}")
for f in fams:
    for g in (2, 4):
        sel = (fam == f) & (NG[:, 0] == g)
        if sel.sum() < 40:
            continue
        mu, Cobs, Clat = latent_cov(sel)
        d = np.maximum(np.diag(Clat), 1e-9)
        sd = np.sqrt(d)
        R = Clat / np.outer(sd, sd)
        share = abs(R[0, 2] * R[1, 2] / R[0, 1]) if abs(R[0, 1]) > 1e-6 else np.nan
        print(f"{f:16s} {sel.sum():5d} {g:4d} | {sd[0]:6.3f} {sd[1]:6.3f} {sd[2]:6.3f}      | "
              f"{R[0,1]:+7.3f} {R[0,2]:+7.3f} {R[1,2]:+7.3f} | {share:22.3f}")

# --------------------------------------------------------------- IRT with bounds
hdr("P3b.3  bounded 2PL and the per-family ability table (all 2,640 items)")
n_item = len(fam)
NF = len(fams)


def make(kind):
    def unpack(x):
        th = x[:n_item]
        r = x[n_item:]
        b = np.concatenate([[0.0], r[:2]])
        a = np.ones(3)
        c = None
        i = 2
        if kind in ("2pl", "fam2pl"):
            a = np.concatenate([[1.0], r[i:i + 2]]); i += 2
        if kind in ("fam", "fam2pl"):
            c = np.zeros((NF, 3)); c[:, 1:] = r[i:i + 2 * NF].reshape(NF, 2); i += 2 * NF
        return th, b, a, c

    def eta(x):
        th, b, a, c = unpack(x)
        e = th[:, None] * a[None, :] + b[None, :]
        if c is not None:
            e = e + c[FI]
        return e

    def nll(x):
        p = np.clip(expit(eta(x)), 1e-9, 1 - 1e-9)
        ll = (K * np.log(p) + (NG - K) * np.log(1 - p)).sum()
        return -(ll - 0.5 * (x[:n_item] ** 2).sum() / 9.0)

    def grad(x):
        th, b, a, c = unpack(x)
        p = expit(eta(x))
        r = K - NG * p
        g = np.zeros_like(x)
        g[:n_item] = -(r * a[None, :]).sum(1) + th / 9.0
        i = n_item
        g[i:i + 2] = -r[:, 1:].sum(0); i += 2
        if kind in ("2pl", "fam2pl"):
            g[i:i + 2] = -(r[:, 1:] * th[:, None]).sum(0); i += 2
        if kind in ("fam", "fam2pl"):
            for fi in range(NF):
                mm = FI == fi
                g[i + 2 * fi: i + 2 * fi + 2] = -r[mm][:, 1:].sum(0)
        return g

    nx = {"1pl": 2, "2pl": 4, "fam": 2 + 2 * NF, "fam2pl": 4 + 2 * NF}[kind]
    x0 = np.zeros(n_item + nx)
    x0[:n_item] = logit(np.clip(K.sum(1) / NG.sum(1), 0.05, 0.95))
    bnds = [(-8, 8)] * n_item + [(-8, 8)] * 2
    if kind in ("2pl", "fam2pl"):
        x0[n_item + 2:n_item + 4] = 1.0
        bnds += [(0.15, 4.0)] * 2
    if kind in ("fam", "fam2pl"):
        bnds += [(-8, 8)] * (2 * NF)
    return nll, grad, unpack, eta, x0, bnds, nx


fits = {}
for kind in ("1pl", "2pl", "fam", "fam2pl"):
    nll, grad, unpack, eta, x0, bnds, nx = make(kind)
    r = minimize(nll, x0, jac=grad, method="L-BFGS-B", bounds=bnds,
                 options=dict(maxiter=20000, maxfun=25000, ftol=1e-14, gtol=1e-9))
    fits[kind] = (r, unpack, eta)
    th, b, a, c = unpack(r.x)
    print(f"  {kind:7s} nll={r.fun:9.2f}  npar_global={nx:3d}  b={np.round(b,3)}  a={np.round(a,3)}")

r, unpack, eta = fits["fam"]
th, b, a, c = unpack(r.x)
print("\n  per-family logit ability (light = 0 reference within family):")
print(f"  {'family':16s} {'mid-light':>10s} {'k1-light':>10s} {'k1-mid':>9s}  interpretation")
tab = {}
for i, f in enumerate(fams):
    m_l = b[1] + c[i, 1]
    k_l = b[2] + c[i, 2]
    tab[f] = (m_l, k_l, k_l - m_l)
    verdict = ("k1 < mid" if k_l - m_l < -0.2 else
               "k1 ~ mid" if k_l - m_l < 0.5 else
               "k1 >> mid" if k_l - m_l > 2.0 else "k1 > mid")
    print(f"  {f:16s} {m_l:+10.3f} {k_l:+10.3f} {k_l-m_l:+9.3f}  {verdict}")
print(f"  {'(global)':16s} {b[1]:+10.3f} {b[2]:+10.3f} {b[2]-b[1]:+9.3f}")

hdr("P3b.4  how much of the total logit-ability spread is family-specific?")
mm = np.array([tab[f][0] for f in fams]); kk = np.array([tab[f][1] for f in fams])
print(f"  mid-vs-light ability: mean {mm.mean():+.3f}  sd across families {mm.std(ddof=1):.3f}"
      f"  range [{mm.min():+.3f},{mm.max():+.3f}]")
print(f"  k1 -vs-light ability: mean {kk.mean():+.3f}  sd across families {kk.std(ddof=1):.3f}"
      f"  range [{kk.min():+.3f},{kk.max():+.3f}]")
print(f"  => k1's ability is {kk.std(ddof=1)/max(mm.std(ddof=1),1e-9):.2f}x more family-dependent than mid's")
