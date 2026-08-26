# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a14 step 2 -- fit the joint latent-p model and draw p | observed k.

Model (per stratum = family x num_generations):
    z_i ~ N(0, Rz)   3-dim
    p_ij = Beta_{a_fj,b_fj}^{-1}( Phi(z_ij) )
    k_ij ~ Binomial(n, p_ij)  independent across j given p
Moments matched exactly:
    E[p_ij]                        = mean(score)
    Var(p_ij)                      = (E[s^2] - E[1/n] E[s]) / (1 - E[1/n]) - mu^2
    Corr(p_ij, p_ij')  (target)    = Cov(s_j,s_j') / sqrt(Var(p_j) Var(p_j'))
Rz is solved by bisection so the induced Pearson corr of p hits the target.

Posterior p | k by importance sampling on a large common prior pool per stratum
(the likelihood only depends on the k-triple, of which there are <=27 for n=2).

Writes experiments/lab/_a14_cache/pdraws_<split>.npz with
  p      (D, N, 3)  posterior draws of the latent p
  phat   (N, 3)     posterior mean  E[p | k]
"""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
from scipy import stats, special

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_split, MODEL_IDS                      # noqa: E402
from ossp_router.similarity import classify_family             # noqa: E402

OUT = HERE / "_a14_cache"
OUT.mkdir(exist_ok=True)
NPOOL = 120_000
NDRAW = 60


def fam_of(sp):
    return np.array([classify_family(t) for t in sp.texts])


def cell_moments(x, nn):
    mu = float(x.mean())
    inv = float((1.0 / nn).mean())
    Ep2 = (float(np.mean(x ** 2)) - inv * mu) / (1.0 - inv)
    vp = Ep2 - mu * mu
    return mu, vp


def beta_ab(mu, vp):
    """Beta params from mean/var, with guards.  Returns None for a point mass."""
    hi = mu * (1 - mu)
    if not (0 < mu < 1) or vp <= 1e-6 * max(hi, 1e-9):
        return None
    vp = min(vp, hi * (1 - 1e-4))
    c = hi / vp - 1.0
    return max(c * mu, 1e-3), max(c * (1 - mu), 1e-3)


def _icdf(z, ab):
    if ab is None:
        return np.zeros_like(z)          # point mass -> zero variance
    return stats.beta.ppf(special.ndtr(z), ab[0], ab[1])


# ---- Hermite expansion of the copula transform: E[f(Z0) g(Z1)] under corr rho
# equals sum_k rho^k a_k b_k / k!  with a_k = E[f(Z) He_k(Z)].  Exact and fast.
_KMAX = 14
_GHX = np.linspace(-9.0, 9.0, 20001)
_W = np.exp(-0.5 * _GHX ** 2) / np.sqrt(2 * np.pi)
_W = _W * (_GHX[1] - _GHX[0])
_W = _W / _W.sum()
_HE = np.stack([np.polynomial.hermite_e.hermeval(
    _GHX, np.eye(_KMAX + 1)[k]) for k in range(_KMAX + 1)])   # (K+1, nodes)
_FACT = np.array([float(math.factorial(k)) for k in range(_KMAX + 1)])


def hermite_coefs(ab):
    """a_k = E[f(Z) He_k(Z)] for f(z) = Beta^-1(Phi(z)); returns (a, sd)."""
    if ab is None:
        return np.zeros(_KMAX + 1), 0.0
    fx = _icdf(_GHX, ab)
    a = _HE @ (_W * fx)
    var = float(np.sum(a[1:] ** 2 / _FACT[1:]))
    return a, np.sqrt(max(var, 0.0))


def induced_corr_from_coefs(a, sa, b, sb, rho):
    if sa <= 0 or sb <= 0:
        return 0.0
    k = np.arange(1, _KMAX + 1)
    return float(np.sum((rho ** k) * a[1:] * b[1:] / _FACT[1:]) / (sa * sb))


def solve_rho(coefs_i, coefs_j, target):
    (a, sa), (b, sb) = coefs_i, coefs_j
    if sa <= 0 or sb <= 0:
        return 0.0
    lo, hi = -0.999, 0.999
    if induced_corr_from_coefs(a, sa, b, sb, hi) < target:
        return hi
    if induced_corr_from_coefs(a, sa, b, sb, lo) > target:
        return lo
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if induced_corr_from_coefs(a, sa, b, sb, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def nearest_psd(R):
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-6, None)
    A = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(A))
    return A / np.outer(d, d)


def fit_and_sample(sp, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    fam = fam_of(sp)
    N = len(sp)
    n_i = sp.ngen[:, 0].astype(int)
    k = np.rint(sp.score * sp.ngen).astype(int)
    P = np.zeros((NDRAW, N, 3))
    phat = np.zeros((N, 3))
    info = []
    strata = sorted(set(zip(fam.tolist(), n_i.tolist())))
    for (f_, ng) in strata:
        m = (fam == f_) & (n_i == ng)
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        abs_ = []
        mus, vps = [], []
        for j in range(3):
            mu, vp = cell_moments(sp.score[m, j], sp.ngen[m, j])
            vp = max(vp, 0.0)
            mus.append(mu); vps.append(vp)
            abs_.append(beta_ab(mu, vp))
        # target latent correlations = observed score covariances / sqrt(var_p)
        Cs = np.cov(sp.score[m].T) if len(idx) > 3 else np.zeros((3, 3))
        coefs = [hermite_coefs(ab) for ab in abs_]
        Rz = np.eye(3)
        for a in range(3):
            for b in range(a + 1, 3):
                den = np.sqrt(vps[a] * vps[b])
                tgt = float(np.clip(Cs[a, b] / den, -0.98, 0.98)) if den > 1e-9 else 0.0
                r = solve_rho(coefs[a], coefs[b], tgt)
                Rz[a, b] = Rz[b, a] = r
        Rz = nearest_psd(Rz)
        L = np.linalg.cholesky(Rz)
        Z = rng.standard_normal((NPOOL, 3)) @ L.T
        pool = np.column_stack([_icdf(Z[:, j], abs_[j]) if abs_[j] is not None
                                else np.full(NPOOL, mus[j]) for j in range(3)])
        # importance weights per distinct k-triple
        logp = np.log(np.clip(pool, 1e-12, 1 - 1e-12))
        log1p_ = np.log(np.clip(1 - pool, 1e-12, 1 - 1e-12))
        ktr = [tuple(row) for row in k[m]]
        uniq = sorted(set(ktr))
        for kt in uniq:
            ll = np.zeros(NPOOL)
            for j in range(3):
                ll += kt[j] * logp[:, j] + (ng - kt[j]) * log1p_[:, j]
            w = np.exp(ll - ll.max())
            w /= w.sum()
            sel_rows = idx[[t == kt for t in ktr]]
            pm = w @ pool
            phat[sel_rows] = pm
            ch = rng.choice(NPOOL, size=(NDRAW, len(sel_rows)), p=w)
            P[:, sel_rows, :] = pool[ch]
            info.append(dict(fam=f_, ng=ng, k=kt, n=len(sel_rows), ess=1.0 / (w ** 2).sum()))
        if verbose:
            print(f"  {f_:16s} ng={ng} N={len(idx):4d} mu={np.round(mus,3)} "
                  f"vp={np.round(vps,3)} Rz_offdiag={np.round([Rz[0,1],Rz[0,2],Rz[1,2]],3)}")
    return P, phat, info


def check(sp, P, phat):
    """posterior-predictive check: does the model reproduce the observed score
    distribution, the cross-model correlations and the k-histogram?"""
    rng = np.random.default_rng(7)
    n = sp.ngen.astype(int)
    print("\n  posterior-predictive check (replicate k from p draws):")
    reps = []
    for d in range(min(20, P.shape[0])):
        reps.append(rng.binomial(n, P[d]) / n)
    reps = np.array(reps)
    print(f"    {'model':12s} {'obs mean':>9s} {'rep mean':>9s} {'obs sd':>8s} {'rep sd':>8s} "
          f"{'obs P(s=1)':>11s} {'rep P(s=1)':>11s}")
    for j, m in enumerate(MODEL_IDS):
        print(f"    {m:12s} {sp.score[:,j].mean():9.4f} {reps[:,:,j].mean():9.4f} "
              f"{sp.score[:,j].std():8.4f} {reps[:,:,j].std():8.4f} "
              f"{np.mean(sp.score[:,j]==1):11.4f} {np.mean(reps[:,:,j]==1):11.4f}")
    co = np.mean([np.corrcoef(r.T)[np.triu_indices(3, 1)] for r in reps], axis=0)
    print(f"    obs score corr (01,02,12) = {np.round(np.corrcoef(sp.score.T)[np.triu_indices(3,1)],4)}")
    print(f"    rep score corr (01,02,12) = {np.round(co,4)}")
    pv = P.reshape(-1, 3)
    print(f"    latent p: mean={np.round(pv.mean(0),4)} sd={np.round(pv.std(0),4)} "
          f"corr={np.round(np.corrcoef(pv.T)[np.triu_indices(3,1)],4)}")
    print(f"    phat    : mean={np.round(phat.mean(0),4)} sd={np.round(phat.std(0),4)}")
    print(f"    P(p>0.95)={np.round((pv>0.95).mean(0),3)}  P(p<0.05)={np.round((pv<0.05).mean(0),3)}")


if __name__ == "__main__":
    for name in ("dev", "train"):
        sp = load_split(name)
        print(f"=== fitting {name} ===")
        P, phat, info = fit_and_sample(sp, seed=11 if name == "dev" else 12)
        ess = np.array([i["ess"] for i in info])
        print(f"  importance-sampling ESS: min={ess.min():.0f} med={np.median(ess):.0f} "
              f"(pool={NPOOL})")
        check(sp, P, phat)
        np.savez_compressed(OUT / f"pdraws_{name}.npz", p=P.astype(np.float32),
                            phat=phat.astype(np.float32))
        print(f"  wrote {OUT / f'pdraws_{name}.npz'}")
