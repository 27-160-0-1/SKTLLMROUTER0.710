# SPDX-License-Identifier: Apache-2.0
"""a14 step 1 -- hierarchical noise model for the label.

score_ij = k_ij / n_i,  k_ij ~ Binomial(n_i, p_ij),  n_i in {2,4}.

Identities used (exact, no modelling assumption):
    E[s_ij]                = E[p_ij]
    Var(s_ij)              = Var(p_ij) + E[p_ij(1-p_ij)]/n
    Cov(s_ij, s_ij')       = Cov(p_ij, p_ij')     for j != j'   (noise independent
                                                   across models given p)
so the *cross-model* covariance of the latent p is observed directly and only
the diagonal needs a noise correction.

Outputs the Var decomposition per (family, model) and per model overall, and
fits two joint models of the latent p field used later:
  A) Gaussian copula + Beta marginals, moments matched  (smooth)
  B) joint NPMLE on a 3-d grid via EM                    (spiky, non-parametric)
Both are then used for posterior sampling of p given the observed k.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_split, MODEL_IDS                      # noqa: E402
from ossp_router.similarity import classify_family             # noqa: E402

OUT = HERE / "_a14_cache"
OUT.mkdir(exist_ok=True)


def fam_of(sp):
    return np.array([classify_family(t) for t in sp.texts])


def var_decomp(s, n, mask):
    """returns mu, var_s, var_p, noise_var for one (group, model) cell."""
    x = s[mask]
    nn = n[mask]
    mu = x.mean()
    var_s = x.var()
    # E[s^2] = E[1/n]*E[p] + (1-E[1/n])*E[p^2]   (assumes p _|_ n inside the cell;
    # cells are (family, n) strata below so this is exact there)
    inv = (1.0 / nn).mean()
    Ep2 = (np.mean(x ** 2) - inv * mu) / (1.0 - inv)
    var_p = Ep2 - mu * mu
    noise = var_s - var_p
    return mu, var_s, var_p, noise


def main():
    tr = load_split("train")
    dv = load_split("dev")
    rows = []
    print("=" * 100)
    print("VARIANCE DECOMPOSITION   Var(score) = Var(p) + E[p(1-p)]/n")
    print("=" * 100)
    for sp in (tr, dv):
        fam = fam_of(sp)
        print(f"\n--- {sp.name} (n={len(sp)}) ---")
        print(f"{'family':16s} {'N':>4s} {'ngen':>5s} | " + " | ".join(
            f"{m:>28s}" for m in MODEL_IDS))
        print(f"{'':16s} {'':4s} {'':5s} | " + " | ".join(
            f"{'mu':>6s}{'Var(s)':>8s}{'Var(p)':>8s}{'sig%':>6s}" for _ in MODEL_IDS))
        for f_ in sorted(set(fam)):
            m0 = fam == f_
            for ng in sorted(set(sp.ngen[m0, 0].astype(int))):
                m = m0 & (sp.ngen[:, 0].astype(int) == ng)
                if m.sum() < 5:
                    continue
                cells = []
                for j in range(3):
                    mu, vs, vp, nz = var_decomp(sp.score[:, j], sp.ngen[:, j], m)
                    frac = 100 * vp / vs if vs > 0 else float("nan")
                    cells.append(f"{mu:6.3f}{vs:8.4f}{vp:8.4f}{frac:6.1f}")
                    rows.append(dict(split=sp.name, family=f_, ngen=int(ng), model=MODEL_IDS[j],
                                     N=int(m.sum()), mu=mu, var_s=vs, var_p=vp, noise=nz))
                print(f"{f_:16s} {m.sum():4d} {ng:5d} | " + " | ".join(cells))
        # overall per model (pooled, using per-item n)
        print(f"{'ALL':16s} {len(sp):4d} {'mix':>5s} | " + " | ".join(
            "{:6.3f}{:8.4f}{:8.4f}{:6.1f}".format(*(lambda t: (t[0], t[1], t[2],
                                                              100 * t[2] / t[1]))(
                var_decomp(sp.score[:, j], sp.ngen[:, j], np.ones(len(sp), bool))))
            for j in range(3)))
    (OUT / "var_decomp.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- pooled numbers
    print("\n" + "=" * 100)
    print("POOLED (train+dev, 2640) -- the numbers that matter for the exchange rate")
    print("=" * 100)
    s = np.vstack([tr.score, dv.score])
    n = np.vstack([tr.ngen, dv.ngen])
    fam = np.concatenate([fam_of(tr), fam_of(dv)])
    print(f"{'model':12s} {'mu':>7s} {'sd(s)':>8s} {'sd(p)':>8s} {'sd(p)/sd(s)':>12s} "
          f"{'reliab':>8s} {'shrink b':>9s}")
    for j, m in enumerate(MODEL_IDS):
        mu, vs, vp, nz = var_decomp(s[:, j], n[:, j], np.ones(len(s), bool))
        rel = vp / vs                      # = corr(s, p)^2 ; also the OLS slope p~s
        print(f"{m:12s} {mu:7.4f} {np.sqrt(vs):8.4f} {np.sqrt(vp):8.4f} "
              f"{np.sqrt(vp/vs):12.4f} {rel:8.4f} {rel:9.4f}")
    # within-family (family effect removed) -- how much of Var(p) is *not* family mean
    print("\nwithin-family decomposition (family means removed):")
    for j, m in enumerate(MODEL_IDS):
        resid = s[:, j].copy()
        famvar = 0.0
        gm = s[:, j].mean()
        for f_ in set(fam):
            k = fam == f_
            famvar += k.mean() * (s[k, j].mean() - gm) ** 2
            resid[k] -= s[k, j].mean()
        # noise variance is unchanged by removing a group mean
        _, vs, vp, nz = var_decomp(s[:, j], n[:, j], np.ones(len(s), bool))
        vp_within = vp - famvar
        print(f"  {m:12s} Var(p)={vp:.4f} = family {famvar:.4f} ({100*famvar/vp:.1f}%) "
              f"+ within-family {vp_within:.4f} ({100*vp_within/vp:.1f}%)")

    # ---------------------------------------------------------------- cross-model cov
    print("\ncross-model LATENT p correlation (dev; Cov(p_j,p_j') = Cov(s_j,s_j') exactly):")
    fdv = fam_of(dv)
    C = np.cov(dv.score.T)
    vp = np.array([var_decomp(dv.score[:, j], dv.ngen[:, j], np.ones(len(dv), bool))[2]
                   for j in range(3)])
    Cp = C.copy()
    np.fill_diagonal(Cp, vp)
    R = Cp / np.sqrt(np.outer(vp, vp))
    print("  observed score corr:\n", np.round(np.corrcoef(dv.score.T), 4))
    print("  implied latent p corr:\n", np.round(R, 4))
    print("  eigenvalues of latent corr:", np.round(np.linalg.eigvalsh(R), 4))

    print("\nGAIN variance (what the allocator actually consumes), dev:")
    for (a, b, lab) in ((0, 1, "mid-light"), (1, 2, "k1-mid"), (0, 2, "k1-light")):
        gs = dv.score[:, b] - dv.score[:, a]
        vg_s = gs.var()
        vg_p = vp[a] + vp[b] - 2 * Cp[a, b]
        print(f"  {lab:10s} mean={gs.mean():+.4f} Var(obs gain)={vg_s:.4f} "
              f"Var(latent gain)={vg_p:.4f} signal-frac={vg_p/vg_s:.3f} "
              f"sd(latent)={np.sqrt(max(vg_p,0)):.4f}")


if __name__ == "__main__":
    main()
