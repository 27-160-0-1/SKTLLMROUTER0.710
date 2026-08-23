# SPDX-License-Identifier: Apache-2.0
"""a12 P3 -- (a) where k1 is WORSE than mid, and is it real or label noise?
             (b) latent-ability structure: single difficulty + model ability, or
                 genuine model-specific competencies?

(b) is answered three ways:
    1. joint binomial MLE of logit p_im = theta_i + b_m   (1PL / Rasch)
       vs    logit p_im = theta_i + b_m + c_{fam(i),m}    (+family x model)
       vs    logit p_im = a_m * theta_i + b_m             (2PL, per-model slope)
       trained on TRAIN, evaluated on DEV by LEAVE-ONE-MODEL-OUT log-likelihood:
       theta_i on dev is estimated from the other two models only.
    2. disattenuated residual correlations between models (binomial noise removed).
    3. per-family ability estimates with standard errors.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from labdata import load_split  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

rng = np.random.default_rng(0)
tr, dv = load_split("train"), load_split("dev")
famtr = np.array([classify_family(t) for t in tr.texts])
famdv = np.array([classify_family(t) for t in dv.texts])
fams = sorted(set(famtr) | set(famdv))
FI = {f: i for i, f in enumerate(fams)}

S = {"train": tr.score, "dev": dv.score}
N = {"train": tr.ngen.astype(int), "dev": dv.ngen.astype(int)}
K = {s: np.rint(S[s] * N[s]).astype(int) for s in S}
F = {"train": np.array([FI[f] for f in famtr]), "dev": np.array([FI[f] for f in famdv])}
ALLS = np.vstack([S["train"], S["dev"]])
ALLN = np.vstack([N["train"], N["dev"]])
ALLK = np.vstack([K["train"], K["dev"]])
ALLF = np.concatenate([F["train"], F["dev"]])
ALLfam = np.concatenate([famtr, famdv])
OTG = np.vstack([tr.otok / tr.ngen, dv.otok / dv.ngen])
ITG = np.vstack([tr.itok / tr.ngen, dv.itok / dv.ngen])
C = np.vstack([tr.cost, dv.cost])


def hdr(t):
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


# ------------------------------------------------------------------ (a)
hdr("P3.a1  items where k1 loses to mid -- raw counts (train+dev, n=2640)")
worse = ALLS[:, 2] < ALLS[:, 1]
better = ALLS[:, 2] > ALLS[:, 1]
print(f"  overall P(k1<mid)={worse.mean():.3f}  P(k1>mid)={better.mean():.3f}  "
      f"mean loss when worse={-(ALLS[worse,2]-ALLS[worse,1]).mean():.3f}  "
      f"mean gain when better={(ALLS[better,2]-ALLS[better,1]).mean():.3f}")
print(f"\n{'family':16s} {'n':>5s} {'d21':>7s} {'se':>6s} {'z':>6s} | {'P(k1<mid)':>9s} {'P(k1>mid)':>9s} "
      f"| {'null P(<)':>9s} {'null P(>)':>9s} | verdict")
for f in fams:
    m = ALLfam == f
    d = ALLS[m, 2] - ALLS[m, 1]
    se = d.std(ddof=1) / np.sqrt(m.sum())
    # null: both models share the SAME latent p (take pooled per-item estimate), so any
    # observed disagreement is pure binomial noise.  Simulate.
    p = np.clip((ALLK[m, 1] + ALLK[m, 2]) / (ALLN[m, 1] + ALLN[m, 2]), 1e-6, 1 - 1e-6)
    lt, gt = [], []
    for _ in range(200):
        a = rng.binomial(ALLN[m, 1], p) / ALLN[m, 1]
        b = rng.binomial(ALLN[m, 2], p) / ALLN[m, 2]
        lt.append(np.mean(b < a)); gt.append(np.mean(b > a))
    v = "k1 WORSE" if d.mean() / max(se, 1e-9) < -1.5 else ("k1 better" if d.mean() / se > 2 else "~tie")
    print(f"{f:16s} {m.sum():5d} {d.mean():+7.3f} {se:6.3f} {d.mean()/se:+6.2f} | "
          f"{np.mean(ALLS[m,2]<ALLS[m,1]):9.3f} {np.mean(ALLS[m,2]>ALLS[m,1]):9.3f} | "
          f"{np.mean(lt):9.3f} {np.mean(gt):9.3f} | {v}")

hdr("P3.a2  ruletaker + longdoc dissected (the two families where premium is not worth it)")
for f in ("ruletaker", "longdoc", "belebele", "hrmcr"):
    m = ALLfam == f
    print(f"\n-- {f} (n={m.sum()})")
    itk = ITG[m, 2]
    q = np.percentile(itk, [33, 67])
    for lab, sel in (("short", itk <= q[0]), ("mid", (itk > q[0]) & (itk <= q[1])), ("long", itk > q[1])):
        idx = np.where(m)[0][sel]
        s = ALLS[idx]
        print(f"   itok {lab:5s} n={len(idx):4d} med_itok={np.median(ITG[idx,2]):6.0f} "
              f"s={s[:,0].mean():.3f}/{s[:,1].mean():.3f}/{s[:,2].mean():.3f} "
              f"d21={(s[:,2]-s[:,1]).mean():+.3f} k1_out_med={np.median(OTG[idx,2]):6.0f} "
              f"c2/c0={C[idx,2].mean()/C[idx,0].mean():5.1f}")
    # long-thinking subset
    idx = np.where(m)[0]
    lo = OTG[idx, 2] <= np.median(OTG[idx, 2])
    for lab, sel in (("k1 thinks short", lo), ("k1 thinks long", ~lo)):
        s = ALLS[idx[sel]]
        print(f"   {lab:16s} n={sel.sum():4d} s={s[:,0].mean():.3f}/{s[:,1].mean():.3f}/{s[:,2].mean():.3f} "
              f"d21={(s[:,2]-s[:,1]).mean():+.3f}")

hdr("P3.a3  'overthinking' regressions: where does k1 destroy an answer mid already had?")
print("  restricted to items with s_mid = 1 (mid solved it fully)")
for f in fams + ["ALL"]:
    m = (np.ones(len(ALLfam), bool) if f == "ALL" else (ALLfam == f)) & (ALLS[:, 1] == 1.0)
    if m.sum() < 20:
        continue
    print(f"   {f:16s} n={m.sum():5d}  P(k1 still 1)={np.mean(ALLS[m,2]==1):.3f}  "
          f"E[s_k1]={ALLS[m,2].mean():.3f}  k1 out med={np.median(OTG[m,2]):6.0f}  "
          f"E[s_light]={ALLS[m,0].mean():.3f}")
print("\n  restricted to items with s_light = 1 (even light solved it)")
for f in fams + ["ALL"]:
    m = (np.ones(len(ALLfam), bool) if f == "ALL" else (ALLfam == f)) & (ALLS[:, 0] == 1.0)
    if m.sum() < 20:
        continue
    print(f"   {f:16s} n={m.sum():5d}  E[s_mid]={ALLS[m,1].mean():.3f}  E[s_k1]={ALLS[m,2].mean():.3f}  "
          f"mean cost mult k1={C[m,2].mean()/C[m,0].mean():6.1f}")


# ------------------------------------------------------------------ (b) IRT fits
def nll_factory(kk, nn, ff, kind):
    n_item = kk.shape[0]

    def unpack(x):
        th = x[:n_item]
        rest = x[n_item:]
        if kind == "1pl":
            b = np.concatenate([[0.0], rest[:2]])
            return th, b, None, None
        if kind == "fam":
            b = np.concatenate([[0.0], rest[:2]])
            c = np.zeros((len(fams), 3))
            c[:, 1:] = rest[2:].reshape(len(fams), 2)
            return th, b, c, None
        if kind == "2pl":
            b = np.concatenate([[0.0], rest[:2]])
            a = np.concatenate([[1.0], rest[2:4]])
            return th, b, None, a
        raise ValueError

    def eta(x):
        th, b, c, a = unpack(x)
        e = th[:, None] * (1.0 if a is None else a[None, :]) + b[None, :]
        if c is not None:
            e = e + c[ff]
        return e

    def nll(x):
        e = eta(x)
        p = expit(e)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        ll = (kk * np.log(p) + (nn - kk) * np.log(1 - p)).sum()
        # weak prior on theta to keep +-inf items finite
        th = x[:n_item]
        return -(ll - 0.5 * (th ** 2).sum() / 9.0)

    def grad(x):
        th, b, c, a = unpack(x)
        e = eta(x)
        p = expit(e)
        r = kk - nn * p                      # (n,3)
        g = np.zeros_like(x)
        slope = np.ones(3) if a is None else a
        g[:n_item] = -(r * slope[None, :]).sum(1) + th / 9.0
        off = n_item
        g[off:off + 2] = -r[:, 1:].sum(0)
        off += 2
        if kind == "fam":
            for fi in range(len(fams)):
                mm = ff == fi
                g[off + 2 * fi: off + 2 * fi + 2] = -r[mm][:, 1:].sum(0)
        if kind == "2pl":
            g[off:off + 2] = -(r[:, 1:] * th[:, None]).sum(0)
        return g

    return nll, grad, unpack, eta


def fit(kk, nn, ff, kind):
    n_item = kk.shape[0]
    nxtra = {"1pl": 2, "fam": 2 + 2 * len(fams), "2pl": 4}[kind]
    x0 = np.zeros(n_item + nxtra)
    x0[:n_item] = logit(np.clip(kk.sum(1) / nn.sum(1), 0.05, 0.95))
    if kind == "2pl":
        x0[n_item + 2:n_item + 4] = 1.0
    nll, grad, unpack, eta = fit_cache(kk, nn, ff, kind)
    res = minimize(nll, x0, jac=grad, method="L-BFGS-B",
                   options=dict(maxiter=4000, maxfun=6000, ftol=1e-12, gtol=1e-8))
    return res, unpack, eta


def fit_cache(kk, nn, ff, kind):
    return nll_factory(kk, nn, ff, kind)


hdr("P3.b1  IRT fits on TRAIN (1,760 items), leave-one-model-out evaluation on DEV (880)")
results = {}
for kind in ("1pl", "fam", "2pl"):
    res, unpack, eta = fit(K["train"], N["train"], F["train"], kind)
    th, b, c, a = unpack(res.x)
    results[kind] = (res.x, unpack)
    extra = ""
    if kind == "fam":
        extra = "\n     family offsets (light is the reference, 0):\n" + "\n".join(
            f"       {f:16s} mid {c[i,1]:+.3f}  k1 {c[i,2]:+.3f}" for i, f in enumerate(fams))
    if kind == "2pl":
        extra = f"  slopes a = [1.000, {a[1]:.3f}, {a[2]:.3f}]"
    print(f"  {kind:4s} nll={res.fun:10.2f}  b = [0.000, {b[1]:+.3f}, {b[2]:+.3f}]{extra}")

# LOMO evaluation on dev: theta from the two observed models, predict the third
print("\n  leave-one-model-out on DEV: mean log-likelihood per observed generation")
print(f"  {'held-out':10s} {'globalmean':>11s} {'familymean':>11s} {'1pl':>9s} {'fam':>9s} {'2pl':>9s} "
      f"| {'1pl corr':>9s} {'fam corr':>9s}")


def theta_from(kk, nn, obs, b, a, c, ff):
    """1-D Newton for theta given observed models."""
    th = np.zeros(kk.shape[0])
    for _ in range(60):
        e = th[:, None] * a[None, :] + b[None, :] + (0 if c is None else c[ff])
        p = expit(e)
        r = ((kk - nn * p) * a[None, :])[:, obs].sum(1) - th / 9.0
        h = -((nn * p * (1 - p)) * (a ** 2)[None, :])[:, obs].sum(1) - 1 / 9.0
        step = r / h
        th = th - np.clip(step, -2, 2)
    return th


rowsout = []
for hold in range(3):
    obs = [j for j in range(3) if j != hold]
    kk, nn, ff = K["dev"], N["dev"], F["dev"]
    line = {}
    # baselines
    pg = np.clip(K["train"][:, hold].sum() / N["train"][:, hold].sum(), 1e-6, 1 - 1e-6)
    line["global"] = binom.logpmf(kk[:, hold], nn[:, hold], pg).sum() / nn[:, hold].sum()
    pf = np.array([np.clip(K["train"][F["train"] == i, hold].sum() /
                           max(N["train"][F["train"] == i, hold].sum(), 1), 1e-4, 1 - 1e-4)
                   for i in range(len(fams))])[ff]
    line["family"] = binom.logpmf(kk[:, hold], nn[:, hold], pf).sum() / nn[:, hold].sum()
    corrs = {}
    for kind in ("1pl", "fam", "2pl"):
        x, unpack = results[kind]
        th_tr, b, c, a = unpack(x)
        a = np.ones(3) if a is None else a
        th = theta_from(kk, nn, obs, b, a, c, ff)
        e = th * a[hold] + b[hold] + (0 if c is None else c[ff, hold])
        p = np.clip(expit(e), 1e-6, 1 - 1e-6)
        line[kind] = binom.logpmf(kk[:, hold], nn[:, hold], p).sum() / nn[:, hold].sum()
        corrs[kind] = np.corrcoef(p, S["dev"][:, hold])[0, 1]
    print(f"  {['light','mid','k1'][hold]:10s} {line['global']:11.4f} {line['family']:11.4f} "
          f"{line['1pl']:9.4f} {line['fam']:9.4f} {line['2pl']:9.4f} | "
          f"{corrs['1pl']:9.3f} {corrs['fam']:9.3f}")
    rowsout.append((hold, line, corrs))

hdr("P3.b2  disattenuated residual correlations (binomial noise removed)")
# fit the 'fam' model on ALL 2640, take residuals s_obs - p_hat, disattenuate
res, unpack, eta = fit(ALLK, ALLN, ALLF, "fam")
th, b, c, a = unpack(res.x)
P = expit(eta(res.x))
resid = ALLS - P
noise_var = (P * (1 - P) / ALLN).mean(0)
print(f"  model     obs resid var   binomial noise var   latent resid var   reliability")
lat = []
for j, m in enumerate(("light", "mid", "k1")):
    ov = resid[:, j].var()
    lv = max(ov - noise_var[j], 1e-9)
    lat.append(lv)
    print(f"  {m:8s} {ov:14.5f} {noise_var[j]:20.5f} {lv:18.5f} {lv/ov:12.3f}")
print("\n  pairwise residual correlations (raw / disattenuated):")
for i, j, nm in ((0, 1, "light-mid"), (0, 2, "light-k1"), (1, 2, "mid-k1")):
    cov = np.cov(resid[:, i], resid[:, j])[0, 1]
    raw = cov / np.sqrt(resid[:, i].var() * resid[:, j].var())
    dis = cov / np.sqrt(lat[i] * lat[j])
    print(f"    {nm:10s} raw {raw:+.3f}   disattenuated {dis:+.3f}")
print("\n  NOTE: theta_i was fitted from all three models, which forces residuals to be")
print("  roughly mean-zero per item; the informative object is the RELATIVE ordering of")
print("  the three disattenuated correlations, not their absolute level.")

hdr("P3.b3  same, but theta fitted WITHOUT one model (clean residual for that model)")
for hold in range(3):
    obs = [j for j in range(3) if j != hold]
    kk = ALLK[:, obs]; nn = ALLN[:, obs]
    # 1PL on the two observed models only
    n_item = kk.shape[0]

    def nll(x):
        th = x[:n_item]; bb = np.array([0.0, x[n_item]])
        p = np.clip(expit(th[:, None] + bb[None, :]), 1e-9, 1 - 1e-9)
        return -((kk * np.log(p) + (nn - kk) * np.log(1 - p)).sum() - 0.5 * (th ** 2).sum() / 9.0)

    x0 = np.zeros(n_item + 1)
    x0[:n_item] = logit(np.clip(kk.sum(1) / nn.sum(1), 0.05, 0.95))
    r = minimize(nll, x0, method="L-BFGS-B", options=dict(maxiter=3000))
    th = r.x[:n_item]
    # per-family regression of held-out model score on theta -> residual = model-specific skill
    p_pred = expit(th + 0.0)
    rr = np.corrcoef(th, ALLS[:, hold])[0, 1]
    print(f"  held-out {['light','mid','k1'][hold]:6s}: corr(theta_from_other_two, s_holdout) = {rr:+.3f}")
    # how much extra does family membership explain on top of theta?
    X1 = np.column_stack([np.ones(n_item), th, th ** 2])
    Xf = np.column_stack([X1, np.stack([(ALLF == i).astype(float) for i in range(len(fams))], 1)])
    for nm, X in (("theta only", X1), ("theta+family", Xf)):
        w, *_ = np.linalg.lstsq(X, ALLS[:, hold], rcond=None)
        e = ALLS[:, hold] - X @ w
        print(f"      {nm:14s} R2={1-e.var()/ALLS[:,hold].var():.4f}  rmse={e.std():.4f}")
