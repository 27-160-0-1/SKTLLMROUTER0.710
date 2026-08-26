# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a04 / step 4 -- what the confidence is actually worth.

(a) between-family vs within-family decomposition of the score-prediction corr
(b) is the ordinal-head variance informative beyond mu(1-mu)?
(c) honest recalibration: Platt / isotonic fitted on TRAIN cross-fit probs,
    applied to dev -> ECE, corr, final score
(d) head redundancy: 4 thresholds vs the 2 that carry mass
(e) perturbation sensitivity: jitter the predicted scores by their measured
    error and look at the spread of the tier score; which items flip most
(f) variance-aware shrinkage of E[s] (dev-fit upper bound, honest train-fit)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import similarity

tr, dv = load_split("train"), load_split("dev")
n = len(dv); IDX = np.arange(n)
DEP = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
Z = np.load(Path(__file__).resolve().parents[0] / "a04_ordinal_cache.npz")
Pdev, Ptr, Edev = Z["Pdev"], Z["Ptr"], Z["Edev"]
Etr = 0.25 * Ptr.sum(2)
fam = np.array([similarity.classify_family(t) for t in dv.texts])
famtr = np.array([similarity.classify_family(t) for t in tr.texts])
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
THRESH = (0.25, 0.5, 0.75, 1.0)


def final(S, C=None, safe=SAFE):
    tot = 0.0; parts = []
    for t in TIERS:
        cc = DEP[f"cost_{t}"] if C is None else C
        r = tier_result(S if S.ndim == 2 else S, cc, dv, t, safe[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else '!BUST'}")
    return tot, "  ".join(parts)


print("=" * 100)
print("STEP 4a  how much of the score-prediction correlation is just the family label?")
print("=" * 100)
for name, S in (("deployed", DEP["score_fast"]), ("mine(ordinal)", Edev)):
    print(f"\n  -- {name}")
    for j, m in enumerate(MODEL_IDS):
        p, y = S[:, j], dv.score[:, j]
        # family-mean centring
        pc, yc = p.copy(), y.copy()
        for f in set(fam):
            mm = fam == f
            pc[mm] -= p[mm].mean(); yc[mm] -= y[mm].mean()
        # family means alone
        pf = np.array([p[fam == f].mean() for f in fam])
        yf = np.array([y[fam == f].mean() for f in fam])
        print(f"     {m:11s} overall corr={np.corrcoef(p,y)[0,1]:.3f} | "
              f"within-family corr={np.corrcoef(pc,yc)[0,1]:.3f} | "
              f"family-mean-only corr={np.corrcoef(pf,y)[0,1]:.3f} | "
              f"var explained by family: pred {1-pc.var()/p.var():.3f} true {1-yc.var()/y.var():.3f}")

print()
print("=" * 100)
print("STEP 4b  is the head variance informative beyond mu(1-mu)?")
print("=" * 100)
grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
Pm = np.minimum.accumulate(np.clip(Pdev, 0, 1), axis=2)
surv = np.concatenate([np.ones((n, 3, 1)), Pm], axis=2)
mass = np.concatenate([-np.diff(surv, axis=2), Pm[:, :, 3:4]], axis=2)
mass = np.clip(mass, 0, None); mass /= mass.sum(2, keepdims=True)
mu = (mass * grid).sum(2)
var = (mass * grid ** 2).sum(2) - mu ** 2
bern = Edev * (1 - Edev)
resid = var - bern
for j, m in enumerate(MODEL_IDS):
    e2 = (Edev[:, j] - dv.score[:, j]) ** 2
    def rho(a, b):
        return np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
    print(f"  {m:11s} rho(var,e2)={rho(var[:,j],e2):+.3f}  rho(mu(1-mu),e2)={rho(bern[:,j],e2):+.3f}  "
          f"rho(var-mu(1-mu),e2)={rho(resid[:,j],e2):+.3f}  corr(var,mu(1-mu))={np.corrcoef(var[:,j],bern[:,j])[0,1]:.3f}")
    # partial: residualise var on a quadratic in mu, then correlate with e2
    A = np.column_stack([np.ones(n), Edev[:, j], Edev[:, j] ** 2])
    r = var[:, j] - A @ np.linalg.lstsq(A, var[:, j], rcond=None)[0]
    print(f"              partial rho(var | quadratic in E[s], e2) = {rho(r, e2):+.3f}")

print()
print("=" * 100)
print("STEP 4c  honest recalibration of the 12 heads (Platt fit on TRAIN cross-fit probs)")
print("=" * 100)
def ece(p, y, nb=10):
    qs = np.quantile(p, np.linspace(0, 1, nb + 1)); qs[0] -= 1e-9
    b = np.digitize(p, qs[1:-1]); e = 0.0
    for k in range(nb):
        m = b == k
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
Pcal = np.zeros_like(Pdev); Piso = np.zeros_like(Pdev)
print(f"  {'head':22s} {'a':>7s} {'b':>7s} | {'ECE raw':>8s} {'ECE platt':>9s} {'ECE iso':>8s}")
for j, m in enumerate(MODEL_IDS):
    for ti, th in enumerate(THRESH):
        ytr = (tr.score[:, j] >= th - 1e-9).astype(int)
        ydv = (dv.score[:, j] >= th - 1e-9).astype(float)
        if ytr.min() == ytr.max():
            Pcal[:, j, ti] = Pdev[:, j, ti]; Piso[:, j, ti] = Pdev[:, j, ti]; continue
        lr = LogisticRegression(C=1e6).fit(logit(Ptr[:, j, ti]).reshape(-1, 1), ytr)
        Pcal[:, j, ti] = lr.predict_proba(logit(Pdev[:, j, ti]).reshape(-1, 1))[:, 1]
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(Ptr[:, j, ti], ytr)
        Piso[:, j, ti] = ir.predict(Pdev[:, j, ti])
        print(f"  {m[:9]:9s} P(s>={th:.2f})  {lr.intercept_[0]:+7.3f} {lr.coef_[0,0]:7.3f} | "
              f"{ece(Pdev[:,j,ti],ydv):8.4f} {ece(Pcal[:,j,ti],ydv):9.4f} {ece(Piso[:,j,ti],ydv):8.4f}")
Ecal = 0.25 * Pcal.sum(2); Eiso = 0.25 * Piso.sum(2)
print()
for lbl, E in (("raw", Edev), ("platt", Ecal), ("isotonic", Eiso)):
    cor = [np.corrcoef(E[:, j], dv.score[:, j])[0, 1] for j in range(3)]
    rms = [np.sqrt(((E[:, j] - dv.score[:, j]) ** 2).mean()) for j in range(3)]
    f, parts = final(E)
    print(f"  {lbl:9s} corr={'/'.join(f'{c:.3f}' for c in cor)} rmse={'/'.join(f'{r:.4f}' for r in rms)} "
          f"final={f:.4f}  {parts}")

print()
print("=" * 100)
print("STEP 4d  head redundancy: how much distinct mass do the 4 thresholds carry?")
print("=" * 100)
for sp in (tr, dv):
    print(f"  {sp.name}: P(s==0.25) = " + " ".join(f"{np.mean(np.isclose(sp.score[:,j],0.25)):.4f}" for j in range(3))
          + "   P(s==0.75) = " + " ".join(f"{np.mean(np.isclose(sp.score[:,j],0.75)):.4f}" for j in range(3))
          + f"   frac ngen==4 = {np.mean(sp.ngen[:,0]==4):.3f}")
E2 = 0.5 * (Pdev[:, :, 1] + Pdev[:, :, 3])      # only the two thresholds that carry mass
E4 = Edev
for j, m in enumerate(MODEL_IDS):
    y = dv.score[:, j]
    print(f"  {m:11s} 4-head corr={np.corrcoef(E4[:,j],y)[0,1]:.3f} rmse={np.sqrt(((E4[:,j]-y)**2).mean()):.4f} | "
          f"2-head corr={np.corrcoef(E2[:,j],y)[0,1]:.3f} rmse={np.sqrt(((E2[:,j]-y)**2).mean()):.4f} | "
          f"corr(4,2)={np.corrcoef(E4[:,j],E2[:,j])[0,1]:.4f}")
f4, p4 = final(E4); f2, p2 = final(E2)
print(f"  final: 4-head {f4:.4f} ({p4})")
print(f"         2-head {f2:.4f} ({p2})")
Emono = 0.25 * Pm.sum(2)
fm, pm = final(Emono)
print(f"  monotonised 4-head {fm:.4f} ({pm})   corr="
      + "/".join(f"{np.corrcoef(Emono[:,j],dv.score[:,j])[0,1]:.3f}" for j in range(3)))

print()
print("=" * 100)
print("STEP 4e  perturbation sensitivity of the tier score")
print("  jitter: S' = S + eps, eps ~ N(0, sigma_j) with sigma_j = the measured per-model RMSE")
print("=" * 100)
rng = np.random.default_rng(0)
sig = np.array([np.sqrt(((DEP['score_fast'][:, j] - dv.score[:, j]) ** 2).mean()) for j in range(3)])
print(f"  per-model score RMSE (deployed): {np.round(sig,4)}")
flip_count = {t: np.zeros(n) for t in TIERS}
for t in TIERS:
    base = tier_result(DEP[f"score_{t}"], DEP[f"cost_{t}"], dv, t, SAFE[t])
    scores, ratios, busts = [], [], 0
    for rep in range(200):
        for scale in (1.0,):
            S = DEP[f"score_{t}"] + rng.normal(0, 1, (n, 3)) * sig * scale
            r = tier_result(np.clip(S, 0, 1), DEP[f"cost_{t}"], dv, t, SAFE[t])
            scores.append(r["score"]); ratios.append(r["ratio"]); busts += (not r["passed"])
            flip_count[t] += (r["sel"] != base["sel"])
    scores = np.array(scores); ratios = np.array(ratios)
    print(f"  {t:9s} base={base['score']:.4f}/r{base['ratio']:.3f} | jittered score "
          f"mean={scores.mean():.4f} sd={scores.std():.4f} p5={np.percentile(scores,5):.4f} "
          f"p95={np.percentile(scores,95):.4f} | ratio sd={ratios.std():.4f} busts={busts}/200")
    fc = flip_count[t] / 200
    print(f"            flip prob: mean={fc.mean():.3f}  >0.5 for {np.sum(fc>0.5)} items  "
          f"=0 for {np.sum(fc==0)} items")
    # which families are the most fragile
    for f in sorted(set(fam)):
        mm = fam == f
        print(f"              {f:16s} n={mm.sum():4d} flip={fc[mm].mean():.3f}")

print()
print("=" * 100)
print("STEP 4f  variance-aware shrinkage of E[s] toward the family mean")
print("  S' = S + k * (fammean - S) * (var / mean_var)   (k>0 shrinks uncertain items harder)")
print("=" * 100)
fmean = np.zeros_like(Edev)
for f in set(fam):
    mm = fam == f
    fmean[mm] = Etr[famtr == f].mean(0) if (famtr == f).sum() >= 8 else Etr.mean(0)
w = var / var.mean(0, keepdims=True)
print(f"  {'k':>6s} {'final':>8s}  corr(L/M/K)                per-tier")
for k in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
    S = Edev + k * (fmean - Edev) * w
    S = np.clip(S, 0, 1)
    f_, parts = final(S)
    cor = "/".join(f"{np.corrcoef(S[:,j],dv.score[:,j])[0,1]:.3f}" for j in range(3))
    print(f"  {k:6.2f} {f_:8.4f}  {cor}    {parts}")
print("  (uniform shrink control, no variance weighting)")
for k in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
    S = np.clip(Edev + k * (fmean - Edev), 0, 1)
    f_, parts = final(S)
    cor = "/".join(f"{np.corrcoef(S[:,j],dv.score[:,j])[0,1]:.3f}" for j in range(3))
    print(f"  {k:6.2f} {f_:8.4f}  {cor}    {parts}")
