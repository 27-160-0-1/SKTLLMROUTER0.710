# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a05 step 7:
(a) posterior-predictive re-estimate of the oracle's label-noise inflation
(b) global per-model decision-price multiplier (trust-weighted pricing)
(c) per-decision gain shrinkage gamma1 (mid-light) x gamma2 (k1-mid)
with a bootstrap on whatever moves."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
ROOT = Path(__file__).resolve().parents[2]

dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
Cc = np.load(ROOT / "experiments/lab/a05-selection-failure_cache.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
N = len(dv); IDX = np.arange(N); FAM = Cc["fam"]; PHAT = Cc["phat"]
L = dv.cost[:, 0].sum()

# ---------------------------------------------------------------- (a) posterior predictive
print("=== (a) posterior-predictive check of oracle label-noise inflation ===")
n = dv.ngen.astype(int); k = np.rint(dv.score * n).astype(int)
A = np.ones_like(dv.score); B = np.ones_like(dv.score)
for f_ in set(FAM.tolist()):
    msk = FAM == f_
    for j in range(3):
        x = dv.score[msk, j]; mu = float(x.mean()); var = float(x.var()); nn = float(n[msk, j].mean())
        if not (0 < mu < 1):
            continue
        vp = (var - (mu - mu * mu) / nn) / (1 - 1.0 / nn) if nn > 1 else var
        vp = float(np.clip(vp, 1e-4, mu * (1 - mu) - 1e-4))
        c = mu * (1 - mu) / vp - 1
        A[msk, j] = max(c * mu, 0.05); B[msk, j] = max(c * (1 - mu), 0.05)
pa, pb = A + k, B + (n - k)
rng = np.random.default_rng(7)
inn, out, dep = [], [], []
for _ in range(40):
    p = rng.beta(pa, pb)                       # a plausible latent-p world
    s1 = rng.binomial(n, p) / n                # a fresh label draw from that world
    a = b = c2 = 0.0
    for t in TIERS:
        r = tier_result(s1, dv.cost, dv, t, 1.0)
        a += TIER_WEIGHT[t] * (s1[IDX, r["sel"]].mean() if r["passed"] else 0.0)
        b += TIER_WEIGHT[t] * (p[IDX, r["sel"]].mean() if r["passed"] else 0.0)
        sd = Cc[f"sel_d_{t}"]
        c2 += TIER_WEIGHT[t] * p[IDX, sd].mean()
    inn.append(a); out.append(b); dep.append(c2)
inn, out, dep = map(np.array, (inn, out, dep))
print(f"  simulated world: sd(p) per model = {np.round(rng.beta(pa,pb).std(0),3)}  "
      f"(EB posterior-mean sd {np.round(PHAT.std(0),3)}, realised-label sd {np.round(dv.score.std(0),3)})")
print(f"  oracle on its own noisy labels  = {inn.mean():.4f}")
print(f"  same allocation on latent p     = {out.mean():.4f}   -> inflation {inn.mean()-out.mean():.4f}")
print(f"  deployed allocation on latent p = {dep.mean():.4f}")
print(f"  simulated recoverable gap       = {out.mean()-dep.mean():.4f}  "
      f"(direct EB estimate on real data: 0.0861)")

# ---------------------------------------------------------------- helpers
def run(mk_s, mk_c, tune=True, grid=np.arange(0.5, 1.301, 0.005)):
    tot_r = tot_e = 0.0; sfs = []
    for t in TIERS:
        if tune:
            best = None; bsf = None
            for sf in grid:
                r = tier_result(mk_s(t), mk_c(t), dv, t, float(sf))
                if r["passed"] and (best is None or r["score"] > best["score"]):
                    best = r; bsf = float(sf)
            r = best; sfs.append(bsf)
        else:
            r = tier_result(mk_s(t), mk_c(t), dv, t, SAFE[t]); sfs.append(SAFE[t])
        e = PHAT[IDX, r["sel"]].mean() if r["passed"] else 0.0
        tot_r += TIER_WEIGHT[t] * r["tier_score"]; tot_e += TIER_WEIGHT[t] * e
    return tot_r, tot_e, sfs

ps = lambda t: P[f"score_{t}"]
pc = lambda t: P[f"cost_{t}"]

# ---------------------------------------------------------------- (b) decision-price multiplier
print("\n=== (b) global decision-price multiplier on one model (decision only; budget accounting unchanged) ===")
print("    NOTE: the multiplier scales the cost used in the utility AND in the bisection accounting,")
print("    so it is a pure re-pricing of the k1 (resp. mid) column in the decision problem.")
print(f"  {'model':6s} {'mult':>5s} {'realised':>9s} {'EB':>7s} {'safety f/b/p':>18s}")
for j, nm in ((1, "mid"), (2, "k1")):
    for mult in (0.6, 0.8, 1.0, 1.25, 1.6, 2.5):
        def mkc(t, j=j, mult=mult):
            C = P[f"cost_{t}"].copy(); C[:, j] *= mult; return C
        r, e, sfs = run(ps, mkc)
        print(f"  {nm:6s} {mult:5.2f} {r:9.4f} {e:7.4f} {'/'.join(f'{x:.3f}' for x in sfs):>18s}")

# ---------------------------------------------------------------- (c) gain shrinkage
print("\n=== (c) per-decision gain shrinkage: g1 *= gamma1 (mid-light), g2 *= gamma2 (k1-mid) ===")
gams = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
print("       gamma2 ->  " + "  ".join(f"{g:6.2f}" for g in gams))
best = None
for g1 in gams:
    row_r = []
    for g2 in gams:
        def mk(t, g1=g1, g2=g2):
            S = P[f"score_{t}"].copy()
            a = S[:, 1] - S[:, 0]; b = S[:, 2] - S[:, 1]
            S[:, 1] = S[:, 0] + g1 * a
            S[:, 2] = S[:, 1] + g2 * b
            return S
        r, e, sfs = run(mk, pc)
        row_r.append(r)
        if best is None or r > best[0]:
            best = (r, e, g1, g2, sfs)
    print(f"  g1={g1:4.2f}   " + "  ".join(f"{v:6.4f}" for v in row_r))
print(f"  best: gamma1={best[2]} gamma2={best[3]} realised={best[0]:.4f} EB={best[1]:.4f} "
      f"safety={'/'.join(f'{x:.3f}' for x in best[4])}")

print("\n  same grid at the DEPLOYED safety (.98/.89/.88), no re-tuning:")
print("       gamma2 ->  " + "  ".join(f"{g:6.2f}" for g in gams))
for g1 in gams:
    row = []
    for g2 in gams:
        def mk(t, g1=g1, g2=g2):
            S = P[f"score_{t}"].copy()
            a = S[:, 1] - S[:, 0]; b = S[:, 2] - S[:, 1]
            S[:, 1] = S[:, 0] + g1 * a; S[:, 2] = S[:, 1] + g2 * b
            return S
        r, e, _ = run(mk, pc, tune=False)
        row.append(r)
    print(f"  g1={g1:4.2f}   " + "  ".join(f"{v:6.4f}" for v in row))
