# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a02-critic #3.  An HONEST exchange rate, and an honest label-noise budget.

The BRIEF's headline number ("blend the score head toward the truth by lam ->
final 0.7063 / 0.7436 / 0.7702") blends toward the REALISED score and then
scores on the REALISED score.  That is self-referential: part of the gain is
the allocator exploiting the label's own sampling noise, which no prompt-only
model can ever reproduce.

Fix: score = k/n with n = num_generations in {2,4}.  Conditional on k, splitting
the n generations into two halves gives k_A ~ Hypergeometric(n, k, n/2) and
k_B = k - k_A.  That reproduces EXACTLY the joint law of two independent
Binomial(n/2, p) half-samples.  So s_A and s_B are conditionally independent
given the latent p.  We may therefore
  * blend the predictor toward s_A  (information about p, no shared noise), and
  * evaluate the resulting allocation on s_B  (unbiased for the mean p of the
    selected arms).
Nothing is double-counted.  We also get split-half reliability for free, which
converts every reported corr-with-realised-score into a corr-with-p.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, TIER_MULT, tier_result

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
N = len(dv)
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 40

n = dv.ngen.astype(int)
k = np.rint(dv.score * n).astype(int)
half = n // 2


def split_draw(rng):
    kA = rng.hypergeometric(np.maximum(k, 0), np.maximum(n - k, 0), half)
    sA = kA / half
    sB = (k - kA) / half
    return sA, sB


print("=" * 96)
print("A.  How noisy is the label really?  (split-half reliability of the score)")
print("=" * 96)
rng = np.random.default_rng(0)
rel = np.zeros((NREP, 3))
for r in range(NREP):
    sA, sB = split_draw(rng)
    for j in range(3):
        rel[r, j] = np.corrcoef(sA[:, j], sB[:, j])[0, 1]
rel_half = rel.mean(0)
rel_full = 2 * rel_half / (1 + rel_half)          # Spearman-Brown, n vs n/2 draws
print(f"{'model':12s} {'rel(half=n/2)':>14s} {'rel(full=n)':>12s}  {'var(p)/var(s)':>13s}")
for j, m in enumerate(MODEL_IDS):
    print(f"{m:12s} {rel_half[j]:14.3f} {rel_full[j]:12.3f}")
print("\n=> reported corr(pred, realised s) rescaled to corr(pred, latent p):")
for t in ("fast",):
    for j, m in enumerate(MODEL_IDS):
        c = np.corrcoef(P[f"score_{t}"][:, j], dv.score[:, j])[0, 1]
        print(f"   {m:12s} corr_vs_s={c:.3f}   corr_vs_p={c/np.sqrt(rel_full[j]):.3f}"
              f"   (max attainable corr_vs_s for a perfect p-model = {np.sqrt(rel_full[j]):.3f})")

print("\nvariance decomposition (per model): var(s) = var(p) + E[p(1-p)]/n")
for j, m in enumerate(MODEL_IDS):
    vs = dv.score[:, j].var()
    vp = vs * rel_full[j]
    print(f"   {m:12s} var(s)={vs:.4f}  var(p)={vp:.4f}  noise var={vs-vp:.4f} "
          f"({(1-rel_full[j])*100:.1f}% of the observed score variance is sampling noise)")

print()
print("=" * 96)
print("B.  Honest oracle ceilings (allocate on one half, evaluate on the OTHER half)")
print("=" * 96)


def wsum(alloc_s, eval_s, cost_for_alloc, safety=None, grid=None):
    tot = 0.0
    parts = []
    for t in TIERS:
        best = None
        sgrid = [1.0] if grid is None else grid
        for sf in sgrid:
            r = tier_result(alloc_s, cost_for_alloc, dv, t, float(sf))
            if not r["passed"]:
                continue
            v = eval_s[np.arange(N), r["sel"]].mean()
            if best is None or v > best[0]:
                best = (v, float(sf))
        if best is None:
            best = (0.0, 1.0)
        tot += TIER_WEIGHT[t] * best[0]
        parts.append(f"{t[:4]}={best[0]:.4f}")
    return tot, " ".join(parts)


rng = np.random.default_rng(1)
acc = {"realised-oracle (BRIEF 0.8034 style)": [], "alloc on s_A, eval on s_B": [],
       "alloc on full s, eval on s_B": [], "alloc on pred, eval on s_B": [],
       "alloc on pred, eval on full s": []}
for r in range(NREP):
    sA, sB = split_draw(rng)
    acc["realised-oracle (BRIEF 0.8034 style)"].append(wsum(dv.score, dv.score, dv.cost)[0])
    acc["alloc on s_A, eval on s_B"].append(wsum(sA, sB, dv.cost)[0])
    acc["alloc on full s, eval on s_B"].append(wsum(dv.score, sB, dv.cost)[0])
    acc["alloc on pred, eval on s_B"].append(
        sum(TIER_WEIGHT[t] * (lambda rr: sB[np.arange(N), rr["sel"]].mean() if rr["passed"] else 0.0)(
            tier_result(P[f"score_{t}"], dv.cost, dv, t, 1.0)) for t in TIERS))
    acc["alloc on pred, eval on full s"].append(
        sum(TIER_WEIGHT[t] * tier_result(P[f"score_{t}"], dv.cost, dv, t, 1.0)["tier_score"] for t in TIERS))
for kk, v in acc.items():
    print(f"  {kk:42s} {np.mean(v):.4f} +- {np.std(v)/np.sqrt(len(v)):.4f}")
print("  (all use TRUE cost and safety 1.0 so only the score side differs)")

print()
print("=" * 96)
print("C.  HONEST exchange rate.  predictor = (1-lam)*deployed + lam*s_A ; evaluate on s_B")
print("    contaminated version (BRIEF table) = blend toward full s, evaluate on full s")
print("=" * 96)
SG = np.arange(0.60, 1.301, 0.01)


def cal(t):
    C = P[f"cost_{t}"].copy()
    return C * (dv.cost.sum(0) / C.sum(0))[None, :]


COSTM = {"as-is": lambda t: P[f"cost_{t}"], "calibrated": cal, "true": lambda t: dv.cost}
LAMS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
rng = np.random.default_rng(2)
draws = [split_draw(rng) for _ in range(max(6, NREP // 4))]

print(f"{'cost':11s} {'lam':>5s} {'HONEST final':>13s} {'corr_p(l/m/k)':>16s} "
      f"{'CONTAMINATED':>13s} {'corr_s(l/m/k)':>16s}")
rows = []
for cname, cf in COSTM.items():
    for lam in LAMS:
        hon = []
        cps = np.zeros(3)
        for (sA, sB) in draws:
            tot = 0.0
            for t in TIERS:
                S = (1 - lam) * P[f"score_{t}"] + lam * sA
                best = 0.0
                for sf in SG:
                    r = tier_result(S, cf(t), dv, t, float(sf))
                    if r["passed"]:
                        v = sB[np.arange(N), r["sel"]].mean()
                        best = max(best, v)
                tot += TIER_WEIGHT[t] * best
            hon.append(tot)
            Sf = (1 - lam) * P["score_fast"] + lam * sA
            for j in range(3):
                cps[j] += np.corrcoef(Sf[:, j], sB[:, j])[0, 1] / np.sqrt(rel_half[j]) / len(draws)
        # contaminated (BRIEF-style)
        tot_c = 0.0
        cs = np.zeros(3)
        for t in TIERS:
            S = (1 - lam) * P[f"score_{t}"] + lam * dv.score
            best = 0.0
            for sf in SG:
                r = tier_result(S, cf(t), dv, t, float(sf))
                if r["passed"]:
                    best = max(best, r["score"])
            tot_c += TIER_WEIGHT[t] * best
        Sf = (1 - lam) * P["score_fast"] + lam * dv.score
        for j in range(3):
            cs[j] = np.corrcoef(Sf[:, j], dv.score[:, j])[0, 1]
        print(f"{cname:11s} {lam:5.2f} {np.mean(hon):13.4f} "
              f"{'/'.join(f'{c:.2f}' for c in cps):>16s} {tot_c:13.4f} "
              f"{'/'.join(f'{c:.2f}' for c in cs):>16s}")
        rows.append((cname, lam, np.mean(hon), tot_c, cps.copy()))

print("\nD.  Marginal exchange rate near the operating point (cost model = as-is):")
sub = [r for r in rows if r[0] == "as-is"]
for i in range(1, len(sub)):
    d_final_h = sub[i][2] - sub[0][2]
    d_final_c = sub[i][3] - sub[0][3]
    d_corr = sub[i][4].mean() - sub[0][4].mean()
    if d_corr > 1e-9:
        print(f"  lam={sub[i][1]:.2f}  d(corr_p)={d_corr:+.3f}  honest d(final)={d_final_h:+.4f}"
              f"  -> {d_final_h/d_corr:+.3f} final per unit corr   |  contaminated "
              f"{d_final_c:+.4f} -> {d_final_c/(sub[i][4].mean()-sub[0][4].mean()):+.3f}")
