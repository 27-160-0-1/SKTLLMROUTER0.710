# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a02-critic #9.  Split the score-side gap into (i) the 9x3 family-mean table
being wrong and (ii) the within-family item ranking being wrong.

Both corrections are applied as modifications of the predicted score matrix, so
the ordinary Lagrangian allocator still enforces the budget -- no infeasible
transplants.  The "oracle" information comes from an INDEPENDENT half of the
generations (exact hypergeometric split, see script #3) and the evaluation uses
the other half, so nothing is double counted.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, TIER_MULT, tier_result
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
fam = np.array([classify_family(t) for t in dv.texts])
fams = sorted(set(fam))
N = len(dv)
ar = np.arange(N)
n = dv.ngen.astype(int)
kk = np.rint(dv.score * n).astype(int)
half = n // 2
SG = np.arange(0.60, 1.401, 0.005)


def split_draw(rng):
    kA = rng.hypergeometric(np.maximum(kk, 0), np.maximum(n - kk, 0), half)
    return kA / half, (kk - kA) / half


def fmean(X):
    Y = np.empty_like(X)
    for f in fams:
        m = fam == f
        Y[m] = X[m].mean(0)
    return Y


VARIANTS = {
    "deployed predictions": lambda S, sA: S,
    "fix the 9x3 family table only": lambda S, sA: S - fmean(S) + fmean(sA),
    "fix within-family residuals only": lambda S, sA: fmean(S) + (sA - fmean(sA)),
    "fix both (= use s_A)": lambda S, sA: sA,
}

rng = np.random.default_rng(11)
NREP = 12
acc = {k: [] for k in VARIANTS}
per_tier = {k: {t: [] for t in TIERS} for k in VARIANTS}
for rep in range(NREP):
    sA, sB = split_draw(rng)
    for name, fn in VARIANTS.items():
        tot = 0.0
        for t in TIERS:
            S = fn(P[f"score_{t}"], sA)
            best = 0.0
            for sf in SG:
                r = tier_result(S, P[f"cost_{t}"], dv, t, float(sf))
                if r["passed"]:
                    best = max(best, sB[ar, r["sel"]].mean())
            per_tier[name][t].append(best)
            tot += TIER_WEIGHT[t] * best
        acc[name].append(tot)

print("evaluated on an independent half-sample s_B; safety oracle-tuned per row;")
print("predicted costs everywhere, so this isolates the SCORE side.\n")
print(f"{'variant':36s} {'fast':>8s} {'balanced':>9s} {'premium':>8s} {'weighted':>9s} {'delta':>8s}")
base = np.mean(acc["deployed predictions"])
for name in VARIANTS:
    pt = [np.mean(per_tier[name][t]) for t in TIERS]
    v = np.mean(acc[name])
    print(f"{name:36s} {pt[0]:8.4f} {pt[1]:9.4f} {pt[2]:8.4f} {v:9.4f} {v-base:+8.4f}")

a = np.mean(acc["fix the 9x3 family table only"]) - base
b = np.mean(acc["fix within-family residuals only"]) - base
c = np.mean(acc["fix both (= use s_A)"]) - base
print(f"\n  family-table error       -> {a:+.4f}  ({a/c*100:.0f}% of the total {c:+.4f})")
print(f"  within-family rank error -> {b:+.4f}  ({b/c*100:.0f}% of the total)")
print(f"  (a+b) - total = {a+b-c:+.4f}  (interaction)")

print("\nhow wrong is the 9x3 family table today?  predicted vs realised family means:")
S = P["score_premium"]
print(f"  {'family':11s} " + "  ".join(f"{m:>21s}" for m in MODEL_IDS))
for f in fams:
    m = fam == f
    print(f"  {f:11s} " + "  ".join(
        f"pred {S[m,j].mean():.3f} true {dv.score[m,j].mean():.3f}" for j in range(3)))
