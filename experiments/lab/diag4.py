# SPDX-License-Identifier: Apache-2.0
"""How high can a prompt-only router go?  Estimate the noise-free ceiling.

The observed score is k/n with n = num_generations, so the realised score is a
noisy view of the latent per-item success probability p.  The published
"oracle" (allocate on the realised score) exploits that noise and is therefore
optimistic.  Here we (1) fit a per-(family, model) Beta prior by moment
matching, (2) draw p from the Beta-Binomial posterior of each item, and
(3) allocate on p and *evaluate on p* -- the ceiling for a router that knows
E[score] exactly.  We also report duplicate-prompt agreement as a direct,
assumption-free noise probe.
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, tier_result
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

tr, dv = load_split("train"), load_split("dev")

# ---- duplicate prompts across / within splits -> independent draws of the same p
print("=== duplicate prompts ===")
allt = tr.texts + dv.texts
alls = np.vstack([tr.score, dv.score]); alln = np.vstack([tr.ngen, dv.ngen])
pos = collections.defaultdict(list)
for i, t in enumerate(allt):
    pos[t].append(i)
dups = [v for v in pos.values() if len(v) > 1]
print(f"  duplicate prompt groups: {len(dups)} covering {sum(len(v) for v in dups)} episodes")
if dups:
    d0 = np.array([v[0] for v in dups]); d1 = np.array([v[1] for v in dups])
    for j, m in enumerate(MODEL_IDS):
        a, b = alls[d0, j], alls[d1, j]
        print(f"    {m:11s} agree={np.mean(a==b):.3f} corr={np.corrcoef(a,b)[0,1]:.3f} "
              f"mean|a-b|={np.abs(a-b).mean():.3f}")

# ---- Beta prior per (family, model) by moment matching on k/n
def ceiling(sp):
    fam = np.array([classify_family(t) for t in sp.texts])
    n = sp.ngen.astype(int)
    k = np.rint(sp.score * n).astype(int)
    A = np.ones_like(sp.score); B = np.ones_like(sp.score)
    for f_ in set(fam):
        msk = fam == f_
        for j in range(3):
            x = sp.score[msk, j]
            mu = x.mean(); var = x.var()
            nn = n[msk, j].mean()
            # var(k/n) = var(p) + E[p(1-p)]/n  ->  var(p) = var - (mu-mu^2-var(p))/n
            vp = (var - (mu - mu * mu) / nn) / (1 - 1.0 / nn) if nn > 1 else var
            vp = float(np.clip(vp, 1e-4, mu * (1 - mu) - 1e-4)) if 0 < mu < 1 else 1e-4
            if not (0 < mu < 1):
                A[msk, j], B[msk, j] = 1.0, 1.0
                continue
            c = mu * (1 - mu) / vp - 1
            A[msk, j], B[msk, j] = max(c * mu, 0.05), max(c * (1 - mu), 0.05)
    rng = np.random.default_rng(0)
    post_a, post_b = A + k, B + (n - k)
    tot = []
    for _ in range(24):
        p = rng.beta(post_a, post_b)
        s = 0.0
        for t in TIERS:
            r = tier_result(p, sp.cost, sp, t, 1.0)      # allocate on p
            val = p[np.arange(len(sp)), r["sel"]].mean()  # evaluate on p (noise-free)
            s += TIER_WEIGHT[t] * (val if r["passed"] else 0.0)
        tot.append(s)
    # also: posterior-mean p, which is what a perfect prompt-only model could
    # at best approach if the prompt determined p exactly
    pm = post_a / (post_a + post_b)
    s_pm_real = sum(TIER_WEIGHT[t] * tier_result(pm, sp.cost, sp, t, 1.0)["tier_score"] for t in TIERS)
    s_true = sum(TIER_WEIGHT[t] * tier_result(sp.score, sp.cost, sp, t, 1.0)["tier_score"] for t in TIERS)
    print(f"  {sp.name}: realised-score oracle = {s_true:.4f}")
    print(f"  {sp.name}: posterior-mean-p allocation, scored on realised s = {s_pm_real:.4f}")
    print(f"  {sp.name}: know-p-exactly ceiling (alloc & eval on p) = {np.mean(tot):.4f} +- {np.std(tot):.4f}")

print("\n=== noise-free ceiling ===")
ceiling(dv)
ceiling(tr)
