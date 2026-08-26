# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a02-critic #7.  Is the mid-vs-light upgrade gain -- the quantity the fast tier
(weight 0.4) is entirely built on -- even learnable?

Same exact half-split of the generations as script #3, so every reliability is
measured, not assumed.
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
n = dv.ngen.astype(int)
k = np.rint(dv.score * n).astype(int)
half = n // 2
NREP = 60


def split_draw(rng):
    kA = rng.hypergeometric(np.maximum(k, 0), np.maximum(n - k, 0), half)
    return kA / half, (k - kA) / half


rng = np.random.default_rng(0)
PAIRS = [(0, 1, "mid - light"), (1, 2, "k1 - mid"), (0, 2, "k1 - light")]
rel = {p[2]: [] for p in PAIRS}
for r in range(NREP):
    sA, sB = split_draw(rng)
    for a, b, lab in PAIRS:
        gA = sA[:, b] - sA[:, a]
        gB = sB[:, b] - sB[:, a]
        rel[lab].append(np.corrcoef(gA, gB)[0, 1])

print("=" * 96)
print("Reliability of the UPGRADE GAIN label (split-half of the generations)")
print("=" * 96)
print(f"{'pair':12s} {'rel(half)':>10s} {'rel(full)':>10s} {'var(dp)':>9s} {'var(ds)':>9s} "
      f"{'max corr':>9s} {'our corr_s':>11s} {'our corr_dp':>12s} {'% of ceiling':>13s}")
for a, b, lab in PAIRS:
    rh = float(np.mean(rel[lab]))
    rf = 2 * rh / (1 + rh)
    gt = dv.score[:, b] - dv.score[:, a]
    gp = P["score_premium"][:, b] - P["score_premium"][:, a]
    cs = np.corrcoef(gp, gt)[0, 1]
    cdp = cs / np.sqrt(rf)
    print(f"{lab:12s} {rh:10.3f} {rf:10.3f} {gt.var()*rf:9.4f} {gt.var():9.4f} "
          f"{np.sqrt(rf):9.3f} {cs:11.3f} {cdp:12.3f} {cdp*100:12.1f}%")

print("\n=> 'max corr' is the highest corr with the REALISED gain that a perfect")
print("   knower-of-p could achieve.  'our corr_dp' is our correlation with the")
print("   latent gain after removing label noise.")

print()
print("=" * 96)
print("Per-tier honest ceilings: allocate on an independent half-sample, evaluate on")
print("the other half.  TRUE cost, safety chosen to just fit the cap.")
print("=" * 96)
SG = np.arange(0.60, 1.301, 0.005)
rng = np.random.default_rng(3)
draws = [split_draw(rng) for _ in range(12)]
rows = {}
for name, mk in (("deployed prediction", lambda sA: None),
                 ("independent half-sample s_A", lambda sA: sA),
                 ("full realised s (CONTAMINATED)", lambda sA: dv.score)):
    per = {t: [] for t in TIERS}
    for (sA, sB) in draws:
        alloc_s = mk(sA)
        for t in TIERS:
            S = P[f"score_{t}"] if alloc_s is None else alloc_s
            best = 0.0
            for sf in SG:
                r = tier_result(S, dv.cost, dv, t, float(sf))
                if r["passed"]:
                    best = max(best, sB[np.arange(N), r["sel"]].mean())
            per[t].append(best)
    rows[name] = {t: float(np.mean(v)) for t, v in per.items()}
    w = sum(TIER_WEIGHT[t] * rows[name][t] for t in TIERS)
    print(f"  {name:34s} " + "  ".join(f"{t[:4]}={rows[name][t]:.4f}" for t in TIERS)
          + f"   weighted={w:.4f}")
print("\n  headroom still available per tier (independent-half oracle minus deployed):")
for t in TIERS:
    d = rows["independent half-sample s_A"][t] - rows["deployed prediction"][t]
    print(f"    {t:9s} {d:+.4f}   (weight {TIER_WEIGHT[t]} -> {TIER_WEIGHT[t]*d:+.4f} of the final)")

print()
print("=" * 96)
print("What the fast tier is actually deciding")
print("=" * 96)
r = tier_result(P["score_fast"], P["cost_fast"], dv, "fast", 0.98)
sel = r["sel"]
print(f"  fast tier picks: light {int((sel==0).sum())}, mid {int((sel==1).sum())}, k1 {int((sel==2).sum())}")
g = dv.score[:, 1] - dv.score[:, 0]
print(f"  realised mid-light gain: mean over UPGRADED items {g[sel==1].mean():+.4f}, "
      f"over items left on light {g[sel==0].mean():+.4f}  (all items {g.mean():+.4f})")
gp = P["score_fast"][:, 1] - P["score_fast"][:, 0]
print(f"  predicted gain: upgraded {gp[sel==1].mean():+.4f}, not upgraded {gp[sel==0].mean():+.4f}")
best = np.argsort(-(g / np.maximum(dv.cost[:, 1] - dv.cost[:, 0], 1e-9)))
print(f"  fraction of the TRUE top-{int((sel==1).sum())} efficiency items that we picked: "
      f"{np.mean(np.isin(best[:int((sel==1).sum())], np.where(sel==1)[0])):.3f}")
print(f"  a random selector of the same size would get "
      f"{int((sel==1).sum())/N:.3f}")
