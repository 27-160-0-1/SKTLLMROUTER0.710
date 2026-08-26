# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a13 / provider view - part 4: operator-audit robustness + truncation oracle.

A  Permutation audit.  The rules say the operator may re-run the router with
   shuffled episode_ids and input order and check that the decision for the same
   prompt+tier is unchanged.  Our allocator sums float costs over the batch and
   bisects a Lagrange multiplier, so float-addition order could in principle flip
   an item near the multiplier boundary.  Measure it.

B  Value of an oracle 'k1 will blow up' flag: how much final score does perfect
   knowledge of the long-output / truncated tail buy?

C  Is the regex family 'gsm8k_or_other' two populations?  (ngen=4 vs 2)  and does
   the deployed pipeline already separate them?
Run: .venv/Scripts/python.exe experiments/lab/a13_provider_04_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
sys.path.insert(0, str(ROOT / "src"))

import labdata  # noqa: E402
from labdata import TIERS, TIER_WEIGHT, allocate, tier_result  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

dv = labdata.load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
fam = np.array([classify_family(t) for t in dv.texts])


def final(ps_fn, pc_fn, safe=SAFE, tag=""):
    tot, parts = 0.0, []
    for t in TIERS:
        r = tier_result(ps_fn(t), pc_fn(t), dv, t, safe[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]} {r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else ' BUST'}")
    if tag:
        print(f"{tag:52s} {tot:.4f}   " + "  ".join(parts))
    return tot


ps = lambda t: P[f"score_{t}"]
pc = lambda t: P[f"cost_{t}"]

print("=" * 92)
print("A. permutation-invariance audit of the deployed allocator")
print("=" * 92)
rng = np.random.default_rng(0)
for t in TIERS:
    S, C = P[f"score_{t}"], P[f"cost_{t}"]
    base = allocate(S, C, dv.cost, labdata.TIER_MULT[t], SAFE[t])
    flips_tot = 0
    worst = 0
    for trial in range(20):
        perm = rng.permutation(len(dv))
        inv = np.argsort(perm)
        selp = allocate(S[perm], C[perm], dv.cost[perm], labdata.TIER_MULT[t], SAFE[t])
        back = selp[inv]
        d = int((back != base).sum())
        flips_tot += d
        worst = max(worst, d)
    print(f"  {t:9s} 20 random permutations: total flipped decisions {flips_tot}, "
          f"worst single permutation {worst} / {len(dv)}")
    # also: reversed order (adversarial for float summation)
    perm = np.arange(len(dv))[::-1]
    selp = allocate(S[perm], C[perm], dv.cost[perm], labdata.TIER_MULT[t], SAFE[t])
    print(f"            reversed order: {int((selp[::-1] != base).sum())} flips")

print()
print("=" * 92)
print("B. oracle value of a 'k1 blows up' flag")
print("=" * 92)
CAP = 32768
opg = dv.otok[:, 2] / dv.ngen[:, 2]
ctx = (dv.itok[:, 2] + dv.otok[:, 2]) / dv.ngen[:, 2]
base = final(ps, pc, tag="deployed E43 (safety .98/.87/.85)")

for name, mask in [
    ("truncated (ctx >= 0.99 cap)", ctx >= CAP * 0.99),
    ("out/gen >= 8192", opg >= 8192),
    ("out/gen >= 4096", opg >= 4096),
    ("out/gen >= 2048", opg >= 2048),
    ("k1 cost decile 10 (true)", dv.cost[:, 2] >= np.quantile(dv.cost[:, 2], 0.9)),
]:
    n = int(mask.sum())

    def mk_c(t, m=mask):
        C = P[f"cost_{t}"].copy()
        C[m, 2] = dv.cost[m, 2]
        return C

    def mk_s(t, m=mask):
        S = P[f"score_{t}"].copy()
        S[m, 2] = dv.score[m, 2]
        return S

    def mk_ban(t, m=mask):
        S = P[f"score_{t}"].copy()
        S[m, 2] = -1e9
        return S

    final(ps, mk_c, tag=f"  true k1 COST on {name} (n={n})")
    final(mk_s, pc, tag=f"  true k1 SCORE on {name} (n={n})")
    final(mk_s, mk_c, tag=f"  true k1 SCORE+COST on {name} (n={n})")
    final(mk_ban, pc, tag=f"  ban k1 on {name} (n={n})")

print()
print("=" * 92)
print("C. 'gsm8k_or_other' is two populations (ngen 4 vs 2)")
print("=" * 92)
g = fam == "gsm8k_or_other"
g4 = g & (dv.ngen[:, 0] == 4)
g2 = g & (dv.ngen[:, 0] == 2)
print(f"  n4={g4.sum()} n2={g2.sum()}")
print(f"  {'':22s} {'true light':>10s} {'true mid':>9s} {'true k1':>8s} "
      f"{'pred light':>11s} {'pred mid':>9s} {'pred k1':>8s}")
for nm, m in (("ngen=4 (true GSM8K)", g4), ("ngen=2 (other)", g2)):
    S = P["score_fast"]
    print(f"  {nm:22s} {dv.score[m,0].mean():10.4f} {dv.score[m,1].mean():9.4f} "
          f"{dv.score[m,2].mean():8.4f} {S[m,0].mean():11.4f} {S[m,1].mean():9.4f} "
          f"{S[m,2].mean():8.4f}")
C = P["cost_fast"]
for nm, m in (("ngen=4", g4), ("ngen=2", g2)):
    print(f"  {nm:22s} true k1 cost {dv.cost[m,2].mean():.5f}  "
          f"pred k1 cost {C[m,2].mean():.5f}  ratio {C[m,2].mean()/dv.cost[m,2].mean():.3f}"
          f"  true k1/light {(dv.cost[m,2]/dv.cost[m,0]).mean():7.1f}")
# how separable by cheap text features?
ln = np.array([len(t) for t in dv.texts], float)
print(f"  prompt chars: ngen4 median {np.median(ln[g4]):.0f}  ngen2 median {np.median(ln[g2]):.0f}")
has_q = np.array([("?" in t) for t in dv.texts])
print(f"  contains '?': ngen4 {has_q[g4].mean():.2f}  ngen2 {has_q[g2].mean():.2f}")
# residual: does the deployed light-score prediction already know?
resid4 = dv.score[g4, 0] - P["score_fast"][g4, 0]
resid2 = dv.score[g2, 0] - P["score_fast"][g2, 0]
print(f"  light score residual (true-pred): ngen4 {resid4.mean():+.4f}  ngen2 {resid2.mean():+.4f}")
print(f"  same for k1 logcost: ngen4 "
      f"{np.mean(np.log(dv.cost[g4,2])-np.log(P['cost_fast'][g4,2])):+.4f}  ngen2 "
      f"{np.mean(np.log(dv.cost[g2,2])-np.log(P['cost_fast'][g2,2])):+.4f}")
