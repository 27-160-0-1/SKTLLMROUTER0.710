# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a13 / provider view - part 6: what the cost policy and the budget ladder mean.

- output rate is EXACTLY 4x input rate for all three models => cost factorises
  into (model price) x (billable tokens = in + 4*out).  Verify numerically.
- the budget ladder 1.25 / 2.0 / 4.0 vs the natural reference points
  (all-light = 1, all-mid, all-k1).
- what each budget actually buys under an oracle.
- num_generations == 4  <=>  AIME or GSM8K  (exact identity check).
Run: .venv/Scripts/python.exe experiments/lab/a13_provider_06_policy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
sys.path.insert(0, str(ROOT / "src"))

import labdata  # noqa: E402
from labdata import TIERS, TIER_MULT, TIER_WEIGHT, MODEL_IDS, RATES  # noqa: E402

tr, dv = labdata.load_all()

print("=" * 88)
print("A. cost factorises: cost_j = price_j * (in_j + 4*out_j) / 1e6 ?")
print("=" * 88)
for sp in (tr, dv):
    for j, m in enumerate(MODEL_IDS):
        pr = RATES[m][1]
        billable = sp.itok[:, j] + 4.0 * sp.otok[:, j]
        recon = pr * billable / 1e6
        err = np.abs(recon - sp.cost[:, j]).max()
        print(f"  {sp.name:5s} {m:11s} price={pr:6.3f}  max |recon - cost| = {err:.3e}")
print("  => the three models differ ONLY by a scalar price; the token mix is the "
      "same functional form.")
print(f"  price ratios light:mid:k1 = 1 : {RATES['ax31'][1]:.3f} : {RATES['axk1-think'][1]:.3f}"
      f"   (k1/mid = {RATES['axk1-think'][1]/RATES['ax31'][1]:.4f})")
print("  output/input multiplier per model:",
      {m: RATES[m][2] / RATES[m][1] for m in MODEL_IDS})

print()
print("=" * 88)
print("B. share of each model's cost that is input vs output; billable-token view")
print("=" * 88)
for sp in (tr, dv):
    print(f"-- {sp.name}")
    for j, m in enumerate(MODEL_IDS):
        i = sp.itok[:, j].sum()
        o = 4.0 * sp.otok[:, j].sum()
        print(f"   {m:11s} input share of billable tokens {i/(i+o)*100:5.1f}%  "
              f"output share {o/(i+o)*100:5.1f}%  "
              f"mean out/in ratio {(sp.otok[:,j]/sp.itok[:,j]).mean():6.3f}")

print()
print("=" * 88)
print("C. the ladder: reference cost ratios vs the three budgets")
print("=" * 88)
for sp in (tr, dv):
    L = sp.cost[:, 0].sum()
    print(f"-- {sp.name}: all-light 1.000  all-mid {sp.cost[:,1].sum()/L:.3f}  "
          f"all-k1 {sp.cost[:,2].sum()/L:.3f}")
    for t in TIERS:
        b = TIER_MULT[t]
        # how many items can be upgraded light->mid / light->k1 with the slack
        slack = (b - 1.0) * L
        dmid = np.sort(sp.cost[:, 1] - sp.cost[:, 0])
        dk1 = np.sort(sp.cost[:, 2] - sp.cost[:, 0])
        nmid = int(np.searchsorted(np.cumsum(dmid), slack))
        nk1 = int(np.searchsorted(np.cumsum(dk1), slack))
        print(f"   {t:9s} budget {b:4.2f}x  slack {slack/L:5.2f}x  "
              f"=> cheapest-first capacity: {nmid:5d} mid upgrades ({nmid/len(sp)*100:5.1f}%) "
              f"or {nk1:4d} k1 upgrades ({nk1/len(sp)*100:5.1f}%)")
    print(f"   all-mid busts balanced? {sp.cost[:,1].sum()/L > 2.0}   "
          f"all-mid busts premium?  {sp.cost[:,1].sum()/L > 4.0}")

print()
print("=" * 88)
print("D. oracle allocation mix per tier (true score, true cost, safety 1.0)")
print("=" * 88)
for sp in (tr, dv):
    print(f"-- {sp.name}")
    for t in TIERS:
        r = labdata.oracle_tier(sp, t)
        sel = r["sel"]
        mix = np.bincount(sel, minlength=3) / len(sel)
        print(f"   {t:9s} score {r['score']:.4f} ratio {r['ratio']:.3f}  "
              f"mix light/mid/k1 = {mix[0]*100:5.1f}% {mix[1]*100:5.1f}% {mix[2]*100:5.1f}%")

print()
print("=" * 88)
print("E. num_generations == 4  <=>  AIME or GSM8K ?")
print("=" * 88)
dm = json.loads((ROOT / "data/sources/deepmind-mathematics-selection.v1.json").read_text(encoding="utf-8"))
dmids = {r["episode_id"] for rows in dm["splits"].values() for r in rows}
aimeids = set()
for sp in ("train", "dev"):
    sel = json.loads((ROOT / f"data/{sp}/aime-selection.json").read_text(encoding="utf-8"))
    aimeids |= {e["episode_id"] for e in sel["episodes"]}
n4 = 0
for sp in (tr, dv):
    m = sp.ngen[:, 0] == 4
    n4 += int(m.sum())
    ids = set(np.array(sp.episode_ids)[m])
    print(f"  {sp.name}: ngen=4 n={m.sum()}  of which confirmed AIME {len(ids & aimeids)}  "
          f"confirmed dmmath {len(ids & dmids)}")
print(f"  total ngen=4 across public data = {n4}")
print(f"  confirmed AIME episodes = {len(aimeids)}")
print(f"  EXPERIMENT_LOG E41 records that build_pool.py --verify reproduced 333 GSM8K "
      f"test prompts inside the public data")
print(f"  36 AIME + 333 GSM8K = {36+333}   vs   observed ngen=4 count = {n4}")

print()
print("=" * 88)
print("F. label noise implied by ngen: Var(score|p) = p(1-p)/ngen")
print("=" * 88)
for sp in (tr, dv):
    for g in (2, 4):
        m = sp.ngen[:, 0] == g
        print(f"  {sp.name} ngen={g} n={m.sum():4d}: "
              f"share of scores in the interior (0<s<1) "
              f"light {np.mean((sp.score[m,0]>0)&(sp.score[m,0]<1)):.3f} "
              f"mid {np.mean((sp.score[m,1]>0)&(sp.score[m,1]<1)):.3f} "
              f"k1 {np.mean((sp.score[m,2]>0)&(sp.score[m,2]<1)):.3f}")
# what an ordinal head is actually being asked for at each threshold
print("  P(s>=0.25) means P(>=1 of 2 correct) for ngen=2 items and "
      "P(>=1 of 4) for ngen=4 items -> the same head fits two different "
      "functions of the latent p.")
