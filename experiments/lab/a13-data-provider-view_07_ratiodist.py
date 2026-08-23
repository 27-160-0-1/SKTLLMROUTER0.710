# SPDX-License-Identifier: Apache-2.0
"""a13 / provider view - part 7: sanity-check the bust probabilities of part 3.

Part 3 reports much higher bust rates at the deployed safety triple than E39/E43
did.  E39/E43 bootstrapped the 2,640-item CV predictions; part 3 bootstraps the
880-item honest held-out prediction set.  Measure the realised-ratio distribution
directly so the difference is explained rather than asserted.
Run: .venv/Scripts/python.exe experiments/lab/a13_provider_07_ratiodist.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
sys.path.insert(0, str(ROOT / "src"))

import labdata  # noqa: E402
from labdata import TIERS, TIER_MULT, allocate, tier_result  # noqa: E402

dv = labdata.load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}

print("=" * 88)
print("A. point estimate on the real dev set (no resampling)")
print("=" * 88)
for t in TIERS:
    r = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, SAFE[t])
    sel = r["sel"]
    mix = np.bincount(sel, minlength=3)
    cost_by = [dv.cost[sel == j, j].sum() for j in range(3)]
    L = dv.cost[:, 0].sum()
    print(f"  {t:9s} ratio {r['ratio']:.4f} / cap {TIER_MULT[t]:.2f}  "
          f"margin {(TIER_MULT[t]-r['ratio'])/TIER_MULT[t]*100:5.1f}%  "
          f"mix {mix}  cost share light/mid/k1 = "
          f"{cost_by[0]/L:.2f}/{cost_by[1]/L:.2f}/{cost_by[2]/L:.2f}")
    k1 = np.where(sel == 2)[0]
    if len(k1):
        c = np.sort(dv.cost[k1, 2])[::-1]
        print(f"            k1 items {len(k1)}: top-5 true costs {np.round(c[:5],4)} "
              f"= {c[:5].sum()/L:.2f}x light-total; "
              f"top-1 alone {c[0]/L:.3f}x")

print()
print("=" * 88)
print("B. bootstrap distribution of the realised ratio (deployed safety)")
print("=" * 88)


def boot_ratios(tier, n, nboot=2000, seed=7):
    ps, pc = P[f"score_{tier}"], P[f"cost_{tier}"]
    mult, safety = TIER_MULT[tier], SAFE[tier]
    rng = np.random.default_rng(seed)
    out = np.empty(nboot)
    for b in range(nboot):
        ix = rng.integers(0, len(dv), n)
        sel = allocate(ps[ix], pc[ix], dv.cost[ix], mult, safety)
        idx = np.arange(n)
        out[b] = dv.cost[ix][idx, sel].sum() / dv.cost[ix][:, 0].sum()
    return out


for t in TIERS:
    print(f"-- {t} (cap {TIER_MULT[t]})")
    for n in (440, 880, 1760, 2640):
        r = boot_ratios(t, n)
        print(f"   n={n:5d}  mean {r.mean():.4f} sd {r.std():.4f} "
              f"p50 {np.percentile(r,50):.4f} p95 {np.percentile(r,95):.4f} "
              f"p99 {np.percentile(r,99):.4f}  bust {np.mean(r>TIER_MULT[t])*100:5.1f}%")

print()
print("=" * 88)
print("C. where the variance comes from: leave-one-out influence on the premium ratio")
print("=" * 88)
t = "premium"
r = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, SAFE[t])
sel = r["sel"]
L = dv.cost[:, 0].sum()
sel_cost = dv.cost[np.arange(len(dv)), sel]
# influence of duplicating one item (what a bootstrap draw does)
infl = (sel_cost - r["ratio"] * dv.cost[:, 0]) / L
o = np.argsort(-np.abs(infl))[:10]
print("  top-10 items by |d ratio / d duplicate|:")
for i in o:
    print(f"    idx {i:4d} sel={sel[i]} true k1 cost {dv.cost[i,2]:.4f} "
          f"light {dv.cost[i,0]:.5f} influence {infl[i]:+.4f}")
print(f"  sd of the ratio implied by iid resampling: "
      f"{np.sqrt(np.sum(infl**2)/len(dv)):.4f} (n=880) -- compare to bootstrap sd above")
