# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a11 step 1 - post-allocation facts: slack, tail concentration, tier monotonicity.

Everything below is computed from reports/lab/dev_preds_e43.npz (deployed E43
held-out predictions on dev) + the true dev costs/scores from labdata.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, MODEL_IDS, allocate, tier_result

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}   # E43 deployed
n = len(dv)
idx = np.arange(n)

print("=== 0. sanity: deployed E43 constants on dev ===")
tot = 0.0
SEL = {}
for t in TIERS:
    ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
    r = tier_result(ps, pc, dv, t, SAFE[t])
    SEL[t] = r["sel"]
    tot += TIER_WEIGHT[t] * r["tier_score"]
    print(f"  {t:9s} score={r['score']:.4f} true_ratio={r['ratio']:.4f} pass={r['passed']} "
          f"counts={np.bincount(r['sel'], minlength=3)}")
print(f"  FINAL = {tot:.4f}")

print("\n=== 0b. are the three cost matrices identical across tiers? ===")
for a, b in (("fast", "balanced"), ("fast", "premium")):
    d = np.abs(P[f"cost_{a}"] - P[f"cost_{b}"]).max()
    ds = np.abs(P[f"score_{a}"] - P[f"score_{b}"]).max()
    print(f"  max|cost_{a}-cost_{b}| = {d:.3e}   max|score diff| = {ds:.3e}")

print("\n=== 1. leftover budget after bisection (predicted units) ===")
print(f"{'tier':9s} {'cap_pred':>10s} {'used_pred':>10s} {'slack_pred':>10s} {'slack%':>7s} "
      f"{'cheapest_upgrade':>17s} {'n_upg_affordable':>16s}")
for t in TIERS:
    ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
    sel = SEL[t]
    lt = pc[:, 0].sum()
    cap = lt * max(1.0, TIER_MULT[t] * SAFE[t])
    used = pc[idx, sel].sum()
    slack = cap - used
    # cheapest single upgrade available (sel -> sel+1) among items not at k1
    up = np.where(sel < 2)[0]
    dcost = pc[up, sel[up] + 1] - pc[up, sel[up]]
    aff = int((dcost <= slack).sum())
    print(f"{t:9s} {cap:10.4f} {used:10.4f} {slack:10.5f} {100*slack/cap:6.3f}% "
          f"{dcost.min():17.6f} {aff:16d}")

print("\n=== 1b. true-cost headroom actually left on the table ===")
for t in TIERS:
    sel = SEL[t]
    true_lt = dv.cost[:, 0].sum()
    used = dv.cost[idx, sel].sum()
    cap_true = TIER_MULT[t] * true_lt
    print(f"  {t:9s} true_used={used:.3f} true_cap={cap_true:.3f} "
          f"unused={cap_true-used:.3f} ({100*(cap_true-used)/cap_true:.1f}% of cap) "
          f"ratio={used/true_lt:.4f}/{TIER_MULT[t]}")

print("\n=== 2. tail concentration of the realised cost ===")
for t in TIERS:
    sel = SEL[t]
    c = dv.cost[idx, sel]
    tl = dv.cost[:, 0].sum()
    order = np.argsort(-c)
    k1 = sel == 2
    print(f"  {t:9s} n_k1={k1.sum():3d}  k1 share of realised cost="
          f"{c[k1].sum()/c.sum():.3f}")
    for k in (1, 5, 10, 25, 50):
        share = c[order[:k]].sum() / c.sum()
        print(f"      top{k:3d} items = {100*share:5.1f}% of realised cost, "
              f"{c[order[:k]].sum()/tl:.3f} of the ratio budget "
              f"({TIER_MULT[t]:.2f})")

print("\n=== 2b. variance decomposition of the realised ratio (880-bootstrap) ===")
rng = np.random.default_rng(7)
B = 2000
for t in TIERS:
    sel = SEL[t]
    c = dv.cost[idx, sel]
    l = dv.cost[:, 0]
    # bootstrap the ratio with the SELECTION FIXED per item (selection would be
    # re-solved in reality; this isolates the sampling variance of the ratio)
    S = rng.integers(0, n, size=(B, n))
    ratios = c[S].sum(1) / l[S].sum(1)
    # counterfactual: same but with the top-decile-k1-cost items forced to mid
    hi = np.zeros(n, bool)
    k1 = np.where(sel == 2)[0]
    if len(k1):
        thr = np.quantile(dv.cost[k1, 2], 0.9)
        hi[k1[dv.cost[k1, 2] >= thr]] = True
    c2 = c.copy(); c2[hi] = dv.cost[hi, 1]
    r2 = c2[S].sum(1) / l[S].sum(1)
    print(f"  {t:9s} ratio mean={ratios.mean():.4f} sd={ratios.std():.4f} "
          f"p99={np.quantile(ratios,0.99):.4f} | drop top-decile-k1 ({hi.sum()} items): "
          f"mean={r2.mean():.4f} sd={r2.std():.4f} score cost="
          f"{(dv.score[idx,sel].mean()-dv.score[np.where(hi, 1, sel)*0+idx, np.where(hi,1,sel)].mean()):+.4f}")

print("\n=== 3. cross-tier monotonicity of the deployed selection ===")
for a, b in (("fast", "balanced"), ("balanced", "premium"), ("fast", "premium")):
    sa, sb = SEL[a], SEL[b]
    viol = int((sb < sa).sum())
    print(f"  {a:9s} -> {b:9s}: upgrades={int((sb>sa).sum()):4d} same={int((sb==sa).sum()):4d} "
          f"DOWNGRADES={viol:4d}")
    if viol:
        w = np.where(sb < sa)[0]
        print(f"      example items {w[:8].tolist()}  sel_{a}={sa[w[:8]].tolist()} sel_{b}={sb[w[:8]].tolist()}")
        print(f"      mean true score lost by the downgrade = "
              f"{(dv.score[w, sa[w]] - dv.score[w, sb[w]]).mean():+.4f} over {viol} items "
              f"(= {(dv.score[w, sa[w]] - dv.score[w, sb[w]]).sum()/n:+.5f} on the tier mean)")
