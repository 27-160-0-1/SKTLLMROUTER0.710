# SPDX-License-Identifier: Apache-2.0
"""a04 / step 1 -- geometry of the Lagrangian decision boundary on dev.

Questions:
  * what is the optimal penalty lambda* per tier, and what does the utility
    margin distribution look like?
  * how many items sit within epsilon of the boundary, and how much of the
    final score is decided there?
  * what is the score cost of getting the near-boundary items wrong
    (band oracle / band adversary)?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result, MODEL_IDS

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv)
IDX = np.arange(n)

SAFE_HOLDOUT = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}   # E43 held-out constants
SAFE_DIAG = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}      # diag1 constants


def allocate_pen(ps, pc, mult, safety):
    """Same bisection as labdata.allocate but also returns the final penalty."""
    lt = pc[:, 0].sum()
    cap = lt * max(1.0, mult * safety)

    def choose(pen):
        util = ps - pen * pc / lt
        sel = util.argmax(axis=1)
        return sel, pc[IDX[: len(sel)], sel].sum()

    sel, total = choose(0.0)
    pen = 0.0
    if total > cap:
        low, high = 0.0, 1.0
        sel, total = choose(high)
        while total > cap and high < 2 ** 60:
            low, high = high, high * 2.0
            sel, total = choose(high)
        pen = high
        for _ in range(40):
            mid = (low + high) / 2.0
            cand, ct = choose(mid)
            if ct <= cap:
                high, sel, total, pen = mid, cand, ct, mid
            else:
                low = mid
    if total > cap:
        sel = np.zeros(len(ps), dtype=int)
        pen = np.inf
    return sel, pen, total, cap


def margins(ps, pc, pen, lt):
    """utility margin of the chosen model over the best alternative (>=0)."""
    util = ps - pen * pc / lt
    order = np.argsort(-util, axis=1)
    best = order[:, 0]
    u_sorted = np.take_along_axis(util, order, axis=1)
    return best, u_sorted[:, 0] - u_sorted[:, 1], order[:, 1]


print("=" * 100)
print("STEP 1  Lagrangian boundary geometry (dev, E43 held-out safety .98/.87/.85)")
print("=" * 100)

summary = {}
for t in TIERS:
    ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
    sel, pen, tot, cap = allocate_pen(ps, pc, TIER_MULT[t], SAFE_HOLDOUT[t])
    lt = pc[:, 0].sum()
    best, marg, runner = margins(ps, pc, pen, lt)
    true_ratio = dv.cost[IDX, sel].sum() / dv.cost[:, 0].sum()
    sc = dv.score[IDX, sel].mean()
    cnt = np.bincount(sel, minlength=3)
    print(f"\n--- {t}: lambda*={pen:.6f}  pred_ratio={tot/lt:.4f} cap={cap/lt:.4f} "
          f"true_ratio={true_ratio:.4f} score={sc:.4f}")
    print(f"    selection counts light/mid/k1 = {cnt}")
    qs = np.percentile(marg, [1, 5, 10, 25, 50, 75, 90])
    print("    margin percentiles (1/5/10/25/50/75/90): " + " ".join(f"{q:.4f}" for q in qs))
    summary[t] = dict(sel=sel, pen=pen, marg=marg, runner=runner, lt=lt)

print()
print("=" * 100)
print("STEP 1b  how much of the score is decided near the boundary?")
print("  band = |margin| < eps.  'flip' = force every band item to its runner-up model.")
print("  'oracle-band' = replace predicted score by TRUE score for band items only, re-allocate.")
print("=" * 100)
EPS = [0.005, 0.01, 0.02, 0.05, 0.10]
print(f"{'tier':10s} {'eps':>6s} {'n_band':>7s} {'%':>6s} {'flip_dscore':>12s} {'flip_dratio':>12s} "
      f"{'oracleband_dsc':>15s} {'oracleband_ratio':>17s}")
for t in TIERS:
    ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
    S = summary[t]
    sel, marg, runner = S["sel"], S["marg"], S["runner"]
    base_sc = dv.score[IDX, sel].mean()
    base_ratio = dv.cost[IDX, sel].sum() / dv.cost[:, 0].sum()
    for eps in EPS:
        band = marg < eps
        sel2 = sel.copy()
        sel2[band] = runner[band]
        d_sc = dv.score[IDX, sel2].mean() - base_sc
        d_ra = dv.cost[IDX, sel2].sum() / dv.cost[:, 0].sum() - base_ratio
        # oracle inside the band only
        ps2 = ps.copy()
        ps2[band] = dv.score[band]
        r2 = tier_result(ps2, pc, dv, t, SAFE_HOLDOUT[t])
        print(f"{t:10s} {eps:6.3f} {band.sum():7d} {100*band.mean():5.1f}% {d_sc:+12.4f} {d_ra:+12.4f} "
              f"{r2['score']-base_sc:+15.4f} {r2['ratio']:17.4f}")

print()
print("=" * 100)
print("STEP 1c  which items actually have a live decision at all?")
print("  'contested' = the argmax changes somewhere on the lambda path in [0, 4*lambda*]")
print("=" * 100)
for t in TIERS:
    ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
    S = summary[t]
    lt = S["lt"]
    pens = np.linspace(0, max(S["pen"] * 4, 1e-6), 200)
    picks = np.stack([(ps - p * pc / lt).argmax(1) for p in pens], 1)
    contested = (picks.min(1) != picks.max(1))
    # marginal decision: how many items would change if lambda moved +-10%
    for f in (0.9, 1.1):
        p2 = S["pen"] * f
        sel2 = (ps - p2 * pc / lt).argmax(1)
        ch = (sel2 != S["sel"]).sum()
        print(f"  {t:10s} lambda*x{f:.1f}: {ch:4d} items change selection "
              f"(true score {dv.score[IDX, sel2].mean():.4f} vs {dv.score[IDX, S['sel']].mean():.4f}, "
              f"true ratio {dv.cost[IDX, sel2].sum()/dv.cost[:,0].sum():.4f})")
    print(f"  {t:10s} contested over lambda in [0,4L]: {contested.sum()} items "
          f"({100*contested.mean():.1f}%);  always-light {np.mean(picks.max(1)==0)*100:.1f}%")

print()
print("=" * 100)
print("STEP 1d  value of a perfect score head restricted to the band vs everywhere")
print("=" * 100)
for t in TIERS:
    ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
    S = summary[t]
    base = tier_result(ps, pc, dv, t, SAFE_HOLDOUT[t])
    full = tier_result(dv.score, pc, dv, t, SAFE_HOLDOUT[t])
    rows = []
    for eps in (0.02, 0.05, 0.10, 0.25):
        band = S["marg"] < eps
        ps2 = ps.copy(); ps2[band] = dv.score[band]
        r = tier_result(ps2, pc, dv, t, SAFE_HOLDOUT[t])
        rows.append((eps, band.sum(), r["score"] - base["score"]))
    print(f"  {t:10s} base={base['score']:.4f} full-true-score={full['score']:.4f} "
          f"(+{full['score']-base['score']:.4f})")
    for eps, k, d in rows:
        print(f"      band eps<{eps:.2f} (n={k:4d}): +{d:.4f}  "
              f"= {100*d/max(full['score']-base['score'],1e-9):.0f}% of the full-oracle gain")
