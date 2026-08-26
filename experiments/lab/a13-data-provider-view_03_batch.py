# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a13 / provider view - part 3: private-set SIZE is a free parameter of the
safety factor, and the AIME left-over count pins it.

Bust risk is driven by the sampling variance of the realised cost ratio, which
falls like 1/sqrt(n).  The deployed safety ratios (.98/.87/.85) were tuned with
880-item bootstrap batches because dev is 880.  If the private evaluation set is
larger, the same ratios are strictly too conservative.

This script measures EV(safety, batch_size) on the deployed E43 dev predictions.
Run: .venv/Scripts/python.exe experiments/lab/a13_provider_03_batch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
sys.path.insert(0, str(ROOT / "src"))

import labdata  # noqa: E402
from labdata import TIERS, TIER_MULT, TIER_WEIGHT  # noqa: E402

dv = labdata.load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
DEPLOYED = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
NBOOT = 200


def alloc_sample(ps, pc, tc, ts, mult, safety):
    """One bootstrap batch: returns (true_ratio, mean_true_score)."""
    light_total = pc[:, 0].sum()
    cap = light_total * max(1.0, mult * safety)

    def choose(pen):
        return (ps - pen * pc / light_total).argmax(axis=1)

    sel = choose(0.0)
    tot = pc[np.arange(len(sel)), sel].sum()
    if tot > cap:
        low, high = 0.0, 1.0
        sel = choose(high)
        tot = pc[np.arange(len(sel)), sel].sum()
        while tot > cap and high < 2 ** 60:
            low, high = high, high * 2.0
            sel = choose(high)
            tot = pc[np.arange(len(sel)), sel].sum()
        for _ in range(40):
            mid = (low + high) / 2.0
            cand = choose(mid)
            ct = pc[np.arange(len(cand)), cand].sum()
            if ct <= cap:
                high, sel, tot = mid, cand, ct
            else:
                low = mid
    if tot > cap:
        sel = np.zeros(len(ps), dtype=int)
    idx = np.arange(len(sel))
    ratio = tc[idx, sel].sum() / tc[:, 0].sum()
    return ratio, ts[idx, sel].mean()


def ev_curve(tier, safeties, n, seeds=(7, 17), nboot=NBOOT):
    """returns dict safety -> (EV, bust_prob, mean score | pass)"""
    ps_all, pc_all = P[f"score_{tier}"], P[f"cost_{tier}"]
    mult = TIER_MULT[tier]
    out = {}
    for sf in safeties:
        evs, busts, sc = [], [], []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            for _ in range(nboot):
                ix = rng.integers(0, len(dv), n)
                r, s = alloc_sample(ps_all[ix], pc_all[ix], dv.cost[ix], dv.score[ix],
                                    mult, sf)
                ok = r <= mult + 1e-15
                evs.append(s if ok else 0.0)
                busts.append(0.0 if ok else 1.0)
                if ok:
                    sc.append(s)
        out[sf] = (float(np.mean(evs)), float(np.mean(busts)),
                   float(np.mean(sc)) if sc else float("nan"))
    return out


print("=" * 92)
print("EV(safety, batch size) per tier - deployed E43 predictions on dev")
print("=" * 92)
GRID = {
    "fast": [0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.04],
    "balanced": [0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 1.00],
    "premium": [0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1.00, 1.03],
}
SIZES = [440, 880, 1760, 2640]
best = {}
ALL = {}
for tier in TIERS:
    print(f"\n---- {tier}  (budget x{TIER_MULT[tier]}, weight {TIER_WEIGHT[tier]})")
    hdr = f"{'safety':>7s}"
    for n in SIZES:
        hdr += f" | n={n:<5d} EV   bust"
    print(hdr)
    tab = {n: ev_curve(tier, sorted(set(GRID[tier] + [DEPLOYED[tier]])), n) for n in SIZES}
    ALL[tier] = tab
    for sf in GRID[tier]:
        line = f"{sf:7.3f}"
        for n in SIZES:
            ev, bp, _ = tab[n][sf]
            line += f" | {ev:.4f} {bp*100:5.1f}%"
        mark = "  <= deployed" if abs(sf - DEPLOYED[tier]) < 1e-9 else ""
        print(line + mark)
    for n in SIZES:
        b = max(GRID[tier], key=lambda s: tab[n][s][0])
        best.setdefault(n, {})[tier] = (b, tab[n][b][0])
        print(f"   best @ n={n}: safety {b:.3f} EV {tab[n][b][0]:.4f} "
              f"(deployed {DEPLOYED[tier]:.2f} EV {tab[n][DEPLOYED[tier]][0]:.4f}"
              f" -> delta {tab[n][b][0]-tab[n][DEPLOYED[tier]][0]:+.4f})")

print()
print("=" * 92)
print("weighted EV: deployed safety vs size-optimal safety, per assumed batch size")
print("=" * 92)
for n in SIZES:
    dep = sum(TIER_WEIGHT[t] * ALL[t][n][DEPLOYED[t]][0] for t in TIERS)
    opt = sum(TIER_WEIGHT[t] * ALL[t][n][best[n][t][0]][0] for t in TIERS)
    sel = " / ".join(f"{best[n][t][0]:.2f}" for t in TIERS)
    print(f"  n={n:5d}:  deployed-safety EV {dep:.4f}   size-optimal EV {opt:.4f}   "
          f"gap {opt-dep:+.4f}   optimal = {sel}")

print()
print("cross-application: apply the n-optimal safety triple at every OTHER n")
print(f"{'tuned@n':>8s} " + " ".join(f"{'eval@'+str(n):>11s}" for n in SIZES))
for nt in SIZES:
    trip = {t: best[nt][t][0] for t in TIERS}
    line = f"{nt:8d} "
    for ne in SIZES:
        v = sum(TIER_WEIGHT[t] * ALL[t][ne][trip[t]][0] for t in TIERS)
        line += f" {v:11.4f}"
    print(line + f"   ({'/'.join(f'{trip[t]:.2f}' for t in TIERS)})")
