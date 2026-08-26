# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a03: cross the score head and the cost head of the plain ridge with those of
the deployed E43 stack, to find out which half is responsible for the deployed
stack scoring higher despite a LOWER score correlation.

Also reports a bootstrap-EV (880 resamples of dev) figure so the comparison
does not rest on a single dev-tuned safety point.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lab"))

import labdata  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "a03curve", Path(__file__).with_name("a03-train-failure_curve.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

GRID = np.round(np.arange(0.60, 1.0001, 0.01), 3)


def boot_ev(ps, pc, dv, reps=120, seed=7):
    """E[tier score x budget pass] over 880-size bootstrap resamples of dev,
    safety chosen per tier to maximise that EV (project standard, dev-only)."""
    n = len(dv)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, n, n) for _ in range(reps)]
    out, best_s = {}, {}
    for t in labdata.TIERS:
        mult = labdata.TIER_MULT[t]
        bev, bs = -1.0, None
        for s in GRID:
            acc = 0.0
            for d in draws:
                p_s, p_c = ps[d], pc[d]
                cost = dv.cost[d]
                sel = labdata.allocate(p_s, p_c, cost, mult, float(s))
                idx = np.arange(n)
                ratio = cost[idx, sel].sum() / cost[:, 0].sum()
                acc += dv.score[d][idx, sel].mean() if ratio <= mult + 1e-15 else 0.0
            acc /= len(draws)
            if acc > bev:
                bev, bs = acc, float(s)
        out[t], best_s[t] = bev, bs
    ev = sum(labdata.TIER_WEIGHT[t] * out[t] for t in labdata.TIERS)
    return ev, out, best_s


def main():
    dv = labdata.load_split("dev")
    Xtr, Xdv, m = _c.load()
    G, Gd = _c.grams(Xtr, Xdv)
    Y, Ldv = m["Ytr"], m["Ldv"]
    ridge = _c.ridge_dual(G, Gd, Y, np.arange(len(Y)), 10.0)[0]
    rs, rc = _c.to_alloc(ridge)
    d = np.load(ROOT / "reports/lab/dev_preds_e43.npz")
    ds, dc = d["score_fast"], d["cost_fast"]
    ls, lc = _c.to_alloc(Ldv)
    combos = {
        "ridge s + ridge c": (rs, rc),
        "ridge s + E43   c": (rs, dc),
        "E43   s + ridge c": (ds, rc),
        "E43   s + E43   c": (ds, dc),
        "legacy s + legacy c": (ls, lc),
        "legacy s + E43   c": (ls, dc),
        "E43   s + legacy c": (ds, lc),
    }
    print("== crossing score and cost heads (dev)")
    print(f"{'combo':22s} {'devtuned':>9s} {'bootEV':>8s}  safety(f/b/p)   "
          f"corr s0/s1/s2       lcRMSE c0/c1/c2")
    Ydv = m["Ydv"]
    for name, (ps, pc) in combos.items():
        f, _, _ = _c.best_final(np.column_stack([ps, np.log(pc)]), dv)
        ev, per, bs = boot_ev(ps, pc, dv)
        cs = [np.corrcoef(ps[:, j], Ydv[:, j])[0, 1] for j in range(3)]
        rm = [float(np.sqrt(np.mean((np.log(pc[:, j]) - Ydv[:, 3 + j]) ** 2))) for j in range(3)]
        print(f"{name:22s} {f:9.4f} {ev:8.4f}  {bs['fast']:.2f}/{bs['balanced']:.2f}/"
              f"{bs['premium']:.2f}   {cs[0]:.3f}/{cs[1]:.3f}/{cs[2]:.3f}   "
              f"{rm[0]:.3f}/{rm[1]:.3f}/{rm[2]:.3f}")
    print("\n== realised cost ratio of the chosen set at a FIXED safety, fast tier")
    print("   (E42's selection-induced cost bias: predicted vs realised ratio)")
    for name, (ps, pc) in combos.items():
        sel = labdata.allocate(ps, pc, dv.cost, 1.25, 0.90)
        idx = np.arange(len(dv))
        pr = pc[idx, sel].sum() / pc[:, 0].sum()
        ac = dv.cost[idx, sel].sum() / dv.cost[:, 0].sum()
        up = int((sel > 0).sum())
        print(f"   {name:22s} predicted {pr:.3f}  realised {ac:.3f}  "
              f"ratio {ac/pr:.3f}  upgrades {up}")


if __name__ == "__main__":
    main()
