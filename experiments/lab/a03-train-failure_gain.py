# SPDX-License-Identifier: Apache-2.0
"""a03: the allocator consumes GAINS and EFFICIENCIES, not score levels.

Measures, for every candidate predictor, the level correlation, the upgrade-gain
correlation and the efficiency-rank correlation, so they can be compared with
the final score each predictor achieves.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lab"))

import labdata  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "a03curve", Path(__file__).with_name("a03-train-failure_curve.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)


def main():
    dv = labdata.load_split("dev")
    Xtr, Xdv, m = _c.load()
    G, Gd = _c.grams(Xtr, Xdv)
    Y, Ydv, Ldv = m["Ytr"], m["Ydv"], m["Ldv"]
    d = np.load(ROOT / "reports/lab/dev_preds_e43.npz")
    preds = {}
    for a in (3.0, 10.0, 100.0, 1000.0):
        p = _c.ridge_dual(G, Gd, Y, np.arange(len(Y)), a)[0]
        preds[f"ridge a={a:g}"] = _c.to_alloc(p)
    preds["legacy"] = _c.to_alloc(Ldv)
    for w in (0.75, 0.9):
        preds[f"blend w={w:g}"] = _c.to_alloc(
            (1 - w) * _c.ridge_dual(G, Gd, Y, np.arange(len(Y)), 10.0)[0] + w * Ldv)
    preds["E43 deployed"] = (d["score_fast"], d["cost_fast"])
    ts, tc = dv.score, dv.cost
    g1t, g2t = ts[:, 1] - ts[:, 0], ts[:, 2] - ts[:, 1]
    e1t = g1t / (tc[:, 1] - tc[:, 0])
    e2t = g2t / (tc[:, 2] - tc[:, 1])
    print("== level corr / gain corr / efficiency-rank corr, dev")
    print(f"{'predictor':16s} {'lvl s0':>7s}{'s1':>7s}{'s2':>7s} | {'gain 1-0':>9s}"
          f"{'gain 2-1':>9s} | {'eff1 rho':>9s}{'eff2 rho':>9s} | {'final':>7s}")
    for name, (ps, pc) in preds.items():
        lv = [np.corrcoef(ps[:, j], ts[:, j])[0, 1] for j in range(3)]
        g1 = ps[:, 1] - ps[:, 0]
        g2 = ps[:, 2] - ps[:, 1]
        cg1 = np.corrcoef(g1, g1t)[0, 1]
        cg2 = np.corrcoef(g2, g2t)[0, 1]
        e1 = g1 / np.maximum(pc[:, 1] - pc[:, 0], 1e-12)
        e2 = g2 / np.maximum(pc[:, 2] - pc[:, 1], 1e-12)
        r1 = spearmanr(e1, e1t).statistic
        r2 = spearmanr(e2, e2t).statistic
        f, _, _ = _c.best_final(np.column_stack([ps, np.log(pc)]), dv)
        print(f"{name:16s} {lv[0]:7.3f}{lv[1]:7.3f}{lv[2]:7.3f} | {cg1:9.3f}{cg2:9.3f}"
              f" | {r1:9.3f}{r2:9.3f} | {f:7.4f}")
    print("\n== dispersion of the predicted gains (the allocator's raw material)")
    print(f"{'predictor':16s} {'sd g1':>8s}{'sd g2':>8s}  {'frac g1<0':>10s}{'frac g2<0':>10s}")
    for name, (ps, pc) in preds.items():
        g1 = ps[:, 1] - ps[:, 0]
        g2 = ps[:, 2] - ps[:, 1]
        print(f"{name:16s} {g1.std():8.4f}{g2.std():8.4f}  {(g1<0).mean():10.3f}{(g2<0).mean():10.3f}")
    print(f"{'TRUE':16s} {g1t.std():8.4f}{g2t.std():8.4f}  {(g1t<0).mean():10.3f}{(g2t<0).mean():10.3f}")


if __name__ == "__main__":
    main()
