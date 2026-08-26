# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a11 step 5 - the Lagrangian path as an explicit object + verification.

Key structural claim to verify:  a post-hoc repair pass that downgrades items in
increasing order of efficiency is IDENTICAL to walking back along the Lagrangian
path, so any "repair" rule is exactly the deployed allocator with a different
ACCEPTANCE TEST.  This script builds the path and checks that walking it
reproduces labdata.allocate() bit-for-bit at many safety values.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, tier_result, allocate

ROOT = Path(__file__).resolve().parents[2]


def build_path(ps: np.ndarray, pc: np.ndarray):
    """Upper concave envelope per item -> (n,2) segment arrays.

    seg k of item i upgrades from model `frm[i,k]` to `to[i,k]`.
    Segments are ordered by decreasing efficiency within an item, so a global
    sort by efficiency always respects the prerequisite ordering.
    """
    n = len(ps)
    frm = np.zeros((n, 2), dtype=int)
    to = np.zeros((n, 2), dtype=int)
    valid = np.zeros((n, 2), dtype=bool)
    for i in range(n):
        cur = 0
        for k in range(2):
            best, best_eff = -1, 0.0
            for j in range(cur + 1, 3):
                dc = pc[i, j] - pc[i, cur]
                if dc <= 0:
                    continue
                eff = (ps[i, j] - ps[i, cur]) / dc
                if eff > best_eff + 1e-18:
                    best, best_eff = j, eff
            if best < 0:
                break
            frm[i, k], to[i, k], valid[i, k] = cur, best, True
            cur = best
            if cur == 2:
                break
    return frm, to, valid


def path_arrays(ps, pc, ts, tc):
    frm, to, valid = build_path(ps, pc)
    n = len(ps); r = np.arange(n)
    dc_p = np.where(valid, pc[r[:, None], to] - pc[r[:, None], frm], 0.0)
    ds_p = np.where(valid, ps[r[:, None], to] - ps[r[:, None], frm], 0.0)
    dc_t = np.where(valid, tc[r[:, None], to] - tc[r[:, None], frm], 0.0)
    ds_t = np.where(valid, ts[r[:, None], to] - ts[r[:, None], frm], 0.0)
    eff = np.where(valid, ds_p / np.maximum(dc_p, 1e-300), -np.inf)
    return dict(frm=frm, to=to, valid=valid, dc_p=dc_p, ds_p=ds_p,
                dc_t=dc_t, ds_t=ds_t, eff=eff)


def walk(pa, cap, sel_out=False):
    """Longest efficiency-ordered prefix with predicted cost <= cap."""
    v = pa["valid"].ravel()
    eff = pa["eff"].ravel()[v]
    dcp = pa["dc_p"].ravel()[v]
    order = np.argsort(-eff, kind="stable")
    cum = np.cumsum(dcp[order])
    k = int(np.searchsorted(cum, cap, side="right"))
    return order, k


if __name__ == "__main__":
    dv = load_split("dev")
    P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
    n = len(dv); r = np.arange(n)
    print("verifying path-walk == deployed bisection allocator")
    for t in TIERS:
        ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
        pa = path_arrays(ps, pc, dv.score, dv.cost)
        base_pred = pc[:, 0].sum()
        v = pa["valid"].ravel()
        item_of = np.repeat(np.arange(n)[:, None], 2, axis=1).ravel()[v]
        to_f = pa["to"].ravel()[v]
        bad = 0; maxdiff = 0.0
        for s in np.arange(0.70, 1.01, 0.01):
            cap = base_pred * max(1.0, TIER_MULT[t] * s)
            order, k = walk(pa, cap - base_pred)
            sel = np.zeros(n, dtype=int)
            taken = order[:k]
            sel[item_of[taken]] = to_f[taken]
            ref = allocate(ps, pc, dv.cost, TIER_MULT[t], float(s))
            d = int((sel != ref).sum())
            bad += d > 0
            maxdiff = max(maxdiff, d / n)
        print(f"  {t:9s}: safety grid 0.70..1.00, disagreeing grid points={bad}/31, "
              f"max item disagreement={maxdiff:.4%}")
