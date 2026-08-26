# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a11 step 13 - split-half validation of the kappa2 (k1 relative price) gain.

The gain is measured at a MATCHED realised budget ratio, inside each half of a
random 50/50 split of dev, repeated over 20 splits.  kappa2 is NOT fitted inside
the half - it is a fixed constant - so this measures how reliably the effect
reproduces on a fresh 440-item sample.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_MULT
import importlib.util
_spec = importlib.util.spec_from_file_location("a11path", HERE / "a11-postproc-router_5_path.py")
a11path = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(a11path)

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv)


def curve_sub(tier, k2, idx):
    pc = (P[f"cost_{tier}"] * np.array([1.0, 1.0, k2])[None, :])[idx]
    pa = a11path.path_arrays(P[f"score_{tier}"][idx], pc, dv.score[idx], dv.cost[idx])
    v = pa["valid"].ravel(); o = np.argsort(-pa["eff"].ravel()[v], kind="stable")
    ct = np.concatenate([[0.0], np.cumsum(pa["dc_t"].ravel()[v][o])])
    st = np.concatenate([[0.0], np.cumsum(pa["ds_t"].ravel()[v][o])])
    L = dv.cost[idx, 0].sum()
    return (L + ct) / L, (dv.score[idx, 0].sum() + st) / len(idx)


K2 = [1.10, 1.24, 1.50, 2.00, 3.00]
print("=== split-half validation, 20 random 50/50 splits (440 items each) ===")
for tier in TIERS:
    mult = TIER_MULT[tier]
    tg = np.arange(0.80, 1.001, 0.02) * mult
    acc = {k: [] for k in K2}
    for rep in range(20):
        perm = np.random.default_rng(500 + rep).permutation(n)
        for half in (perm[: n // 2], perm[n // 2:]):
            r0, s0 = curve_sub(tier, 1.0, half)
            base = np.array([s0[r0 <= t + 1e-12].max() if (r0 <= t + 1e-12).any() else np.nan
                             for t in tg])
            for k2 in K2:
                r, s = curve_sub(tier, k2, half)
                v = np.array([s[r <= t + 1e-12].max() if (r <= t + 1e-12).any() else np.nan
                              for t in tg])
                acc[k2].append(np.nanmean(v - base))
    print(f"-- {tier}")
    for k2 in K2:
        a = np.array(acc[k2])
        print(f"   kappa2={k2:4.2f}  mean {a.mean():+.5f}  sd {a.std():.5f}  "
              f"halves positive {int((a>0).sum())}/{len(a)}  "
              f"t={a.mean()/(a.std()/np.sqrt(len(a))):+.1f}")
