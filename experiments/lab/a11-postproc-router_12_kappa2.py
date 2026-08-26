# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a11 step 12 - (i) where does the ratio variance live (numerator vs denominator),
(ii) the kappa correction: fine grid, theory value, matched-budget gain."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
import importlib.util
_spec = importlib.util.spec_from_file_location("a11path", HERE / "a11-postproc-router_5_path.py")
a11path = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(a11path)

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv)
DEP = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}

print("=== (i) variance of the realised ratio: numerator vs denominator ===")
g = np.random.default_rng(3)
S = g.integers(0, n, size=(4000, n))
D = dv.cost[:, 0][S].sum(1)
print(f"{'tier':9s} {'sd(logN)':>9s} {'sd(logD)':>9s} {'corr':>6s} {'sd(logR)':>9s} "
      f"{'sd if D known':>14s} {'D share of var':>15s}")
for t in TIERS:
    sel = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, DEP[t])["sel"]
    N = dv.cost[np.arange(n), sel][S].sum(1)
    lN, lD = np.log(N), np.log(D)
    r = np.corrcoef(lN, lD)[0, 1]
    sdR = np.std(lN - lD)
    print(f"{t:9s} {lN.std():9.4f} {lD.std():9.4f} {r:6.3f} {sdR:9.4f} "
          f"{lN.std():14.4f} {1 - (lN.std()/sdR)**2:14.1%}")
print("  ('sd if D known' = the spread that would remain if the true light total")
print("   were known exactly; the last column is how much of the ratio variance")
print("   the denominator removes/adds through its correlation with N.)")

print("\n=== (ii) sigma_j and the theoretical median->mean correction exp(sigma^2/2) ===")
for t in TIERS:
    E = np.log(dv.cost) - np.log(P[f"cost_{t}"])
    s = E.std(0)
    th = np.exp(s ** 2 / 2)
    emp = dv.cost.sum(0) / P[f"cost_{t}"].sum(0)
    print(f"  {t:9s} sigma={np.round(s,3)} exp(s^2/2)={np.round(th,3)} "
          f"rel={np.round(th/th[0],3)} | empirical k/k0={np.round(emp/emp[0],3)}")


def curve(tier, kappa):
    pc = P[f"cost_{tier}"] * np.asarray(kappa)[None, :]
    pa = a11path.path_arrays(P[f"score_{tier}"], pc, dv.score, dv.cost)
    v = pa["valid"].ravel(); o = np.argsort(-pa["eff"].ravel()[v], kind="stable")
    ct = np.concatenate([[0.0], np.cumsum(pa["dc_t"].ravel()[v][o])])
    st = np.concatenate([[0.0], np.cumsum(pa["ds_t"].ravel()[v][o])])
    return (dv.cost[:, 0].sum() + ct) / dv.cost[:, 0].sum(), (dv.score[:, 0].sum() + st) / n


print("\n=== (iii) matched-realised-budget gain vs kappa2 (dev, deterministic) ===")
KGRID = [1.0, 1.05, 1.10, 1.15, 1.20, 1.24, 1.30, 1.40, 1.50, 1.60, 1.80, 2.00, 2.50]
for tier in TIERS:
    mult = TIER_MULT[tier]
    targets = np.arange(0.80, 1.001, 0.02) * mult      # 11 budget points
    r0, s0 = curve(tier, (1, 1, 1))
    base = np.array([s0[r0 <= tg + 1e-12].max() for tg in targets])
    print(f"-- {tier}: mean gain over {len(targets)} budget points in "
          f"[{0.8*mult:.2f}, {mult:.2f}]")
    for k2 in KGRID:
        r, s = curve(tier, (1, 1, k2))
        v = np.array([s[r <= tg + 1e-12].max() for tg in targets])
        d = v - base
        print(f"   kappa2={k2:4.2f}  mean {d.mean():+.5f}  median {np.median(d):+.5f}  "
              f"min {d.min():+.5f}  max {d.max():+.5f}  n_pos={int((d>0).sum())}/{len(d)}")

print("\n=== (iv) same, for the theory-derived kappa and a mid-only variant ===")
for tier in TIERS:
    mult = TIER_MULT[tier]
    targets = np.arange(0.80, 1.001, 0.02) * mult
    r0, s0 = curve(tier, (1, 1, 1))
    base = np.array([s0[r0 <= tg + 1e-12].max() for tg in targets])
    for kap in ((1, 0.955, 1.10), (1, 0.93, 1.24), (1, 0.93, 1.0), (1, 1.0, 1.24)):
        r, s = curve(tier, kap)
        v = np.array([s[r <= tg + 1e-12].max() for tg in targets])
        d = v - base
        print(f"   {tier:9s} kappa={str(kap):18s} mean {d.mean():+.5f} "
              f"n_pos={int((d>0).sum())}/{len(d)}")
