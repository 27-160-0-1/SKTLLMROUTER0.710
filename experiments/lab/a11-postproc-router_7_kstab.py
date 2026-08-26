# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a11 step 7 - is the per-model cost calibration factor k_j estimable and stable?

k_j = sum(true cost_j) / sum(pred cost_j).  Rule B needs k_j/k_0 to transfer from
the fitting sample to the evaluation sample.  Measured here:
  * bootstrap sd of k_j and k_j/k_0
  * how much of k_2 comes from the few most expensive items (jackknife)
  * per-family k_j (does composition shift move it?)
  * a deliberately shifted split: fit k on the "cheap" half, apply to the
    "expensive" half
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, MODEL_IDS
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router import similarity

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv)
fam = np.array([similarity.classify_family(x) for x in dv.texts])
t = "premium"
C = P[f"cost_{t}"]


def kof(mask):
    return dv.cost[mask].sum(0) / C[mask].sum(0)


k_all = kof(np.ones(n, bool))
print(f"=== k_j on all of dev ({t} cost head) ===")
print(f"  k      = {np.round(k_all,4)}")
print(f"  k/k0   = {np.round(k_all/k_all[0],4)}")

print("\n=== bootstrap distribution of k_j and k_j/k_0 (880 resamples, B=2000) ===")
g = np.random.default_rng(11)
S = g.integers(0, n, size=(2000, n))
kb = dv.cost[S].sum(1) / C[S].sum(1)
rel = kb / kb[:, [0]]
for j, m in enumerate(MODEL_IDS):
    print(f"  {m:12s} k mean={kb[:,j].mean():.3f} sd={kb[:,j].std():.3f} "
          f"[p05 {np.quantile(kb[:,j],.05):.3f}, p95 {np.quantile(kb[:,j],.95):.3f}]  "
          f"k/k0 mean={rel[:,j].mean():.3f} sd={rel[:,j].std():.3f} "
          f"[p05 {np.quantile(rel[:,j],.05):.3f}, p95 {np.quantile(rel[:,j],.95):.3f}]")

print("\n=== jackknife: k_2 after removing the N most expensive true k1 items ===")
order = np.argsort(-dv.cost[:, 2])
for N in (0, 1, 2, 5, 10, 20, 50):
    m = np.ones(n, bool); m[order[:N]] = False
    k = kof(m)
    print(f"  drop top {N:3d}: k = {np.round(k,4)}  k/k0 = {np.round(k/k[0],4)}")

print("\n=== per-family k_j (premium head) ===")
print(f"  {'family':16s} {'n':>4s} {'k_light':>8s} {'k_mid':>7s} {'k_k1':>7s} {'k1/k0':>7s}")
for f in sorted(set(fam)):
    m = fam == f
    k = kof(m)
    print(f"  {f:16s} {m.sum():4d} {k[0]:8.3f} {k[1]:7.3f} {k[2]:7.3f} {k[2]/k[0]:7.3f}")

print("\n=== deliberately shifted split (fit on cheap half, apply to expensive half) ===")
med = np.median(dv.cost[:, 2])
cheap = dv.cost[:, 2] < med
for name, m in (("cheap half", cheap), ("expensive half", ~cheap)):
    k = kof(m)
    print(f"  {name:15s} k = {np.round(k,4)}  k/k0 = {np.round(k/k[0],4)}")
lo = kof(cheap); hi = kof(~cheap)
print(f"  transfer error on k_2/k_0: {hi[2]/hi[0] / (lo[2]/lo[0]) - 1:+.1%}")

print("\n=== random 50/50 split, 200 repeats: |k2/k0(A) - k2/k0(B)| ===")
d = []
for r in range(200):
    m = np.random.default_rng(100 + r).permutation(n) < n // 2
    a = kof(m); b = kof(~m)
    d.append(abs(a[2] / a[0] - b[2] / b[0]))
print(f"  mean |diff| = {np.mean(d):.4f}  p90 = {np.quantile(d,0.9):.4f}  "
      f"(k2/k0 level = {k_all[2]/k_all[0]:.4f})")

print("\n=== how many items dominate the k1 cost sum? ===")
c2 = np.sort(dv.cost[:, 2])[::-1]
for N in (1, 5, 10, 25, 50):
    print(f"  top {N:3d} items = {100*c2[:N].sum()/c2.sum():5.1f}% of the total true k1 cost")
