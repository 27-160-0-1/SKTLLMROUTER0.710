# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""A01: quantify the optimism from choosing the safety ratios on dev."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labdata as L

tr, dv = L.load_all()
z = np.load(Path(__file__).resolve().parents[2] / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
print("npz keys:", sorted(z.files))
S = {t: z[f"score_{t}"] for t in L.TIERS}
C = {t: z[f"cost_{t}"] for t in L.TIERS}

def final(safety):
    tot, det = 0.0, {}
    for t in L.TIERS:
        r = L.tier_result(S[t], C[t], dv, t, safety[t])
        det[t] = r
        tot += L.TIER_WEIGHT[t] * r["tier_score"]
    return tot, det

for name, sa in [("E43 deployed .98/.87/.85", dict(fast=.98, balanced=.87, premium=.85)),
                 ("build_meta_gbm defaults .98/.89/.88", dict(fast=.98, balanced=.89, premium=.88)),
                 ("all 1.0", dict(fast=1., balanced=1., premium=1.))]:
    f, d = final(sa)
    print(f"{name:42s} final={f:.6f}  " +
          " ".join(f"{t}:{d[t]['tier_score']:.4f}(r={d[t]['ratio']:.3f},pass={d[t]['passed']})" for t in L.TIERS))

# per-tier grid: what does dev-optimal safety look like vs the chosen constants?
print()
print("tier  best_safety  best_tier_score  score@deployed  delta(dev-optimal - deployed)")
grid = np.round(np.arange(0.50, 1.0001, 0.0025), 6)
dep = dict(fast=.98, balanced=.87, premium=.85)
rows = {}
for t in L.TIERS:
    vals = []
    for sa in grid:
        r = L.tier_result(S[t], C[t], dv, t, float(sa))
        vals.append((r["tier_score"], float(sa), r["ratio"], r["passed"]))
    rows[t] = vals
    best = max(vals)
    rdep = L.tier_result(S[t], C[t], dv, t, dep[t])
    print(f"{t:9s} {best[1]:.4f}      {best[0]:.6f}       {rdep['tier_score']:.6f}     {best[0]-rdep['tier_score']:+.6f}")

# weighted final over the joint grid (what build_router_augmentation._calibrate_safety would find)
bestf = 0.0; bestsa = None
for t in L.TIERS:
    pass
best_per_tier = {t: max(rows[t])[1] for t in L.TIERS}
f, d = final(best_per_tier)
print(f"\ndev-optimal joint safety {best_per_tier} -> final={f:.6f}")
fd, _ = final(dep)
print(f"deployed .98/.87/.85                          -> final={fd:.6f}   optimism={f-fd:+.6f}")

# bust cliff: how wide is the plateau
print("\npremium fine grid around the cliff (safety, ratio, passed, tier_score):")
for sa in np.arange(0.83, 0.925, 0.005):
    r = L.tier_result(S["premium"], C["premium"], dv, "premium", float(sa))
    print(f"  {sa:.3f}  ratio={r['ratio']:.4f} passed={r['passed']} tier={r['tier_score']:.6f} raw={r['score']:.6f}")
print("\nbalanced fine grid:")
for sa in np.arange(0.84, 0.935, 0.005):
    r = L.tier_result(S["balanced"], C["balanced"], dv, "balanced", float(sa))
    print(f"  {sa:.3f}  ratio={r['ratio']:.4f} passed={r['passed']} tier={r['tier_score']:.6f} raw={r['score']:.6f}")
