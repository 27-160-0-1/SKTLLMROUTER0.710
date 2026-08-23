# SPDX-License-Identifier: Apache-2.0
"""a06 step 12: realised group-level upgrade efficiency (score units per
light-budget), for the 11 sub-families, on train and dev, plus what the
deployed allocator actually buys at the premium tier."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from a06_features import build  # noqa: E402
from a06_counterfactual import subfam  # noqa: E402
from labdata import tier_result  # noqa: E402

rows = {}
for nm in ("train", "dev"):
    d = build(nm)
    sp = d["split"]
    g = subfam(d["fam"], d["X"], d["names"])
    lb = sp.cost[:, 0].sum()
    for k in sorted(set(g)):
        m = g == k
        n = int(m.sum())
        d_mid = (sp.score[m, 1] - sp.score[m, 0]).sum()
        c_mid = (sp.cost[m, 1] - sp.cost[m, 0]).sum() / lb
        d_k1 = (sp.score[m, 2] - sp.score[m, 0]).sum()
        c_k1 = (sp.cost[m, 2] - sp.cost[m, 0]).sum() / lb
        rows.setdefault(k, {})[nm] = (n, d_mid / max(c_mid, 1e-9), c_mid,
                                      d_k1 / max(c_k1, 1e-9), c_k1)

print("Group-level realised upgrade efficiency = (sum of score gain) / (extra cost in")
print("light-budget units).  Higher = buy first.  880/1760-item score units.")
print(f"{'subfamily':16s} | " + " | ".join(
    f"{s:>5s} n  eff_mid  dc_mid  eff_k1  dc_k1" for s in ("train", "dev")))
order = sorted(rows, key=lambda k: -rows[k]["dev"][3])
for k in order:
    line = f"{k:16s} |"
    for nm in ("train", "dev"):
        n, em, cm, ek, ck = rows[k][nm]
        line += f" {n:5d} {em:8.1f} {cm:7.3f} {ek:7.1f} {ck:6.3f} |"
    print(line)

print("\nWhat the deployed E43 allocator buys at premium (dev, safety .85)")
dv = build("dev")
spdv = dv["split"]
gdv = subfam(dv["fam"], dv["X"], dv["names"])
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
r = tier_result(P["score_premium"], P["cost_premium"], spdv, "premium", 0.85)
sel = r["sel"]
lb = spdv.cost[:, 0].sum()
print(f"{'subfamily':16s} {'n':>4s} {'%k1':>6s} {'%mid':>6s} {'lb spent':>9s} "
      f"{'score':>7s} {'all-light':>10s} {'all-k1':>7s}")
for k in order:
    m = gdv == k
    spent = spdv.cost[np.arange(len(spdv)), sel][m].sum() / lb
    sc = spdv.score[m][np.arange(m.sum()), sel[m]].mean()
    print(f"{k:16s} {m.sum():4d} {100*np.mean(sel[m]==2):6.1f} {100*np.mean(sel[m]==1):6.1f} "
          f"{spent:9.3f} {sc:7.3f} {spdv.score[m,0].mean():10.3f} {spdv.score[m,2].mean():7.3f}")
print(f"{'TOTAL':16s} {len(spdv):4d} {100*np.mean(sel==2):6.1f} {100*np.mean(sel==1):6.1f} "
      f"{spdv.cost[np.arange(len(spdv)), sel].sum()/lb:9.3f} {r['score']:7.3f}")
