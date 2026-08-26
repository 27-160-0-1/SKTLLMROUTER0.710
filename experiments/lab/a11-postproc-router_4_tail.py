# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a11 step 4 - the catastrophic single item: who is it, is it visible ex ante?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, tier_result
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router import similarity

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv); idx = np.arange(n)
fam = np.array([similarity.classify_family(x) for x in dv.texts])
L = dv.cost[:, 0].sum()

print("=== items whose TRUE k1 cost is a large share of the premium budget ===")
share = dv.cost[:, 2] / L
order = np.argsort(-share)
Cp = P["cost_premium"]
pshare = Cp[:, 2] / Cp[:, 0].sum()
print(f"{'rank':>4s} {'item':>4s} {'fam':16s} {'true_k1/L':>10s} {'pred_k1/Lhat':>13s} "
      f"{'pct_of_pred':>11s} {'true_otok':>10s} {'ngen':>4s} {'s_light':>7s} {'s_k1':>5s}")
for r, i in enumerate(order[:15]):
    pr = 100.0 * (pshare < pshare[i]).mean()
    print(f"{r:4d} {i:4d} {fam[i]:16s} {share[i]:10.4f} {pshare[i]:13.4f} {pr:10.1f}% "
          f"{dv.otok[i,2]:10.0f} {dv.ngen[i,2]:4.0f} {dv.score[i,0]:7.2f} {dv.score[i,2]:5.2f}")

print("\n=== the item that flips at premium s=0.9025 ===")
a = tier_result(P["score_premium"], Cp, dv, "premium", 0.9000)["sel"]
b = tier_result(P["score_premium"], Cp, dv, "premium", 0.9025)["sel"]
w = np.where(a != b)[0]
for i in w:
    print(f"  item {i} fam={fam[i]} sel {a[i]}->{b[i]}  true cost {dv.cost[i,a[i]]:.4f}->{dv.cost[i,b[i]]:.4f} "
          f"(+{(dv.cost[i,b[i]]-dv.cost[i,a[i]])/L:.4f} of L)  pred {Cp[i,a[i]]:.4f}->{Cp[i,b[i]]:.4f} "
          f"(+{(Cp[i,b[i]]-Cp[i,a[i]])/Cp[:,0].sum():.4f})  true score {dv.score[i,a[i]]:.2f}->{dv.score[i,b[i]]:.2f} "
          f"pred score {P['score_premium'][i,a[i]]:.3f}->{P['score_premium'][i,b[i]]:.3f}")
    print(f"      true k1 out_tok={dv.otok[i,2]:.0f} ngen={dv.ngen[i,2]:.0f}  "
          f"pred/true cost ratio = {Cp[i,2]/dv.cost[i,2]:.3f}")

print("\n=== how much ratio mass sits in items under-predicted by >3x? ===")
for t in TIERS:
    C = P[f"cost_{t}"]
    r = tier_result(P[f"score_{t}"], C, dv, t, {"fast":0.98,"balanced":0.87,"premium":0.85}[t])
    sel = r["sel"]
    und = dv.cost[idx, sel] / C[idx, sel]
    for thr in (2.0, 3.0, 5.0):
        m = und > thr
        print(f"  {t:9s} >{thr:.0f}x under-predicted: {m.sum():3d} items, "
              f"{dv.cost[m, sel[m]].sum()/L:.3f} of L "
              f"({100*dv.cost[m, sel[m]].sum()/dv.cost[idx,sel].sum():.1f}% of realised cost); "
              f"excess over pred = {(dv.cost[m,sel[m]]-C[m,sel[m]]).sum()/L:.3f} of L")

print("\n=== is a PREDICTED-share cap able to see the dangerous items? ===")
t = "premium"; C = P[f"cost_{t}"]
sel = tier_result(P[f"score_{t}"], C, dv, t, 0.85)["sel"]
k1 = np.where(sel == 2)[0]
tc = dv.cost[k1, 2] / L
pc = C[k1, 2] / C[:, 0].sum()
print(f"  among the {len(k1)} k1 selections: corr(log pred_share, log true_share) = "
      f"{np.corrcoef(np.log(pc), np.log(tc))[0,1]:.3f}")
for q in (0.9, 0.95, 0.99):
    thr = np.quantile(pc, q)
    hi = pc >= thr
    print(f"    pred top-{100*(1-q):.0f}%: {hi.sum():2d} items hold {tc[hi].sum()/tc.sum():.3f} of "
          f"the true k1 cost; the true top-{100*(1-q):.0f}% hold "
          f"{np.sort(tc)[-hi.sum():].sum()/tc.sum():.3f}")
print(f"  recall of the true top-10% true-cost k1 items by the pred top-10% filter: ", end="")
thr_p = np.quantile(pc, 0.9); thr_t = np.quantile(tc, 0.9)
print(f"{((pc>=thr_p) & (tc>=thr_t)).sum()}/{(tc>=thr_t).sum()}")
