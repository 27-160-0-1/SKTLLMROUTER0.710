# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a08 probe 11: where do the allocator's DECISIONS and its LOSSES live, by family?

Tells a preprocessing designer which families are worth normalising.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import (load_split, MODEL_IDS, TIERS, TIER_WEIGHT, TIER_MULT,
                     tier_result)  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
fam = np.array([classify_family(t) for t in dv.texts])
FS = sorted(set(fam))

for tier in TIERS:
    sel = tier_result(P[f"score_{tier}"], P[f"cost_{tier}"], dv, tier, SAFE[tier])["sel"]
    orc = tier_result(dv.score, dv.cost, dv, tier, 1.0)["sel"]
    print(f"\n=== {tier} (weight {TIER_WEIGHT[tier]}) ===")
    print(f"  {'family':16s} {'n':>4s} {'sel L/M/K':>13s} {'orc L/M/K':>13s} "
          f"{'score':>7s} {'oracle':>7s} {'loss':>7s} {'costshare':>9s}")
    tot_loss = 0.0
    for f in FS:
        m = fam == f
        i = np.where(m)[0]
        s_sel = dv.score[i, sel[i]].sum()
        s_orc = dv.score[i, orc[i]].sum()
        c_sel = dv.cost[i, sel[i]].sum()
        cnt = np.bincount(sel[i], minlength=3)
        cno = np.bincount(orc[i], minlength=3)
        loss = (s_orc - s_sel) / len(dv)
        tot_loss += loss
        print(f"  {f:16s} {m.sum():4d} {'/'.join(map(str,cnt)):>13s} "
              f"{'/'.join(map(str,cno)):>13s} {s_sel/len(dv):7.4f} {s_orc/len(dv):7.4f} "
              f"{loss:7.4f} {c_sel/dv.cost[np.arange(len(dv)),sel].sum():9.3f}")
    print(f"  {'TOTAL':16s} {len(dv):4d} {'':13s} {'':13s} "
          f"{dv.score[np.arange(len(dv)),sel].mean():7.4f} "
          f"{dv.score[np.arange(len(dv)),orc].mean():7.4f} {tot_loss:7.4f}")

print("\n=== where the light-vs-mid (fast tier) decision is actually made ===")
sel = tier_result(P["score_fast"], P["cost_fast"], dv, "fast", SAFE["fast"])["sel"]
up = sel > 0
print(f"  upgraded items: {up.sum()} / {len(dv)}")
for f in FS:
    m = fam == f
    print(f"  {f:16s} n={m.sum():4d} upgraded={int((m&up).sum()):4d} "
          f"({100*(m&up).sum()/max(m.sum(),1):5.1f}%)  "
          f"realised gain of those upgrades="
          f"{(dv.score[np.where(m&up)[0], sel[m&up]] - dv.score[m&up, 0]).sum()/len(dv):+.4f}")

print("\n=== E42 check: predicted vs true cost of the items the allocator UPGRADES ===")
for tier in TIERS:
    sel = tier_result(P[f"score_{tier}"], P[f"cost_{tier}"], dv, tier, SAFE[tier])["sel"]
    up = np.where(sel > 0)[0]
    pc = P[f"cost_{tier}"][up, sel[up]].sum()
    tc = dv.cost[up, sel[up]].sum()
    pall = P[f"cost_{tier}"][:, 0].sum()
    tall = dv.cost[:, 0].sum()
    print(f"  {tier:9s} upgraded n={len(up):3d}  pred/true on upgraded = {pc/tc:.3f}   "
          f"pred/true on light baseline = {pall/tall:.3f}   "
          f"selection bias = {(pc/tc)/(pall/tall):.3f}")
