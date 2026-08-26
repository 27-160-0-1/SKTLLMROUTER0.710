# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a06 step 13: paired 880-item bootstrap of the sub-family correction delta.

Same resampling style as the project's EV harness: resample the 880 dev
episodes with replacement, re-run the whole allocation (budget is recomputed
inside the resample), and take the paired difference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from a06_features import build  # noqa: E402
from a06_counterfactual import subfam  # noqa: E402
from labdata import TIERS, TIER_WEIGHT, tier_result  # noqa: E402


class Sub:
    def __init__(self, score, cost):
        self.score, self.cost = score, cost

    def __len__(self):
        return len(self.score)


tr, dv = build("train"), build("dev")
sptr, spdv = tr["split"], dv["split"]
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
gtr = subfam(tr["fam"], tr["X"], tr["names"])
gdv = subfam(dv["fam"], dv["X"], dv["names"])
ds, dc = {}, {}
for g in sorted(set(gtr)):
    mt = gtr == g
    base = {"gsm": "gsm8k_or_other"}.get(g.split(".")[0], g.split(".")[0])
    mf = tr["fam"] == base
    ds[g] = sptr.score[mt].mean(0) - sptr.score[mf].mean(0)
    dc[g] = np.log(sptr.cost[mt]).mean(0) - np.log(sptr.cost[mf]).mean(0)
DS = np.array([ds[g] for g in gdv])
DC = np.array([dc[g] for g in gdv])

for SAFE in ({"fast": 0.98, "balanced": 0.89, "premium": 0.88},
             {"fast": 0.98, "balanced": 0.87, "premium": 0.85}):
    rng = np.random.default_rng(7)
    d_all, a_all, b_all, bust_a, bust_b = [], [], [], 0, 0
    for _ in range(400):
        idx = rng.integers(0, len(spdv), len(spdv))
        sub = Sub(spdv.score[idx], spdv.cost[idx])
        tot = []
        busts = []
        for w in (0.0, 1.0):
            s = 0.0
            nb = 0
            for t in TIERS:
                S = P[f"score_{t}"][idx] + w * DS[idx]
                C = P[f"cost_{t}"][idx] * np.exp(w * DC[idx])
                r = tier_result(S, C, sub, t, SAFE[t])
                s += TIER_WEIGHT[t] * r["tier_score"]
                nb += 0 if r["passed"] else 1
            tot.append(s)
            busts.append(nb)
        a_all.append(tot[0]); b_all.append(tot[1]); d_all.append(tot[1] - tot[0])
        bust_a += busts[0] > 0
        bust_b += busts[1] > 0
    d = np.array(d_all)
    print(f"safety={list(SAFE.values())}  baseline EV={np.mean(a_all):.4f}  "
          f"corrected EV={np.mean(b_all):.4f}  delta={d.mean():+.4f} "
          f"[{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]  "
          f"P(delta>0)={np.mean(d>0):.3f}  bust runs: base {bust_a}/400 corr {bust_b}/400")
