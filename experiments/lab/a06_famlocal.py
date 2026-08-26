# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a06 step 15: does the within-family dense model actually buy final score?

Blend b of (per-family GBM on the 35 interpretable features, trained on train)
into the deployed E43 dev score predictions, optionally together with the
sub-family cost correction.  Point estimate + 400x bootstrap bust curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

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

# ---- per-family dense score model (train -> dev), and per-family dense log-cost model
DENSE_S = np.zeros((len(spdv), 3))
DENSE_C = np.zeros((len(spdv), 3))
for f in sorted(set(dv["fam"])):
    mt, md = tr["fam"] == f, dv["fam"] == f
    for j in range(3):
        y = sptr.score[mt, j]
        if np.std(y) < 1e-9:
            DENSE_S[md, j] = y.mean()
        else:
            g = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=8,
                                              min_samples_leaf=20, learning_rate=0.08,
                                              random_state=0).fit(tr["X"][mt], y)
            DENSE_S[md, j] = np.clip(g.predict(dv["X"][md]), 0.0, 1.0)
        gc = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15,
                                           min_samples_leaf=15, learning_rate=0.08,
                                           random_state=0).fit(tr["X"][mt],
                                                               np.log(sptr.cost[mt, j]))
        DENSE_C[md, j] = gc.predict(dv["X"][md])

# ---- sub-family correction (train-fitted), as in a06_counterfactual
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


def mk(t, b_s, b_c, sub):
    S = (1 - b_s) * P[f"score_{t}"] + b_s * DENSE_S + (DS if sub else 0.0)
    lc = (1 - b_c) * np.log(P[f"cost_{t}"]) + b_c * DENSE_C + (DC if sub else 0.0)
    return S, np.exp(lc)


print("point estimate on dev, per-tier best safety (0.60..1.20 grid)")
print(f"  {'config':40s} {'final':>7s}  per tier")
for b_s in (0.0, 0.25, 0.5, 0.75, 1.0):
    for b_c, sub in ((0.0, False), (0.0, True), (0.5, True)):
        tot, det = 0.0, []
        for t in TIERS:
            S, C = mk(t, b_s, b_c, sub)
            bb = None
            for sf in np.arange(0.60, 1.201, 0.005):
                r = tier_result(S, C, spdv, t, float(sf))
                if r["passed"] and (bb is None or r["score"] > bb[0]):
                    bb = (r["score"], float(sf))
            tot += TIER_WEIGHT[t] * bb[0]
            det.append(f"{t[:4]}={bb[0]:.4f}@{bb[1]:.3f}")
        print(f"  b_s={b_s:.2f} b_c={b_c:.2f} subfam={int(sub)}            "
              f"{tot:7.4f}  " + " ".join(det))

print("\n400x bootstrap, best safety per tier chosen ON THE BOOTSTRAP EV")
B = 400
rng = np.random.default_rng(7)
IDX = [rng.integers(0, len(spdv), len(spdv)) for _ in range(B)]
GRID = {"fast": np.arange(0.84, 1.001, 0.02),
        "balanced": np.arange(0.73, 0.911, 0.02),
        "premium": np.arange(0.69, 0.891, 0.02)}
for tag, b_s, b_c, sub in (("baseline", 0.0, 0.0, False),
                           ("subfam only", 0.0, 0.0, True),
                           ("dense .5 score only", 0.5, 0.0, False),
                           ("dense .5 + subfam", 0.5, 0.0, True),
                           ("dense .5 score+cost + subfam", 0.5, 0.5, True)):
    tot, det = 0.0, []
    for t in TIERS:
        S, C = mk(t, b_s, b_c, sub)
        best = None
        for sf in GRID[t]:
            sc, bu = [], 0
            for idx in IDX:
                r = tier_result(S[idx], C[idx], Sub(spdv.score[idx], spdv.cost[idx]),
                                t, float(sf))
                sc.append(r["tier_score"])
                bu += 0 if r["passed"] else 1
            ev = float(np.mean(sc))
            if best is None or ev > best[1]:
                best = (float(sf), ev, bu / B)
        tot += TIER_WEIGHT[t] * best[1]
        det.append(f"{t[:4]}={best[1]:.4f}@{best[0]:.2f}({100*best[2]:.0f}%)")
    print(f"  {tag:30s} EV={tot:.4f}  " + " ".join(det))
