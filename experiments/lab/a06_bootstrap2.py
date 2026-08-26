# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a06 step 13b: per-tier bootstrap EV / bust curve for baseline vs the
sub-family correction, so the two can be compared at MATCHED bust risk
(the E39/E43 methodology).  400 resamples of the 880 dev episodes."""
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

B = 400
rng = np.random.default_rng(7)
IDX = [rng.integers(0, len(spdv), len(spdv)) for _ in range(B)]

GRID = {"fast": np.arange(0.86, 1.021, 0.02),
        "balanced": np.arange(0.75, 0.951, 0.02),
        "premium": np.arange(0.71, 0.931, 0.02)}

print("per-tier bootstrap: mean tier score (0 when busted) and bust rate")
best = {}
for t in TIERS:
    print(f"\n  --- {t} (cap x{ {'fast':1.25,'balanced':2.0,'premium':4.0}[t] }) ---")
    print(f"  {'safety':>7s} | {'base EV':>8s} {'bust%':>6s} | {'corr EV':>8s} {'bust%':>6s}")
    for sf in GRID[t]:
        out = []
        for w in (0.0, 1.0):
            S = P[f"score_{t}"] + w * DS
            C = P[f"cost_{t}"] * np.exp(w * DC)
            sc, bu = [], 0
            for idx in IDX:
                r = tier_result(S[idx], C[idx], Sub(spdv.score[idx], spdv.cost[idx]),
                                t, float(sf))
                sc.append(r["tier_score"])
                bu += 0 if r["passed"] else 1
            out.append((float(np.mean(sc)), bu / B))
        print(f"  {sf:7.3f} | {out[0][0]:8.4f} {100*out[0][1]:5.1f}% | "
              f"{out[1][0]:8.4f} {100*out[1][1]:5.1f}%")
        for w, o in zip((0.0, 1.0), out):
            k = (t, w)
            if k not in best or o[0] > best[k][1]:
                best[k] = (float(sf), o[0], o[1])

print("\nbest safety per tier by bootstrap EV")
for w in (0.0, 1.0):
    tot = sum(TIER_WEIGHT[t] * best[(t, w)][1] for t in TIERS)
    print(f"  w={w}: " + "  ".join(
        f"{t}={best[(t,w)][0]:.2f}->{best[(t,w)][1]:.4f}({100*best[(t,w)][2]:.0f}% bust)"
        for t in TIERS) + f"  weighted EV = {tot:.4f}")
