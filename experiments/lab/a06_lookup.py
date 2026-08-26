# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a06 step 11: how far does a PURE partition-mean lookup router get?

Train-fitted per-bucket mean score and per-bucket mean log cost, applied to dev.
No learning beyond the bucket means.  Compared with the deployed E43 held-out
predictions and with the same predictions blended toward the bucket means.
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

SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
tr, dv = build("train"), build("dev")
sptr, spdv = tr["split"], dv["split"]
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)

PARTS = {
    "9 regex families": (tr["fam"], dv["fam"]),
    "11 sub-families": (subfam(tr["fam"], tr["X"], tr["names"]),
                        subfam(dv["fam"], dv["X"], dv["names"])),
}


def lookup(ptr, pdv):
    ms = {g: sptr.score[ptr == g].mean(0) for g in set(ptr)}
    mc = {g: np.exp(np.log(sptr.cost[ptr == g]).mean(0)) for g in set(ptr)}
    S = np.array([ms[g] for g in pdv])
    C = np.array([mc[g] for g in pdv])
    return S, C


def score(tag, S, C, safety=SAFE, sweep=False):
    tot, parts = 0.0, []
    for t in TIERS:
        Si = S(t) if callable(S) else S
        Ci = C(t) if callable(C) else C
        if sweep:
            bb = None
            for sf in np.arange(0.60, 1.201, 0.005):
                r = tier_result(Si, Ci, spdv, t, float(sf))
                if r["passed"] and (bb is None or r["score"] > bb[0]):
                    bb = (r["score"], float(sf), r["ratio"])
            tot += TIER_WEIGHT[t] * bb[0]
            parts.append(f"{t[:4]}={bb[0]:.4f}@{bb[1]:.3f}")
        else:
            r = tier_result(Si, Ci, spdv, t, safety[t])
            tot += TIER_WEIGHT[t] * r["tier_score"]
            parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else 'X'}")
    print(f"  {tag:52s} {tot:.4f}  " + " ".join(parts))
    return tot


print("=== pure lookup routers (train bucket means -> dev), deployed safety ===")
for nm, (ptr, pdv) in PARTS.items():
    S, C = lookup(ptr, pdv)
    score(f"lookup {nm}: pred score + pred cost", S, C)
    score(f"lookup {nm}: pred score + TRUE cost", S, spdv.cost)
print()
score("deployed E43", lambda t: P[f"score_{t}"], lambda t: P[f"cost_{t}"])

print("\n=== same, with the dev-tuned best safety per tier (upper bound) ===")
for nm, (ptr, pdv) in PARTS.items():
    S, C = lookup(ptr, pdv)
    score(f"lookup {nm}", S, C, sweep=True)
score("deployed E43", lambda t: P[f"score_{t}"], lambda t: P[f"cost_{t}"], sweep=True)

print("\n=== correlation with the realised dev score ===")
for nm, (ptr, pdv) in PARTS.items():
    S, _ = lookup(ptr, pdv)
    print(f"  lookup {nm:20s}",
          [round(float(np.corrcoef(S[:, j], spdv.score[:, j])[0, 1]), 3) for j in range(3)])
print("  deployed E43 (premium blend)  ",
      [round(float(np.corrcoef(P["score_premium"][:, j], spdv.score[:, j])[0, 1]), 3)
       for j in range(3)])
print("  deployed + subfam correction  ", end="")
ptr, pdv = PARTS["11 sub-families"]
S9, _ = lookup(*PARTS["9 regex families"])
S11, C11 = lookup(ptr, pdv)
Sc = P["score_premium"] + (S11 - S9)
print([round(float(np.corrcoef(Sc[:, j], spdv.score[:, j])[0, 1]), 3) for j in range(3)])

print("\n=== blend deployed toward the 11-bucket lookup ===")
_, C9 = lookup(*PARTS["9 regex families"])
for b in (0.0, 0.15, 0.3, 0.5, 0.7, 1.0):
    score(f"blend b={b} (score and cost)",
          lambda t, b=b: (1 - b) * P[f"score_{t}"] + b * S11,
          lambda t, b=b: np.exp((1 - b) * np.log(P[f"cost_{t}"]) + b * np.log(C11)))
