# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a05 step 0: verify which safety vector reproduces the E43 held-out numbers."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result, MODEL_IDS

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
print("npz keys:", list(P.keys()))
print("n dev:", len(dv), "pred shape:", P["score_fast"].shape)

for name, SAFE in (("E43 .98/.87/.85", {"fast": .98, "balanced": .87, "premium": .85}),
                   ("old .98/.89/.88", {"fast": .98, "balanced": .89, "premium": .88})):
    tot = 0.0
    parts = []
    for t in TIERS:
        r = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, SAFE[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t}={r['score']:.4f}/r{r['ratio']:.4f}{'' if r['passed'] else ' BUST'}")
    print(f"{name:18s} final={tot:.4f}  " + "  ".join(parts))

print("\nE43 log target: final 0.7019, fast 0.6764 r1.199 / bal 0.6972 r1.784 / prem 0.7406 r3.572")

# oracle
tot = 0.0
for t in TIERS:
    r = tier_result(dv.score, dv.cost, dv, t, 1.0)
    tot += TIER_WEIGHT[t] * r["tier_score"]
    print(f"oracle {t:9s} score={r['score']:.4f} ratio={r['ratio']:.4f} "
          f"sel counts={np.bincount(r['sel'], minlength=3)}")
print(f"oracle final={tot:.4f}")
