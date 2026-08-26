# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a05 step 10: what is the ITEM-LEVEL part of each gain head worth today?
Replace one gain by its within-family mean (killing all item-level content of that
decision) and re-allocate."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_WEIGHT, tier_result
ROOT = Path(__file__).resolve().parents[2]

dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
Cc = np.load(ROOT / "experiments/lab/a05-selection-failure_cache.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
N = len(dv); IDX = np.arange(N); FAM = Cc["fam"]; PHAT = Cc["phat"]


def run(mk_s, tune=True):
    tot_r = tot_e = 0.0; sfs = []
    for t in TIERS:
        if tune:
            best = None; bsf = None
            for sf in np.arange(0.5, 1.301, 0.005):
                r = tier_result(mk_s(t), P[f"cost_{t}"], dv, t, float(sf))
                if r["passed"] and (best is None or r["score"] > best["score"]):
                    best = r; bsf = float(sf)
            r = best; sfs.append(bsf)
        else:
            r = tier_result(mk_s(t), P[f"cost_{t}"], dv, t, SAFE[t]); sfs.append(SAFE[t])
        tot_r += TIER_WEIGHT[t] * r["tier_score"]
        tot_e += TIER_WEIGHT[t] * (PHAT[IDX, r["sel"]].mean() if r["passed"] else 0.0)
    return tot_r, tot_e, sfs


def famflat(v):
    out = v.copy()
    for f_ in set(FAM.tolist()):
        m = FAM == f_
        out[m] = v[m].mean()
    return out


print("=== value of the ITEM-LEVEL content of each gain head (dev-tuned safety, pred costs) ===")
cases = {
    "as deployed": (False, False),
    "kill item-level g1 (mid-light), keep g2": (True, False),
    "kill item-level g2 (k1-mid), keep g1": (False, True),
    "kill both (family-flat gains)": (True, True),
}
for nm, (k1, k2) in cases.items():
    def mk(t, k1=k1, k2=k2):
        S = P[f"score_{t}"].copy()
        a = S[:, 1] - S[:, 0]; b = S[:, 2] - S[:, 1]
        if k1: a = famflat(a)
        if k2: b = famflat(b)
        S[:, 1] = S[:, 0] + a; S[:, 2] = S[:, 1] + b
        return S
    r, e, sfs = run(mk)
    r2, e2, _ = run(mk, tune=False)
    print(f"  {nm:42s} tuned realised={r:.4f} EB={e:.4f} | deployed-safety realised={r2:.4f} EB={e2:.4f}")

print("\n=== and the same replacing one gain with the EB truth (upper bound per decision) ===")
for nm, use1, use2 in (("EB-true g1 only", True, False), ("EB-true g2 only", False, True),
                       ("EB-true both", True, True)):
    def mk(t, use1=use1, use2=use2):
        S = P[f"score_{t}"].copy()
        a = (PHAT[:, 1] - PHAT[:, 0]) if use1 else (S[:, 1] - S[:, 0])
        b = (PHAT[:, 2] - PHAT[:, 1]) if use2 else (S[:, 2] - S[:, 1])
        S[:, 1] = S[:, 0] + a; S[:, 2] = S[:, 1] + b
        return S
    r, e, sfs = run(mk)
    print(f"  {nm:42s} tuned realised={r:.4f} EB={e:.4f} safety={'/'.join(f'{x:.3f}' for x in sfs)}")
