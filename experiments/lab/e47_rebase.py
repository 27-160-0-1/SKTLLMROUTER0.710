# SPDX-License-Identifier: Apache-2.0
"""E47 - rebuild the CV baseline with the legacy head refitted out-of-fold."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, DEPLOYED_CFG, DEPLOYED_EXP, DEPLOYED_SAFETY
import gainlab as G
lab = Lab()
cv, arr = G.stage1(lab, force=True)          # legacy_refit=True is now the default
print("=== deployed cfg, CV-chosen safety (legacy now out-of-fold) ===")
r = G.evaluate(lab, cv, arr, DEPLOYED_CFG, seeds=(7,17,23), nboot=300, label="deployed")
print("=== same, at the E43 deployed safety vector ===")
idx = arr["idx"]; tot = 0
for t in TIERS:
    ps, pc = lab.compose(arr, DEPLOYED_CFG, t)
    pick = lab.allocate(ps, pc, {"fast":1.25,"balanced":2.0,"premium":4.0}[t], DEPLOYED_SAFETY[t])
    rr = np.arange(len(idx))
    ratio = lab.true_c[idx][rr,pick].sum()/lab.true_c[idx][:,0].sum()
    sc = lab.true_s[idx][rr,pick].mean()
    ok = ratio <= {"fast":1.25,"balanced":2.0,"premium":4.0}[t]
    tot += {"fast":0.4,"balanced":0.3,"premium":0.3}[t]*(sc if ok else 0)
    print(f"  {t:9s} score={sc:.6f} ratio={ratio:.4f} passed={ok}")
print(f"  FINAL(E43 safety) = {tot:.6f}")
