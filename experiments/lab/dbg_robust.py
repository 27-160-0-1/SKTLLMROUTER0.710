# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_CFG
import gainlab as G
lab = Lab(verbose=False)
cv, arr = G.stage1(lab)
cands = {
 "deployed": dict(DEPLOYED_CFG),
 "cvbest":   {"legacy_w":0.95,"fam_w":0.2,"conf_scale":0.1,"gain_alpha":1.0,"rank_beta":0.8,
              "blend_fast":0.3,"blend_balanced":0.45,"blend_premium":0.15},
}
for tau in (0.05, 0.02, 0.01):
    print(f"--- tau={tau}")
    for name, cfg in cands.items():
        G.robust_evaluate(lab, cv, arr, cfg, label=f"{name} tau={tau}", tau=tau, nboot=200)
