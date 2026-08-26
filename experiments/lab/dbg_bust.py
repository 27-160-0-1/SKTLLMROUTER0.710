# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, DEPLOYED_CFG
import gainlab as G
lab = Lab(verbose=False)
cv, arr = G.stage1(lab)
cands = {
 "deployed": dict(DEPLOYED_CFG),
 "cvbest":   {"legacy_w":0.95,"fam_w":0.2,"conf_scale":0.1,"gain_alpha":1.0,"rank_beta":0.8,
              "blend_fast":0.3,"blend_balanced":0.45,"blend_premium":0.15},
}
for name, cfg in cands.items():
    r = G.evaluate(lab, cv, arr, cfg, seeds=(7,17,23), nboot=300, label=name, return_curves=True)
    print(f"  bust@chosen: " + " ".join(f"{t}={r['detail'][t]['bust']:.4f}" for t in TIERS))
    for t in TIERS:
        g = np.array(r['detail'][t]['grid']); c = np.array(r['detail'][t]['curve'])
        print(f"    {t:9s} chosen={r['safety'][t]:.2f}  EV curve top5: " +
              " ".join(f"{g[i]:.2f}:{c[i]:.4f}" for i in np.argsort(-c)[:5]))
