# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b03 step 1: reproduce the baseline under the honest protocol, orient on premium."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP, DEPLOYED_SAFETY
import bench2 as B

lab = Lab()
cv0, arr0 = B.stage(lab, DEPLOYED_EXP, tag="base")
cv1, arr1 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
print("cv rows", len(cv0["idx"]), "dev rows", len(arr0["idx"]))

r0 = B.run(lab, cv0, arr0, DEPLOYED_CFG, label="base (EV-opt safety)")
r1 = B.run(lab, cv1, arr1, DEPLOYED_CFG, label="legoof (EV-opt safety)")
B.run(lab, cv0, arr0, DEPLOYED_CFG, label="base @deployed", fixed_safety=DEPLOYED_SAFETY)
B.run(lab, cv1, arr1, DEPLOYED_CFG, label="legoof @deployed", fixed_safety=DEPLOYED_SAFETY)

for nm, r in (("base", r0), ("legoof", r1)):
    print(nm, "safety", r["safety"], "bust", {t: round(r["det"][t]["bust"], 4) for t in TIERS},
          "dev tiers", {t: (round(r["dev_tiers"][t]["score"], 4), round(r["dev_tiers"][t]["ratio"], 3))
                        for t in TIERS})

# premium picks on dev under legoof
ps, pc = lab.compose(arr1, DEPLOYED_CFG, "premium")
pick = lab.allocate(ps, pc, 4.0, r1["safety"]["premium"])
print("premium dev pick counts", np.bincount(pick, minlength=3).tolist())
fam = lab.fam_arr[arr1["idx"]]
k1 = pick == 2
print("k1 by family:", {f: int((fam[k1] == f).sum()) for f in sorted(set(fam[k1]))})
