# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""E49 - compose the confirmed round-1 candidates under the honest protocol."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_EXP, DEPLOYED_CFG, DEPLOYED_SAFETY
from famrepair import classify_v3
import bench2 as B

lab = Lab()
res = []
def go(tag, exp, famv3, label, force=False):
    if famv3:
        lab.set_family([classify_v3(t) for t in lab.texts])
    else:
        lab._finish_features()
    cv, arr = B.stage(lab, exp, tag=tag, force=force)
    r = B.run(lab, cv, arr, DEPLOYED_CFG, label=label)
    r2 = B.run(lab, cv, arr, DEPLOYED_CFG, label=label + " @E43safety",
               fixed_safety=DEPLOYED_SAFETY)
    res.append((label, r, r2))
    return r

from collections import Counter
old = list(lab.fam_names)
new = [classify_v3(t) for t in lab.texts]
c = Counter((a, b) for a, b in zip(old, new) if a != b)
print("family reassignment:", dict(c), f" moved {sum(c.values())}/{lab.n}")

go("base",    DEPLOYED_EXP, False, "A baseline")
go("legoof",  dict(DEPLOYED_EXP, legacy_oof_meta=True), False, "B +legacy-OOF")
go("famv3",   DEPLOYED_EXP, True,  "C +family-v3")
go("bothv3",  dict(DEPLOYED_EXP, legacy_oof_meta=True), True, "D +both")
Path("reports/lab/e49_compose.json").write_text(json.dumps(
    [{"label": l, "ev_opt": {k: v for k, v in r.items() if k != "curves"},
      "at_e43": {k: v for k, v in r2.items() if k != "curves"}} for l, r, r2 in res],
    indent=2, default=float), encoding="utf-8")
