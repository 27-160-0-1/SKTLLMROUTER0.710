# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b04 step 0 - reproduce the two reference points under bench2 and time the stage."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from harness import Lab, DEPLOYED_EXP, DEPLOYED_CFG, DEPLOYED_SAFETY
import bench2 as B

lab = Lab()
t0 = time.perf_counter()
cv0, arr0 = B.stage(lab, DEPLOYED_EXP, tag="base")
cv1, arr1 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
print(f"[b04] stages loaded in {time.perf_counter()-t0:.1f}s", flush=True)

r0 = B.run(lab, cv0, arr0, DEPLOYED_CFG, label="R0 baseline (legacy in-sample)")
r1 = B.run(lab, cv1, arr1, DEPLOYED_CFG, label="R1 legacy-OOF meta (C1)")
print("safety R0", r0["safety"], "R1", r1["safety"])
print("knn block shape", arr0["knn"].shape)
