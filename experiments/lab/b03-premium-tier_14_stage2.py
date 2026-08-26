# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b03: replicate CV stage with a different fold seed (dev arr is fold-independent)."""
import sys, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_EXP
import bench2 as B
lab = Lab()
t0 = time.perf_counter()
cv = B.cv_arrays(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), seed=777)
Path("reports/lab/b03_cv_seed777.pkl").write_bytes(pickle.dumps(cv))
print("built in %.0fs, rows=%d" % (time.perf_counter() - t0, len(cv["idx"])))
