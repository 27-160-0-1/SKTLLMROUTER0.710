# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b04 step 1 - time the pieces so the memory-heavy arms can be budgeted."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from harness import Lab, DEPLOYED_EXP

lab = Lab()
t0 = time.perf_counter()
kf, kh, ff, fh = lab._knn_family(lab.train_idx, lab.dev_idx)
t_knn = time.perf_counter() - t0
print(f"knn_family(train->dev)  {t_knn:.1f}s  shapes {kf.shape} {kh.shape}")

t0 = time.perf_counter()
arr = lab.fit_predict(lab.train_idx, lab.dev_idx, DEPLOYED_EXP)
print(f"fit_predict(train->dev) {time.perf_counter()-t0:.1f}s")
