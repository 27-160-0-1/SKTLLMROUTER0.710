# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b04 step 2 - prove the vectorised kNN replica is bit-equal to the shipped one."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import importlib
lib = importlib.import_module("b04-memory-heavy_lib")
from harness import Lab

lab = lib.MemLab()
t0 = time.perf_counter()
kf_ref, kh_ref, ff_ref, fh_ref = Lab._knn_family(lab, lab.train_idx, lab.dev_idx)
t_ref = time.perf_counter() - t0
t0 = time.perf_counter()
kf, kh, ff, fh = lab._knn_family(lab.train_idx, lab.dev_idx)
t_new = time.perf_counter() - t0
print(f"reference {t_ref:.1f}s   vectorised {t_new:.2f}s   speedup {t_ref/max(t_new,1e-9):.0f}x")
for name, a, b in (("knn_fit", kf_ref, kf[:, :7]), ("knn_hold", kh_ref, kh[:, :7]),
                   ("fam_fit", ff_ref, ff), ("fam_hold", fh_ref, fh)):
    d = np.abs(a - b)
    print(f"{name:9s} max|diff|={d.max():.3e}  rows with >1e-9 diff: {(d.max(axis=1)>1e-9).sum()}/{len(a)}")
