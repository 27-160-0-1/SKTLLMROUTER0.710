# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b07 - build the bench2 stage once under my own tag (10-fold OOF over Train
+ a train-only fit predicting Dev).  Everything else in b07 reuses it."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_EXP
import bench2 as B

if __name__ == "__main__":
    t0 = time.perf_counter()
    lab = Lab()
    cv, arr = B.stage(lab, DEPLOYED_EXP, tag="b07-visualiser")
    print(f"[b07] stage ready cv_n={len(cv['idx'])} dev_n={len(arr['idx'])} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
