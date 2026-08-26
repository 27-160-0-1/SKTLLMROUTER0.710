# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b06 forest-view: the concrete pruning proposal, measured.

Removing gain_alpha and rank_beta deletes 4 of the 22 GBM heads, the 65-node
quantile LUT and the RANK_FLOOR_Q constant.  Removing the ordinal ladder deletes
12 more heads.  Measured under bench2 against the full stack.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B

OUT = Path("reports/lab/b06_prune.json")


def main():
    lab = Lab()
    cv, arr = B.stage(lab, DEPLOYED_EXP, tag="base")
    res = {}

    def go(cv_, arr_, cfg, label, note=""):
        r = B.run(lab, cv_, arr_, cfg, label=label)
        res[label] = dict(EV=r["EV"], dev=r["dev"],
                          safety={t: r["safety"][t] for t in TIERS}, note=note)
        return r

    go(cv, arr, None, "A full deployed stack", "22 GBM heads, 8 constants")
    go(cv, arr, dict(rank_beta=0.0), "B  -rank_beta", "kills 2 rank heads + 65-node LUT")
    go(cv, arr, dict(gain_alpha=0.0), "C  -gain_alpha", "kills the gain reconstruction")
    go(cv, arr, dict(gain_alpha=0.0, rank_beta=0.0), "D  -both (4 heads gone)",
       "18 heads, 6 constants")
    go(cv, arr, dict(gain_alpha=0.0, rank_beta=0.0, blend_balanced=0.6, blend_premium=0.6),
       "E  D + one shared blend", "18 heads, 4 constants")

    t0 = time.perf_counter()
    exp2 = dict(DEPLOYED_EXP, ordinal=False, rank=False)
    cv2, arr2 = B.stage(lab, exp2, tag="b06-noord")
    print(f"[b06] no-ordinal stage built in {time.perf_counter()-t0:.0f}s", flush=True)
    go(cv2, arr2, dict(gain_alpha=0.0, rank_beta=0.0), "F  -ordinal -rank -gain",
       "6 GBM heads only (of 22), 6 constants")
    go(cv2, arr2, None, "G  -ordinal, keep gain", "8 GBM heads")

    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
