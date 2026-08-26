# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b04 - memory-heavy arms measured under the bench2 protocol.

usage:  python b04-memory-heavy_04_arms.py ARM [ARM ...]
Each arm builds its own 10-fold OOF stage (train-only) + a train->dev fit, then
bench2 chooses the per-tier safety from the OOF bootstrap EV and scores dev once.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")
from harness import DEPLOYED_EXP, DEPLOYED_CFG, TIERS
import bench2 as B

BASE_EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)

W16 = dict(kind="word", k=16, top_comp=256)
PF16 = dict(kind="char", k=16, top_comp=256, per_family=True)
CST = dict(kind="char", k=32, top_comp=0, stats=True)

ARMS = {
    # ---- reference points -------------------------------------------------
    "R0": dict(spec={}, exp=DEPLOYED_EXP),
    "R1": dict(spec={}, exp=BASE_EXP),
    # ---- k scaling (the index is already in memory; k is free) -------------
    "K32":  dict(spec=dict(k=32)),
    "K64":  dict(spec=dict(k=64)),
    "K128": dict(spec=dict(k=128)),
    # ---- exact (untruncated) index vectors: 4.1 MB -> ~30 MB --------------
    "X16":  dict(spec=dict(top_comp=0)),
    "X64":  dict(spec=dict(top_comp=0, k=64)),
    # ---- the other direction: HARVEST memory and runtime ------------------
    "T128": dict(spec=dict(top_comp=128)),
    "T64":  dict(spec=dict(top_comp=64)),
    "T512": dict(spec=dict(top_comp=512)),
    # ---- extra similarity views as meta features --------------------------
    "VW":   dict(spec=dict(views=(W16,))),
    "VF":   dict(spec=dict(views=(PF16,))),
    "VS":   dict(spec=dict(views=(CST,))),
    "VALL": dict(spec=dict(views=(W16, PF16, CST))),
    # ---- meta-head seed ensembles (artifact x N) --------------------------
    "S3":   dict(spec=dict(seeds=(11, 23, 37))),
    "S5":   dict(spec=dict(seeds=(11, 23, 37, 51, 67))),
    "S8":   dict(spec=dict(seeds=(11, 23, 37, 51, 67, 83, 97, 109))),
    # ---- composition ------------------------------------------------------
    "BEST": dict(spec=dict(k=64, top_comp=0, views=(W16,), seeds=(11, 23, 37, 51, 67))),
}

OUT = Path("reports/lab/b04_arms.json")


def main(names):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for nm in names:
        a = ARMS[nm]
        t0 = time.perf_counter()
        lab = lib.MemLab(a["spec"], verbose=False)
        cv, arr = B.stage(lab, a.get("exp", BASE_EXP), tag=f"b04_{nm}")
        r = B.run(lab, cv, arr, DEPLOYED_CFG, label=f"{nm}")
        r["knn_cols"] = int(arr["knn"].shape[1])
        r["secs"] = round(time.perf_counter() - t0, 1)
        res[nm] = {k: v for k, v in r.items() if k != "curves"}
        OUT.write_text(json.dumps(res, indent=1, default=float))
        print(f"    ({r['secs']}s, knn cols {r['knn_cols']})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["R0", "R1"])
