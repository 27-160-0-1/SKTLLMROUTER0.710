# SPDX-License-Identifier: Apache-2.0
"""b05 step 1: build the instrumented stage and verify it reproduces C1 exactly."""
import importlib.util, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b05lib", HERE / "b05-time-heavy_lib.py")
lib = importlib.util.module_from_spec(spec); sys.modules["b05lib"] = lib
spec.loader.exec_module(lib)

from harness import DEPLOYED_EXP, DEPLOYED_CFG
import bench2 as B

EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)
lab = lib.XLab()
cv, arr, parts = lib.build_stage(lab, EXP, tag="b05base")
r = B.run(lab, cv, arr, DEPLOYED_CFG, label="b05 stage (== C1 legacy-OOF)")
print("safety", r["safety"])

ps, _ = lab.compose(cv, DEPLOYED_CFG, "fast")
lib.gain_axis(lab, cv["idx"], ps[:, 1] - ps[:, 0], ps[:, 2] - ps[:, 1], "composed OOF (fast blend)")
lib.gain_axis(lab, cv["idx"], cv["gain"][:, 0], cv["gain"][:, 1], "raw gain head OOF")
ps, _ = lab.compose(arr, DEPLOYED_CFG, "fast")
lib.gain_axis(lab, arr["idx"], ps[:, 1] - ps[:, 0], ps[:, 2] - ps[:, 1], "composed DEV (fast blend)")
lib.gain_axis(lab, arr["idx"], arr["gain"][:, 0], arr["gain"][:, 1], "raw gain head DEV")

# reference: family-mean-only gain predictor, honest (fit on the fold's fit set)
fam = lab.fam_arr
d = lab.delta_targets
g1 = np.zeros(len(cv["idx"])); g2 = np.zeros(len(cv["idx"]))
pos = {v: k for k, v in enumerate(cv["idx"])}
for p in parts:
    fi, hi = p["fit_idx"], p["idx"]
    means = {f: d[fi[fam[fi] == f]].mean(axis=0) for f in np.unique(fam[fi])}
    gm = d[fi].mean(axis=0)
    for j, i in enumerate(hi):
        m = means.get(fam[i], gm)
        g1[pos[i]] = m[0]; g2[pos[i]] = m[1]
lib.gain_axis(lab, cv["idx"], g1, g2, "family-mean-only gain OOF")
print(f"[b05] X shape {arr['Xf'].shape} / {arr['Xh'].shape}")
