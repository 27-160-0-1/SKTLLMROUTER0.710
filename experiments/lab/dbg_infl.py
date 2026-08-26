# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

import sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, DEPLOYED_CFG
import bench2 as B
lab = Lab(verbose=False)
print(f"{'stage':10s} {'tier':9s} {'sf':>5s} | {'set':6s} {'pred_ratio':>10s} {'true_ratio':>10s} "
      f"{'infl':>6s} | {'predL':>8s} {'trueL':>8s} {'Lbias':>6s} {'predSel/trueSel':>16s}")
for tag in ("base", "legoof"):
    cv, arr = B.stage(lab, None if tag=="base" else {"legacy_oof_meta": True}, tag=tag)
    for t in TIERS:
        for name, src in (("trainOOF", cv), ("dev", arr)):
            ps, pc = lab.compose(src, DEPLOYED_CFG, t)
            idx = src["idx"]; r = np.arange(len(idx))
            sf = {"fast":0.96,"balanced":0.85,"premium":0.80}[t]
            pick = lab.allocate(ps, pc, MULTS[t], sf)
            pl = pc[:,0].sum(); tl = lab.true_c[idx][:,0].sum()
            psel = pc[r,pick].sum(); tsel = lab.true_c[idx][r,pick].sum()
            pr = psel/pl; tr_ = tsel/tl
            print(f"{tag:10s} {t:9s} {sf:5.2f} | {name:6s} {pr:10.4f} {tr_:10.4f} {tr_/pr:6.3f} | "
                  f"{pl:8.4f} {tl:8.4f} {pl/tl:6.3f} {psel/tsel:16.3f}")
    # per-model cost-ratio calibration
    print(f"  -- {tag}: mean predicted c_m/c_light vs true, by set")
    for name, src in (("trainOOF", cv), ("dev", arr)):
        idx = src["idx"]
        _ps, pc = lab.compose(src, DEPLOYED_CFG, "fast")
        pr = pc / pc[:, [0]]; tr_ = lab.true_c[idx] / lab.true_c[idx][:, [0]]
        print(f"     {name:8s} pred sum-ratio mid={pc[:,1].sum()/pc[:,0].sum():.3f} "
              f"k1={pc[:,2].sum()/pc[:,0].sum():.3f} | true mid="
              f"{lab.true_c[idx][:,1].sum()/lab.true_c[idx][:,0].sum():.3f} "
              f"k1={lab.true_c[idx][:,2].sum()/lab.true_c[idx][:,0].sum():.3f}")
