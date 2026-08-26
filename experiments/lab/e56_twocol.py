# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""E56 - does a second, stronger model column pay?

Column A: skt/A.X-3.1-Light Q6_K (the organiser's own ax31-light).
Column B: Qwen2.5-14B-Instruct Q4_K_M, a genuine capability step above it.
The cross-column delta is a direct proxy for the upgrade gain the allocator ranks.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, DEPLOYED_EXP, DEPLOYED_CFG
import bench2 as B
import priorfeat as PF

CORE = dict(DEPLOYED_EXP, legacy_oof_meta=True, meta_seeds=(11, 23, 37, 53, 71))
lab = Lab()
pr = PF._prompts()
colA = PF.load_column(["local-llm/labels_axlight.jsonl", "local-llm/labels_ext.jsonl"], pr)
colB = PF.load_column(["local-llm/labels_qwen14b.jsonl"], pr)
print(f"col A '{colA['tag']}' {len(colA['entries'])} entries | "
      f"col B '{colB['tag']}' {len(colB['entries'])} entries", flush=True)

FA = PF.build_features(lab.texts, lab.fam_arr, [colA])
FAB = PF.build_features(lab.texts, lab.fam_arr, [colA, colB])
print("coverage A:", {k: round(v, 2) for k, v in PF.coverage_report(FA, lab.fam_arr).items()})
print("coverage B:", round(float(FAB[:, PF.NCOL].mean()), 3))

both = (FAB[:, 0] > 0) & (FAB[:, PF.NCOL] > 0) & (FAB[:, 1] > 0) & (FAB[:, PF.NCOL + 1] > 0)
if both.sum() > 50:
    sA = FAB[both, 2]; sB = FAB[both, PF.NCOL + 2]
    d1 = lab.true_s[both, 1] - lab.true_s[both, 0]
    d2 = lab.true_s[both, 2] - lab.true_s[both, 1]
    print(f"  n={both.sum()}  corr(A, s_light)={np.corrcoef(sA, lab.true_s[both,0])[0,1]:+.3f}"
          f"  corr(B, s_mid)={np.corrcoef(sB, lab.true_s[both,1])[0,1]:+.3f}")
    print(f"  corr(B-A, d1)={np.corrcoef(sB-sA, d1)[0,1]:+.3f}"
          f"  corr(B-A, d2)={np.corrcoef(sB-sA, d2)[0,1]:+.3f}"
          f"  mean(B-A)={np.mean(sB-sA):+.3f}")

res = []
for name, F in (("no prior", None), ("col A only", FA), ("col A + col B", FAB)):
    lab.set_extra_features(F)
    tag = {"no prior": "core5", "col A only": "prior100", "col A + col B": "twocol"}[name]
    cv, arr = B.stage(lab, CORE, tag=tag, force=(name != "no prior"))
    res.append(B.run(lab, cv, arr, DEPLOYED_CFG, label=name))
for frac in (0.7, 0.5):
    lab.set_extra_features(PF.mask_coverage(FAB, frac, seed=3))
    cv, arr = B.stage(lab, CORE, tag=f"twocol{int(frac*100)}", force=True)
    res.append(B.run(lab, cv, arr, DEPLOYED_CFG, label=f"col A + col B @ {frac:.0%} coverage"))
lab.set_extra_features(None)
Path("reports/lab/e56_twocol.json").write_text(json.dumps(
    [{k: v for k, v in r.items() if k != "curves"} for r in res], indent=2, default=float),
    encoding="utf-8")
