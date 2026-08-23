import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_EXP, DEPLOYED_CFG, DEPLOYED_SAFETY
import bench2 as B
lab = Lab()
cv0, arr0 = B.stage(lab, DEPLOYED_EXP, tag="base")
cv1, arr1 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
print("--- EV-optimal safety (honest) ---")
r0 = B.run(lab, cv0, arr0, DEPLOYED_CFG, label="baseline")
r1 = B.run(lab, cv1, arr1, DEPLOYED_CFG, label="legacy-OOF meta")
print("--- at the E43 deployed safety .98/.87/.85 ---")
B.run(lab, cv0, arr0, DEPLOYED_CFG, label="baseline @E43", fixed_safety=DEPLOYED_SAFETY)
B.run(lab, cv1, arr1, DEPLOYED_CFG, label="legacy-OOF @E43", fixed_safety=DEPLOYED_SAFETY)
