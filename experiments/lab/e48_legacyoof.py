import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_EXP, DEPLOYED_CFG
import bench as B
lab = Lab()
cv0, arr0 = B.stage(lab, DEPLOYED_EXP, tag="base", force=True)
B.bench(lab, cv0, arr0, DEPLOYED_CFG, label="baseline (legacy in-sample meta)")
exp1 = dict(DEPLOYED_EXP, legacy_oof_meta=True)
cv1, arr1 = B.stage(lab, exp1, tag="legoof", force=True)
B.bench(lab, cv1, arr1, DEPLOYED_CFG, label="legacy OOF meta feature")
