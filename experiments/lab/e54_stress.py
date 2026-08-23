import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, DEPLOYED_EXP, DEPLOYED_CFG
import bench2 as B
CORE = dict(DEPLOYED_EXP, legacy_oof_meta=True, meta_seeds=(11, 23, 37, 53, 71))
lab = Lab()
cv, arr = B.stage(lab, CORE, tag="core5")
print("--- plain bootstrap only (previous rule) ---")
B.run(lab, cv, arr, DEPLOYED_CFG, label="core, plain", scenarios=("plain",))
print("--- min-regret over {plain, runaway, inflate} ---")
r = B.run(lab, cv, arr, DEPLOYED_CFG, label="core, stressed", keep_curves=True)
import numpy as np
for t in TIERS:
    c = r["curves"][t]; g=np.array(c["grid"]); ev=np.array(c["ev"]); bu=np.array(c["bust"])
    sel=np.arange(0,len(g),max(1,len(g)//12))
    print(f" {t}: " + "  ".join(f"{g[i]:.3f}:{ev[i]:.4f}/{bu[i]*100:.0f}%" for i in sel))
