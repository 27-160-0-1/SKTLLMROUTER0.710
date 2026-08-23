import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab
lab = Lab(verbose=False)
art = json.load(open("src/ossp_router/resources/hash-regex-public.v1.json", encoding="utf-8"))
fm = np.array(art["feature_mean"]); fs = np.array(art["feature_scale"])
M = lab.legacy_raw[lab.train_idx]
print("mean maxdiff:", np.abs(M.mean(0)-fm).max(), " scale maxdiff:", np.abs(np.where(M.std(0)>1e-12,M.std(0),1.0)-fs).max())
for alpha in (0.1, 1, 10, 100, 300, 1000):
    h = lab.fit_legacy(lab.train_idx, alpha)
    p = lab.predict_legacy(h, np.arange(lab.n))
    d = np.abs(p - lab.legacy).max()
    c = np.mean([np.corrcoef(p[:,k], lab.legacy[:,k])[0,1] for k in range(6)])
    print(f"alpha={alpha:6}: maxdiff={d:.5f} corr={c:.6f}")
