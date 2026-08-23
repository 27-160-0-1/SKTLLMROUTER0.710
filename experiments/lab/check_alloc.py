import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, MULTS, TIERS, DEPLOYED_CFG
import protocol as P
import gainlab as G

lab = Lab(verbose=False)
cv, arr = G.stage1(lab)
rng = np.random.default_rng(0)
for t in TIERS:
    for src in (arr, cv):
        ps, pc = lab.compose(src, DEPLOYED_CFG, t)
        for sf in (0.85, 0.95, 1.0):
            a = lab.allocate(ps, pc, MULTS[t], sf)
            b = P.exact_allocate(ps, pc, MULTS[t], sf)
            idx = src["idx"]; r = np.arange(len(idx))
            sa = lab.true_s[idx][r, a].mean(); sb = lab.true_s[idx][r, b].mean()
            ca = lab.true_c[idx][r, a].sum() / lab.true_c[idx][:, 0].sum()
            cb = lab.true_c[idx][r, b].sum() / lab.true_c[idx][:, 0].sum()
            print(f"{t:9s} n={len(idx):5d} sf={sf}: agree={np.mean(a==b):.4f} "
                  f"score {sa:.6f} vs {sb:.6f}  ratio {ca:.4f} vs {cb:.4f}")
# speed
m = len(cv["idx"]); smp = np.asarray(lab.samples_for(m, 7, 150, 880))
ps, pc = lab.compose(cv, DEPLOYED_CFG, "premium")
t0 = time.perf_counter(); P.batch_allocate(ps[smp], pc[smp], 4.0, 0.88); t1 = time.perf_counter()
P.exact_allocate(ps[smp], pc[smp], 4.0, 0.88); t2 = time.perf_counter()
print(f"batch bisection {t1-t0:.3f}s   exact envelope {t2-t1:.3f}s")
