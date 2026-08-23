import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, MULTS, DEPLOYED_CFG
import protocol as P, gainlab as G
lab = Lab(verbose=False)
cv, arr = G.stage1(lab)
m = len(cv["idx"]); smp = np.asarray(lab.samples_for(m, 7, 100, 880))
ts = lab.true_s[cv["idx"]][smp]; tc = lab.true_c[cv["idx"]][smp]
ps, pc = lab.compose(cv, DEPLOYED_CFG, "premium")
PS, PC = ps[smp], pc[smp]
grid = np.arange(0.80, 1.041, 0.02)
ev1 = []
for g in grid:
    pick = P.batch_allocate(PS, PC, 4.0, float(g))
    real = np.take_along_axis(tc, pick[:, :, None], axis=2)[:, :, 0].sum(axis=1)
    sc = np.take_along_axis(ts, pick[:, :, None], axis=2)[:, :, 0].mean(axis=1)
    light = tc[:, :, 0].sum(axis=1)
    ev1.append(float(np.mean(np.where(real/light > 4.0, 0.0, sc))))
ev2, bust, raw = P.safety_curve(PS, PC, ts, tc, 4.0, grid)
for g, a, b in zip(grid, ev1, ev2):
    print(f"  sf={g:.2f}  bisect={a:.5f}  envelope={b:.5f}  diff={b-a:+.5f}")
# compare picks at one safety
g = 0.94
p1 = P.batch_allocate(PS, PC, 4.0, g)
p2 = P.exact_allocate(PS, PC, 4.0, g)
print("agree:", np.mean(p1 == p2), " counts1:", np.bincount(p1.ravel(), minlength=3),
      " counts2:", np.bincount(p2.ravel(), minlength=3))
b = 0
print("sample0 agree:", np.mean(p1[b] == p2[b]))
predtot1 = np.take_along_axis(PC[b], p1[b][:, None], 1).sum()
predtot2 = np.take_along_axis(PC[b], p2[b][:, None], 1).sum()
cap = PC[b][:, 0].sum() * max(1.0, 4.0*g)
print("pred totals", predtot1, predtot2, "cap", cap)
