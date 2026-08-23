import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, DEPLOYED_CFG
import protocol as P, gainlab as G
lab = Lab(verbose=False)
cv, arr = G.stage1(lab)
m = len(cv["idx"])
for t in TIERS:
    ps, pc = lab.compose(cv, DEPLOYED_CFG, t)
    grid = G.GRIDS[t]
    smp = np.asarray(lab.samples_for(m, 7, 300, 880))
    ev, bu, raw = P.safety_curve(ps[smp], pc[smp], lab.true_s[cv["idx"]][smp],
                                 lab.true_c[cv["idx"]][smp], MULTS[t], grid)
    print(f"{t}:")
    for g, e, b, r in zip(grid, ev, bu, raw):
        print(f"   sf={g:.2f} ev={e:.4f} bust={b:.3f} raw={r:.4f}")
    # also: full-train OOF (no bootstrap)
    pick = lab.allocate(ps, pc, MULTS[t], 0.90)
    rr = np.arange(m)
    print(f"   full-OOF @0.90: score={lab.true_s[cv['idx']][rr,pick].mean():.4f} "
          f"ratio={lab.true_c[cv['idx']][rr,pick].sum()/lab.true_c[cv['idx']][:,0].sum():.4f}")
