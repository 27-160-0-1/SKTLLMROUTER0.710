# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b03 step 24: pooled-risk premium safety (mean over 3 independent risk pools +
the 6 shift scenarios), and the final shipped numbers."""
import sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P

lab = Lab()
cv1, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
cv2 = pickle.loads(Path("reports/lab/b03_cv_seed777.pkl").read_bytes())
POOLS = {"oof123": cv1, "oof777": cv2, "dev": arr}
G = np.arange(0.65, 0.901, 0.01)

curves = {}
for nm, x in POOLS.items():
    idx = x["idx"]; ts = lab.true_s[idx]; tc = lab.true_c[idx]; n = len(idx)
    ps, pc = lab.compose(x, DEPLOYED_CFG, "premium")
    ev = np.zeros(len(G)); bu = np.zeros(len(G))
    for s in (7, 17, 23):
        smp = np.asarray(lab.samples_for(n, s, 400, 880))
        e, b, _ = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], 4.0, G)
        ev += e / 3; bu += b / 3
    curves[nm] = (ev, bu)
# shift scenarios on the seed-123 OOF pool
ci = cv1["idx"]; fam = lab.fam_arr[ci]; m = len(ci)
og = lab.otok[ci][:, 2] / lab.ngen[ci][:, 2]
ps0, pc0 = lab.compose(cv1, DEPLOYED_CFG, "premium")
SC = {"rt/ld x3": np.where(np.isin(fam, ["ruletaker", "longdoc"]), 3.0, 1.0),
      "long-think x3": 1.0 + 2.0 * (og > np.quantile(og, 0.8)),
      "reasoning x3": np.where(np.isin(fam, ["code", "dmmath", "aime"]), 3.0, 1.0)}
for nm, w in SC.items():
    p = w / w.sum(); ev = np.zeros(len(G))
    for s in (7, 17, 23):
        rng = np.random.default_rng(s)
        smp = rng.choice(m, size=(400, 880), p=p)
        e, _b, _ = P.safety_curve(ps0[smp], pc0[smp], lab.true_s[ci][smp],
                                  lab.true_c[ci][smp], 4.0, G)
        ev += e / 3
    curves[nm] = (ev, None)

names = list(curves)
print(f"{'safety':>7}" + "".join(f"{n:>15}" for n in names) + f"{'mean3pool':>11}{'min-all':>9}")
M = np.array([curves[n][0] for n in names])
p3 = M[:3].mean(axis=0); mn = M.min(axis=0)
for i, g in enumerate(G):
    print(f"{g:7.2f}" + "".join(f"{M[j,i]:15.4f}" for j in range(len(names)))
          + f"{p3[i]:11.4f}{mn[i]:9.4f}")
print(f"\nargmax mean-of-3-pools : safety {G[int(np.argmax(p3))]:.2f}  ({p3.max():.4f})")
print(f"argmax min-over-all    : safety {G[int(np.argmax(mn))]:.2f}  ({mn.max():.4f})")

SHIP = 0.76
print(f"\n=== SHIPPED PREMIUM POLICY: C1 (legacy-OOF meta) + premium safety {SHIP} ===")
r = B.run(lab, cv1, arr, DEPLOYED_CFG, label="shipped (fast/bal at bench2 argmax)",
          fixed_safety={"fast": 0.960, "balanced": 0.825, "premium": SHIP})
i = int(np.argmin(np.abs(G - SHIP)))
print(f"  premium EV  oof123={curves['oof123'][0][i]:.4f} oof777={curves['oof777'][0][i]:.4f}"
      f" dev={curves['dev'][0][i]:.4f}")
print(f"  premium bust oof123={100*curves['oof123'][1][i]:.2f}% oof777={100*curves['oof777'][1][i]:.2f}%"
      f" dev={100*curves['dev'][1][i]:.2f}%")
print(f"  premium EV under shift: " + ", ".join(
    f"{n}={curves[n][0][i]:.4f}" for n in SC))
