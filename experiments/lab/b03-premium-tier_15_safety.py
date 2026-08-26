# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b03 step 15: pick the premium safety on a robustness criterion, not the nominal argmax."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
ci, di = cv["idx"], arr["idx"]
ts, tc = lab.true_s[ci], lab.true_c[ci]
tsd, tcd = lab.true_s[di], lab.true_c[di]
m = len(ci); MULT = 4.0
fam = lab.fam_arr[ci]
ps0, pc0 = lab.compose(cv, DEPLOYED_CFG, "premium")
psd, pcd = lab.compose(arr, DEPLOYED_CFG, "premium")
og = lab.otok[ci][:, 2] / lab.ngen[ci][:, 2]

SC = {}
SC["nominal"] = np.ones(m)
w = np.ones(m); w[fam == "longdoc"] = 0.5; SC["longdoc x0.5"] = w
w = np.ones(m); w[np.isin(fam, ["code", "dmmath", "aime"])] = 3.0; SC["reasoning x3"] = w
w = np.ones(m); w[np.isin(fam, ["ruletaker", "longdoc"])] = 3.0; SC["rt/ld x3"] = w
w = 1.0 + 2.0 * (og > np.quantile(og, 0.8)); SC["long-think x3"] = w
w = np.ones(m); w[np.isin(fam, ["belebele", "truthfulqa"])] = 3.0; SC["ko/qa x3"] = w

GRID = np.arange(0.70, 0.921, 0.01)
tab = {}
for nm, w in SC.items():
    p = w / w.sum()
    ev = np.zeros(len(GRID)); bu = np.zeros(len(GRID))
    for s in (7, 17, 23):
        rng = np.random.default_rng(s)
        smp = rng.choice(m, size=(400, 880), p=p)
        e, b, _ = P.safety_curve(ps0[smp], pc0[smp], ts[smp], tc[smp], MULT, GRID)
        ev += e / 3; bu += b / 3
    tab[nm] = (ev, bu)

print(f"{'safety':>7}" + "".join(f"{n:>15}" for n in SC) + f"{'MIN':>9}{'devPrem':>9}{'devR':>8}")
for i, g in enumerate(GRID):
    pk = P.exact_allocate(psd, pcd, MULT, float(g))
    r = np.arange(len(di))
    rt = tcd[r, pk].sum() / tcd[:, 0].sum(); sc = tsd[r, pk].mean()
    row = "".join(f"{tab[n][0][i]:9.4f}/{100*tab[n][1][i]:4.1f}" for n in SC)
    mn = min(tab[n][0][i] for n in SC)
    print(f"{g:7.2f}{row}{mn:9.4f}{sc:9.4f}{rt:8.3f}{'' if rt<=4 else ' BUST'}")

nom = tab["nominal"][0]
mins = np.array([min(tab[n][0][i] for n in SC) for i in range(len(GRID))])
print(f"\nnominal argmax   safety={GRID[int(np.argmax(nom))]:.2f}  nominal EV={nom.max():.4f}"
      f"  worst-case EV={mins[int(np.argmax(nom))]:.4f}")
print(f"minimax argmax   safety={GRID[int(np.argmax(mins))]:.2f}  "
      f"nominal EV={nom[int(np.argmax(mins))]:.4f}  worst-case EV={mins.max():.4f}")
for tgt in (0.005, 0.01, 0.02):
    ok = np.where(nom >= nom.max() - tgt)[0]
    j = ok[int(np.argmax(mins[ok]))]
    print(f"give up {tgt:.3f} nominal -> safety={GRID[j]:.2f} worst-case EV={mins[j]:.4f}"
          f" (nominal {nom[j]:.4f})")
