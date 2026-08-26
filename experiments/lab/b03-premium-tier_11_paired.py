# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b03 step 11: PAIRED bootstrap comparison of premium-tier candidates.

Safety for each variant is chosen on seeds (7,17) and the comparison is scored on
an independent seed (101,103,107) so the safety choice is not tuned on the
evaluation resamples.  The same resamples are used for every variant (paired).
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P
import importlib
lib = importlib.import_module("b03-premium-tier_lib")

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
ci, di = cv["idx"], arr["idx"]
ts, tc = lab.true_s[ci], lab.true_c[ci]
m = len(ci); MULT = 4.0
fam_cv, fam_dv = lab.fam_arr[ci], lab.fam_arr[di]
inner = lib.inner_folds(len(ci))
ps0, pc0 = lab.compose(cv, DEPLOYED_CFG, "premium")
psd, pcd = lab.compose(arr, DEPLOYED_CFG, "premium")

# variance-form factors (cross-fit on cv, full-fit for dev)
S_cv = np.zeros((m, 3))
for k in range(lib.FOLDS):
    sg = lib.fit_sigma2(lab, ci[inner != k], pc0[inner != k])
    S_cv[inner == k] = lib.apply_sigma2(sg, fam_cv[inner == k], 1.0)
S_dv = lib.apply_sigma2(lib.fit_sigma2(lab, ci, pc0), fam_dv, 1.0)

VAR = {
    "base":            (ps0, pc0, psd, pcd),
    "k2=1.24":         (ps0, pc0 * [1, 1, 1.24], psd, pcd * [1, 1, 1.24]),
    "k2=1.50":         (ps0, pc0 * [1, 1, 1.50], psd, pcd * [1, 1, 1.50]),
    "var k=0.5":       (ps0, pc0 * S_cv ** 0.5, psd, pcd * S_dv ** 0.5),
    "var k=1.0":       (ps0, pc0 * S_cv, psd, pcd * S_dv),
    "var0.5 + k2=1.24": (ps0, pc0 * S_cv ** 0.5 * [1, 1, 1.24], psd, pcd * S_dv ** 0.5 * [1, 1, 1.24]),
}
GRID = np.arange(0.55, 1.401, 0.005)
TUNE, EVAL = (7, 17), (101, 103, 107)

def curve(ps, pc, seeds, nboot=400):
    ev = np.zeros(len(GRID)); bu = np.zeros(len(GRID))
    per = []
    for s in seeds:
        smp = np.asarray(lab.samples_for(m, s, nboot, 880))
        e, b, _r = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MULT, GRID)
        ev += e / len(seeds); bu += b / len(seeds)
    return ev, bu

def per_sample(ps, pc, safety, seeds, nboot=400):
    out = []
    for s in seeds:
        smp = np.asarray(lab.samples_for(m, s, nboot, 880))
        pk = P.exact_allocate(ps[smp], pc[smp], MULT, safety)
        C = np.take_along_axis(tc[smp], pk[:, :, None], axis=2)[:, :, 0]
        S = np.take_along_axis(ts[smp], pk[:, :, None], axis=2)[:, :, 0]
        R = C.sum(axis=1) / tc[smp][:, :, 0].sum(axis=1)
        out.append(np.where(R > MULT, 0.0, S.mean(axis=1)))
    return np.concatenate(out)

res = {}
for nm, (ps, pc, pd_s, pd_c) in VAR.items():
    ev, bu = curve(ps, pc, TUNE)
    gi = int(np.argmax(ev)); sf = float(GRID[gi])
    x = per_sample(ps, pc, sf, EVAL)
    pk = P.exact_allocate(pd_s, pd_c, MULT, sf)
    r = np.arange(len(di))
    rt = lab.true_c[di][r, pk].sum() / lab.true_c[di][:, 0].sum()
    sc = lab.true_s[di][r, pk].mean()
    res[nm] = dict(x=x, safety=sf, bust=float(np.mean(x == 0)), ev=float(x.mean()),
                   dev=float(sc), ratio=float(rt), nk1=int((pk == 2).sum()))
b = res["base"]
print(f"{'variant':<18}{'safety':>8}{'premEV':>9}{'bust%':>7}{'dEV':>9}{'SE':>8}{'sign%':>7}"
      f"{'devPrem':>9}{'devRatio':>9}{'n_k1':>6}")
for nm, r in res.items():
    d = r["x"] - b["x"]
    se = d.std() / np.sqrt(len(d))
    print(f"{nm:<18}{r['safety']:8.3f}{r['ev']:9.4f}{100*r['bust']:7.2f}{d.mean():+9.4f}"
          f"{se:8.4f}{100*np.mean(d>0):7.1f}{r['dev']:9.4f}{r['ratio']:9.3f}{r['nk1']:6d}")
print("\n(paired over 1,200 independent-seed resamples; SE is the paired SE, "
      "but resamples share the same 1,760-row pool so it is NOT a data SE)")
