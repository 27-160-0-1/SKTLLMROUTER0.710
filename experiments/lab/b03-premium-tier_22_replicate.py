# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b03 step 22: replicate everything on an independent CV fold seed (777)."""
import sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P
import importlib
lib = importlib.import_module("b03-premium-tier_lib")

lab = Lab()
cv1, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
cv2 = pickle.loads(Path("reports/lab/b03_cv_seed777.pkl").read_bytes())

print("EV-optimal safety and EV, two independent fold seeds (dev arr identical)")
for nm, c in (("seed123", cv1), ("seed777", cv2)):
    r = B.run(lab, c, arr, DEPLOYED_CFG, label=f"legoof {nm}")

SF = {"fast": 0.960, "balanced": 0.825, "premium": 0.840}
print(f"\n{'pool':<10}{'tier':<10}{'ratio':>8}{'b_N/b_D':>9}{'sd(logR)':>10}{'bust%':>8}")
for nm, x in (("seed123", cv1), ("seed777", cv2), ("dev", arr)):
    idx = x["idx"]; ts = lab.true_s[idx]; tc = lab.true_c[idx]; n = len(idx)
    for t in TIERS:
        ps, pc = lab.compose(x, DEPLOYED_CFG, t)
        pk = P.exact_allocate(ps, pc, MULTS[t], SF[t]); r = np.arange(n)
        rt = tc[r, pk].sum() / tc[:, 0].sum()
        bN = tc[r, pk].sum() / pc[r, pk].sum(); bD = tc[:, 0].sum() / pc[:, 0].sum()
        bu = 0.0
        for s in (7, 17, 23):
            smp = np.asarray(lab.samples_for(n, s, 400, 880))
            _e, b, _ = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MULTS[t], np.array([SF[t]]))
            bu += b[0] / 3
        smp = np.asarray(lab.samples_for(n, 7, 400, 880))
        pk2 = P.exact_allocate(ps[smp], pc[smp], MULTS[t], SF[t])
        C = np.take_along_axis(tc[smp], pk2[:, :, None], axis=2)[:, :, 0]
        R = C.sum(axis=1) / tc[smp][:, :, 0].sum(axis=1)
        print(f"{nm:<10}{t:<10}{rt:8.4f}{bN/bD:9.3f}{np.log(R).std():10.4f}{100*bu:8.2f}")

# ---- premium candidates on the replicate pool -----------------------------
print("\npremium candidates, replicate fold seed 777 (paired, independent eval seeds)")
ci = cv2["idx"]; ts = lab.true_s[ci]; tc = lab.true_c[ci]; m = len(ci)
fam = lab.fam_arr[ci]; inner = lib.inner_folds(m)
ps0, pc0 = lab.compose(cv2, DEPLOYED_CFG, "premium")
S = np.zeros((m, 3))
for k in range(lib.FOLDS):
    sg = lib.fit_sigma2(lab, ci[inner != k], pc0[inner != k])
    S[inner == k] = lib.apply_sigma2(sg, fam[inner == k], 1.0)
GRID = np.arange(0.50, 1.601, 0.005); TUNE, EVAL = (7, 17), (101, 103, 107)
def ev_of(ps, pc, label, base=None):
    ev = np.zeros(len(GRID))
    for s in TUNE:
        smp = np.asarray(lab.samples_for(m, s, 400, 880))
        e, _b, _r = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], 4.0, GRID); ev += e / 2
    sf = float(GRID[int(np.argmax(ev))]); out = []
    for s in EVAL:
        smp = np.asarray(lab.samples_for(m, s, 400, 880))
        pk = P.exact_allocate(ps[smp], pc[smp], 4.0, sf)
        C = np.take_along_axis(tc[smp], pk[:, :, None], axis=2)[:, :, 0]
        Sc = np.take_along_axis(ts[smp], pk[:, :, None], axis=2)[:, :, 0]
        R = C.sum(axis=1) / tc[smp][:, :, 0].sum(axis=1)
        out.append(np.where(R > 4.0, 0.0, Sc.mean(axis=1)))
    x = np.concatenate(out)
    d = "" if base is None else f"  dEV={(x-base).mean():+.4f}+-{(x-base).std()/np.sqrt(len(x)):.4f}"
    print(f"  {label:<24} safety={sf:5.3f} premEV={x.mean():.4f} bust={100*np.mean(x==0):5.2f}%{d}")
    return x
b = ev_of(ps0, pc0, "base")
ev_of(ps0, pc0 * [1, 1, 1.24], "k2=1.24", b)
ev_of(ps0, pc0 * [1, 1, 1.50], "k2=1.50", b)
ev_of(ps0, pc0 * S ** 0.5, "var-form k=0.5", b)
ev_of(ps0, pc0 * S, "var-form k=1.0", b)
# rescue
r_cv = np.zeros(m)
for k in range(lib.FOLDS):
    co = lib.fit_rescue(ps0[inner != k, 1], ts[inner != k, 2], fam[inner != k])
    r_cv[inner == k] = lib.apply_rescue(co, ps0[inner == k, 1], fam[inner == k])
ps_r = ps0.copy(); ps_r[:, 2] = r_cv
ev_of(ps_r, pc0, "C6 rescue alpha=1", b)
