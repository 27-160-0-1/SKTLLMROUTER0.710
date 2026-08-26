# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b03 step 18: end-to-end value of the runaway-probability cost correction; are the
truncated items ever selected?"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P
import importlib
lib = importlib.import_module("b03-premium-tier_lib")

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
ci, di = cv["idx"], arr["idx"]
ts, tc = lab.true_s[ci], lab.true_c[ci]
m = len(ci); MULT = 4.0
ps0, pc0 = lab.compose(cv, DEPLOYED_CFG, "premium")
psd, pcd = lab.compose(arr, DEPLOYED_CFG, "premium")
inner = lib.inner_folds(m)
Z = np.load("reports/lab/b03_runaway.npz")
r_c = np.log(tc[:, 2]) - np.log(pc0[:, 2])

def corr_for(T, scale=1.0):
    q_c, q_d = Z[f"q{T}_cv"], Z[f"q{T}_dv"]
    cc = np.zeros(m)
    for k in range(lib.FOLDS):
        trn, te = inner != k, inner == k
        X = np.column_stack([np.ones(trn.sum()), q_c[trn]])
        b = np.linalg.lstsq(X, r_c[trn], rcond=None)[0]
        cc[te] = b[0] + b[1] * q_c[te]
    X = np.column_stack([np.ones(m), q_c]); ball = np.linalg.lstsq(X, r_c, rcond=None)[0]
    return np.exp(scale * cc), np.exp(scale * (ball[0] + ball[1] * q_d))

GRID = np.arange(0.50, 1.601, 0.005); TUNE, EVAL = (7, 17), (101, 103, 107)
def evaluate(pcC, pcD, label, base=None):
    ev = np.zeros(len(GRID))
    for s in TUNE:
        smp = np.asarray(lab.samples_for(m, s, 400, 880))
        e, _b, _r = P.safety_curve(ps0[smp], pcC[smp], ts[smp], tc[smp], MULT, GRID)
        ev += e / len(TUNE)
    sf = float(GRID[int(np.argmax(ev))])
    out = []
    for s in EVAL:
        smp = np.asarray(lab.samples_for(m, s, 400, 880))
        pk = P.exact_allocate(ps0[smp], pcC[smp], MULT, sf)
        C = np.take_along_axis(tc[smp], pk[:, :, None], axis=2)[:, :, 0]
        S = np.take_along_axis(ts[smp], pk[:, :, None], axis=2)[:, :, 0]
        R = C.sum(axis=1) / tc[smp][:, :, 0].sum(axis=1)
        out.append(np.where(R > MULT, 0.0, S.mean(axis=1)))
    x = np.concatenate(out)
    pk = P.exact_allocate(psd, pcD, MULT, sf); r = np.arange(len(di))
    rt = lab.true_c[di][r, pk].sum() / lab.true_c[di][:, 0].sum()
    sc = lab.true_s[di][r, pk].mean()
    d = "" if base is None else f"  dEV={(x-base).mean():+.4f}+-{(x-base).std()/np.sqrt(len(x)):.4f} sign%={100*np.mean(x>base):.0f}"
    print(f"  {label:<40} safety={sf:5.3f} premEV={x.mean():.4f} bust={100*np.mean(x==0):5.2f}%"
          f"  devPrem={sc:.4f}/r{rt:.3f}{d}")
    return x

b = evaluate(pc0, pcd, "base")
for T in (2000, 5000):
    for sc in (0.5, 1.0):
        fc, fd = corr_for(T, sc)
        pcC = pc0.copy(); pcC[:, 2] *= fc
        pcD = pcd.copy(); pcD[:, 2] *= fd
        pcC[:, 2] = np.maximum(pcC[:, 2], pcC[:, 1] * (1 + 1e-12))
        pcD[:, 2] = np.maximum(pcD[:, 2], pcD[:, 1] * (1 + 1e-12))
        evaluate(pcC, pcD, f"runaway-prob cost correction T={T} x{sc}", b)

print("\nare the 10 truncated items ever selected?  (all splits, all tiers)")
igo = (lab.itok[:, 2] + lab.otok[:, 2]) / lab.ngen[:, 2]
hit = np.abs(igo - 32768) < 1.0
sfty = {"fast": 0.960, "balanced": 0.825, "premium": 0.840}
for t in TIERS:
    for nm, a in (("train-OOF", cv), ("dev", arr)):
        ps, pc = lab.compose(a, DEPLOYED_CFG, t)
        pk = P.exact_allocate(ps, pc, MULTS[t], sfty[t])
        h = hit[a["idx"]]
        print(f"  {t:<9} {nm:<10} truncated items in pool={int(h.sum())}"
              f"  picked k1={int((pk[h]==2).sum())}  picked mid={int((pk[h]==1).sum())}"
              f"  predicted k1 cost rank (mean pct)="
              f"{100*np.mean([(pc[:,2] < pc[i,2]).mean() for i in np.where(h)[0]]):.0f}")
