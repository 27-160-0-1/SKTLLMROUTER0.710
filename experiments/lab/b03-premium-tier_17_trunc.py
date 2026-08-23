# SPDX-License-Identifier: Apache-2.0
"""b03 step 17: truncation census; runaway probability as a COST correction;
rescue gate keyed on a dedicated P(mid fails) head."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P
import importlib
lib = importlib.import_module("b03-premium-tier_lib")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
ci, di = cv["idx"], arr["idx"]
ts, tc = lab.true_s[ci], lab.true_c[ci]
m = len(ci); MULT = 4.0

# ---------------- truncation census (independent of a12) -------------------
ig = lab.itok[:, 2] / lab.ngen[:, 2]
og = lab.otok[:, 2] / lab.ngen[:, 2]
tot = ig + og
CAP = 32768
hit = np.abs(tot - CAP) < 1.0
print(f"truncation census (all 2,640): items with in/gen+out/gen == {CAP}: {int(hit.sum())}")
print(f"  their s_k1 = {lab.true_s[hit,2]} ; s_mid = {lab.true_s[hit,1]}")
print(f"  families = {sorted(set(lab.fam_arr[hit]))}")
print(f"  share of all k1 cost = {100*lab.true_c[hit,2].sum()/lab.true_c[:,2].sum():.2f}%")
tr, dv = hit[:1760], hit[1760:]
for nm, s, sl in (("train", tr, slice(0, 1760)), ("dev", dv, slice(1760, 2640))):
    L = lab.true_c[sl][:, 0].sum()
    print(f"  {nm}: n={int(s.sum())}  their k1 cost = {lab.true_c[sl][s,2].sum()/L:.3f} light units"
          f" = {100*lab.true_c[sl][s,2].sum()/L/4.0:.1f}% of the premium cap")
print("\nP(s_k1 == 0) by out/gen bucket (all 2,640):")
for lo, hi in ((0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000),
               (8000, 16000), (16000, 32768)):
    s = (og >= lo) & (og < hi)
    if s.sum():
        print(f"  [{lo:5d},{hi:5d})  n={int(s.sum()):5d}  P(s_k1=0)={np.mean(lab.true_s[s,2]==0):.3f}"
              f"  mean s_k1={lab.true_s[s,2].mean():.3f}  mean s_mid={lab.true_s[s,1].mean():.3f}"
              f"  mean c_k1/c_mid={np.mean(lab.true_c[s,2]/lab.true_c[s,1]):7.1f}")

# ---------------- runaway probability as a COST correction ------------------
Z = np.load("reports/lab/b03_runaway.npz")
ps0, pc0 = lab.compose(cv, DEPLOYED_CFG, "premium")
psd, pcd = lab.compose(arr, DEPLOYED_CFG, "premium")
r_c = np.log(tc[:, 2]) - np.log(pc0[:, 2])
print(f"\nk1 log-cost error: rmse cv={np.sqrt((r_c**2).mean()):.3f}"
      f"  dev={np.sqrt(((np.log(lab.true_c[di][:,2])-np.log(pcd[:,2]))**2).mean()):.3f}")
inner = lib.inner_folds(m)
for T in (2000, 5000):
    q_c, q_d = Z[f"q{T}_cv"], Z[f"q{T}_dv"]
    # cross-fit linear correction  log c_k1 += a + b*q
    corr_c = np.zeros(m)
    for k in range(lib.FOLDS):
        trn, te = inner != k, inner == k
        X = np.column_stack([np.ones(trn.sum()), q_c[trn]])
        b = np.linalg.lstsq(X, r_c[trn], rcond=None)[0]
        corr_c[te] = b[0] + b[1] * q_c[te]
    X = np.column_stack([np.ones(m), q_c]); ball = np.linalg.lstsq(X, r_c, rcond=None)[0]
    corr_d = ball[0] + ball[1] * q_d
    rm = np.sqrt(((r_c - corr_c) ** 2).mean())
    rd = np.sqrt(((np.log(lab.true_c[di][:, 2]) - np.log(pcd[:, 2]) - corr_d) ** 2).mean())
    print(f"  + runaway prob q(T={T}) correction: rmse cv={rm:.3f} dev={rd:.3f}"
          f"   (slope={ball[1]:+.3f})")

# ---------------- rescue gate on a dedicated P(mid fails) head --------------
Zd = np.load("reports/lab/b03_d2head.npz")
pmid_c, pmid_d = -Zd["m_cv"], -Zd["m_dv"]           # higher = more likely mid fails
GRID = np.arange(0.50, 1.401, 0.005); TUNE, EVAL = (7, 17), (101, 103, 107)

def evaluate(ps, pc, psD, pcD, label, base=None):
    ev = np.zeros(len(GRID))
    for s in TUNE:
        smp = np.asarray(lab.samples_for(m, s, 400, 880))
        e, _b, _r = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MULT, GRID)
        ev += e / len(TUNE)
    sf = float(GRID[int(np.argmax(ev))])
    out = []
    for s in EVAL:
        smp = np.asarray(lab.samples_for(m, s, 400, 880))
        pk = P.exact_allocate(ps[smp], pc[smp], MULT, sf)
        C = np.take_along_axis(tc[smp], pk[:, :, None], axis=2)[:, :, 0]
        S = np.take_along_axis(ts[smp], pk[:, :, None], axis=2)[:, :, 0]
        R = C.sum(axis=1) / tc[smp][:, :, 0].sum(axis=1)
        out.append(np.where(R > MULT, 0.0, S.mean(axis=1)))
    x = np.concatenate(out)
    pk = P.exact_allocate(psD, pcD, MULT, sf); r = np.arange(len(di))
    rt = lab.true_c[di][r, pk].sum() / lab.true_c[di][:, 0].sum()
    sc = lab.true_s[di][r, pk].mean()
    d = "" if base is None else f"  dEV={(x-base).mean():+.4f}+-{(x-base).std()/np.sqrt(len(x)):.4f}"
    print(f"  {label:<36} safety={sf:5.3f} premEV={x.mean():.4f} bust={100*np.mean(x==0):5.2f}%"
          f"  devPrem={sc:.4f}/r{rt:.3f} n_k1={int((pk==2).sum()):3d}{d}")
    return x

print("\n=== rescue gate: k1 only where the mid head says mid is likely to fail ===")
b = evaluate(ps0, pc0, psd, pcd, "base")
for q in (0.30, 0.50, 0.70):
    thr = np.quantile(pmid_c, 1 - q)
    p1 = ps0.copy(); p1[pmid_c < thr, 2] = 0.0
    thr_d = np.quantile(pmid_d, 1 - q)
    p2 = psd.copy(); p2[pmid_d < thr_d, 2] = 0.0
    evaluate(p1, pc0, p2, pcd, f"k1 only in the top {q:.0%} of P(mid fails)", b)
