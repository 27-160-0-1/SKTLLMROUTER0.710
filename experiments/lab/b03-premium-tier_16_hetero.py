# SPDX-License-Identifier: Apache-2.0
"""b03 step 16: per-ITEM heteroscedastic cost re-transformation (a09's best premium
variant) under the honest protocol, plus the E42 selection-bias diagnostic."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P
import importlib
lib = importlib.import_module("b03-premium-tier_lib")
from sklearn.ensemble import HistGradientBoostingRegressor

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
ci, di = cv["idx"], arr["idx"]
ts, tc = lab.true_s[ci], lab.true_c[ci]
m = len(ci); MULT = 4.0
inner = lib.inner_folds(m)
ps0, pc0 = lab.compose(cv, DEPLOYED_CFG, "premium")
psd, pcd = lab.compose(arr, DEPLOYED_CFG, "premium")

def feats(a, pc):
    idx = a["idx"]
    return np.hstack([lab.dense[idx], lab.fam_onehot[idx], a["legacy"], a["lin"],
                      a["knn"], a["meta"], np.log(pc)])
Xc, Xd = feats(cv, pc0), feats(arr, pcd)
GP = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
          l2_regularization=3.0, early_stopping=True, validation_fraction=0.15, random_state=11)
res_c = np.log(tc) - np.log(pc0)
S_cv = np.zeros((m, 3)); S_dv = np.zeros((len(di), 3))
for j in range(3):
    y = res_c[:, j] ** 2
    for k in range(lib.FOLDS):
        tr, te = inner != k, inner == k
        S_cv[te, j] = HistGradientBoostingRegressor(**GP).fit(Xc[tr], y[tr]).predict(Xc[te])
    S_dv[:, j] = HistGradientBoostingRegressor(**GP).fit(Xc, y).predict(Xd)
S_cv = np.clip(S_cv, 0.0, None); S_dv = np.clip(S_dv, 0.0, None)
print("mean sigma^2 hat (cv) L/M/K =", S_cv.mean(axis=0).round(3),
      " realised var =", res_c.var(axis=0).round(3),
      " corr(sig2hat, res^2) k1 =", round(float(np.corrcoef(S_cv[:, 2], res_c[:, 2] ** 2)[0, 1]), 3))

GRID = np.arange(0.50, 1.601, 0.005)
TUNE, EVAL = (7, 17), (101, 103, 107)

def evaluate(ps, pc, psD, pcD, label):
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
    pk = P.exact_allocate(psD, pcD, MULT, sf)
    r = np.arange(len(di))
    rt = lab.true_c[di][r, pk].sum() / lab.true_c[di][:, 0].sum()
    sc = lab.true_s[di][r, pk].mean()
    # E42 selection-bias diagnostic on the OOF pool at that safety
    pkc = P.exact_allocate(ps0, pc0, MULT, sf) if label == "base" else P.exact_allocate(ps, pc, MULT, sf)
    up = pkc > 0
    bias = (tc[np.arange(m)[up], pkc[up]].sum() / pc0[np.arange(m)[up], pkc[up]].sum())
    print(f"  {label:<34} safety={sf:5.3f} premEV={x.mean():.4f} bust={100*np.mean(x==0):5.2f}%"
          f"  devPrem={sc:.4f}/r{rt:.3f} n_k1={int((pk==2).sum()):3d}"
          f"  true/pred cost of upgraded set={bias:.3f}")
    return x

print("\n=== per-item heteroscedastic re-transformation ===")
b = evaluate(ps0, pc0, psd, pcd, "base")
for kp in (0.25, 0.5, 1.0):
    x = evaluate(ps0, pc0 * np.exp(kp * S_cv), psd, pcd * np.exp(kp * S_dv),
                 f"GBM sigma^2 kappa={kp}")
    print(f"      paired dEV = {(x-b).mean():+.4f} +- {(x-b).std()/np.sqrt(len(x)):.4f}"
          f"  sign%={100*np.mean(x>b):.1f}")
# k1 column only
for kp in (0.5, 1.0):
    S2 = np.ones((m, 3)); S2[:, 2] = np.exp(kp * S_cv[:, 2])
    S2d = np.ones((len(di), 3)); S2d[:, 2] = np.exp(kp * S_dv[:, 2])
    x = evaluate(ps0, pc0 * S2, psd, pcd * S2d, f"GBM sigma^2 k1 column only kappa={kp}")
    print(f"      paired dEV = {(x-b).mean():+.4f} +- {(x-b).std()/np.sqrt(len(x)):.4f}"
          f"  sign%={100*np.mean(x>b):.1f}")

print("\n=== E42 diagnostic: new d2 head vs base ===")
Z = np.load("reports/lab/b03_d2head.npz")
ps_new = ps0.copy(); ps_new[:, 2] = np.clip(ps0[:, 1] + Z["r_cv"], 0, 1)
psd_new = psd.copy(); psd_new[:, 2] = np.clip(psd[:, 1] + Z["r_dv"], 0, 1)
evaluate(ps_new, pc0, psd_new, pcd, "new d2 head (beta=1)")
