# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b03 step 7: a dedicated 'k1 beats mid' head, stacked on the pipeline outputs.

Two forms, both trained ONLY on the train-OOF rows:
  (i)  classifier  y = 1{s_k1 > s_mid}          -> p_hat
  (ii) regressor   y = s_k1 - s_mid             -> d2_hat
Honest evaluation: inner 5-fold cross-fit for the CV rows, full fit for dev.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import importlib
lib = importlib.import_module("b03-premium-tier_lib")
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
ci = cv["idx"]; di = arr["idx"]

def feats(a):
    idx = a["idx"]
    ps, pc = lab.compose(a, DEPLOYED_CFG, "premium")
    return np.hstack([lab.dense[idx], lab.fam_onehot[idx], a["legacy"], a["lin"],
                      a["knn"], a["meta"], a["gain"], a["rank_eff"],
                      ps, np.log(pc), ps[:, 1:2] - ps[:, 0:1], ps[:, 2:3] - ps[:, 1:2]])

Xc, Xd = feats(cv), feats(arr)
ts_c, ts_d = lab.true_s[ci], lab.true_s[di]
d2_c, d2_d = ts_c[:, 2] - ts_c[:, 1], ts_d[:, 2] - ts_d[:, 1]
yc, yd = (d2_c > 0).astype(int), (d2_d > 0).astype(int)
inner = lib.inner_folds(len(ci))
GP = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
          l2_regularization=3.0, early_stopping=True, validation_fraction=0.15, random_state=11)

p_cv = np.zeros(len(ci)); r_cv = np.zeros(len(ci)); m_cv = np.zeros(len(ci))
for k in range(lib.FOLDS):
    tr, te = inner != k, inner == k
    p_cv[te] = HistGradientBoostingClassifier(**GP).fit(Xc[tr], yc[tr]).predict_proba(Xc[te])[:, 1]
    r_cv[te] = HistGradientBoostingRegressor(**GP).fit(Xc[tr], d2_c[tr]).predict(Xc[te])
    m_cv[te] = HistGradientBoostingRegressor(**GP).fit(Xc[tr], ts_c[tr, 1]).predict(Xc[te])
clf = HistGradientBoostingClassifier(**GP).fit(Xc, yc)
reg = HistGradientBoostingRegressor(**GP).fit(Xc, d2_c)
mreg = HistGradientBoostingRegressor(**GP).fit(Xc, ts_c[:, 1])
p_dv = clf.predict_proba(Xd)[:, 1]; r_dv = reg.predict(Xd); m_dv = mreg.predict(Xd)

ps_c, pc_c = lab.compose(cv, DEPLOYED_CFG, "premium")
ps_d, pc_d = lab.compose(arr, DEPLOYED_CFG, "premium")
d2p_c = ps_c[:, 2] - ps_c[:, 1]; d2p_d = ps_d[:, 2] - ps_d[:, 1]

print(f"{'signal':<28}{'AUC cv':>9}{'AUC dev':>9}{'corr d2 cv':>12}{'corr d2 dev':>12}")
for nm, sc, sd_ in (("deployed d2 (composed)", d2p_c, d2p_d), ("new classifier p_hat", p_cv, p_dv),
                    ("new d2 regressor", r_cv, r_dv), ("0.5*d2 + 0.5*p_hat(z)", None, None)):
    if sc is None:
        z = lambda v: (v - v.mean()) / v.std()
        sc = 0.5 * z(d2p_c) + 0.5 * z(p_cv); sd_ = 0.5 * z(d2p_d) + 0.5 * z(p_dv)
    print(f"{nm:<28}{roc_auc_score(yc, sc):9.3f}{roc_auc_score(yd, sd_):9.3f}"
          f"{np.corrcoef(sc, d2_c)[0,1]:12.3f}{np.corrcoef(sd_, d2_d)[0,1]:12.3f}")

# within-family AUC of the new classifier
fam_c = lab.fam_arr[ci]; fam_d = lab.fam_arr[di]
print("\nwithin-family AUC  (deployed d2 -> new p_hat)")
for f in sorted(set(fam_c)):
    s, s2 = fam_c == f, fam_d == f
    if yc[s].min() == yc[s].max():
        continue
    a1 = roc_auc_score(yc[s], d2p_c[s]); a2 = roc_auc_score(yc[s], p_cv[s])
    b1 = b2 = float("nan")
    if s2.sum() > 20 and yd[s2].min() != yd[s2].max():
        b1 = roc_auc_score(yd[s2], d2p_d[s2]); b2 = roc_auc_score(yd[s2], p_dv[s2])
    print(f"  {f:<18} n={int(s.sum()):4d}  cv {a1:.3f} -> {a2:.3f}   dev {b1:.3f} -> {b2:.3f}")

# also: how well can we predict 'mid fails' (the actual driver)?
print("\nP(mid fails) head:  AUC(mid==0) cv=%.3f dev=%.3f | composed s_mid AUC cv=%.3f dev=%.3f"
      % (roc_auc_score((ts_c[:, 1] <= 0).astype(int), -m_cv),
         roc_auc_score((ts_d[:, 1] <= 0).astype(int), -m_dv),
         roc_auc_score((ts_c[:, 1] <= 0).astype(int), -ps_c[:, 1]),
         roc_auc_score((ts_d[:, 1] <= 0).astype(int), -ps_d[:, 1])))
np.savez("reports/lab/b03_d2head.npz", p_cv=p_cv, p_dv=p_dv, r_cv=r_cv, r_dv=r_dv,
         m_cv=m_cv, m_dv=m_dv)
