# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b03 step 8: end-to-end value of (a) the new d2 head, (b) a k1-runaway predictor."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import importlib
lib = importlib.import_module("b03-premium-tier_lib")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
ci = cv["idx"]; di = arr["idx"]
Z = np.load("reports/lab/b03_d2head.npz")
r_cv, r_dv = Z["r_cv"], Z["r_dv"]

# ---------------- runaway detector: y = 1{k1 out/gen >= T} -----------------
def feats(a):
    idx = a["idx"]
    ps, pc = lab.compose(a, DEPLOYED_CFG, "premium")
    return np.hstack([lab.dense[idx], lab.fam_onehot[idx], a["legacy"], a["lin"],
                      a["knn"], a["meta"], np.log(pc)])
Xc, Xd = feats(cv), feats(arr)
og_c = lab.otok[ci][:, 2] / lab.ngen[ci][:, 2]
og_d = lab.otok[di][:, 2] / lab.ngen[di][:, 2]
inner = lib.inner_folds(len(ci))
GP = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
          l2_regularization=3.0, early_stopping=True, validation_fraction=0.15, random_state=11)
RUN = {}
for T in (2000, 5000, 10000):
    yc = (og_c >= T).astype(int); yd = (og_d >= T).astype(int)
    q_cv = np.zeros(len(ci))
    for k in range(lib.FOLDS):
        tr, te = inner != k, inner == k
        q_cv[te] = HistGradientBoostingClassifier(**GP).fit(Xc[tr], yc[tr]).predict_proba(Xc[te])[:, 1]
    q_dv = HistGradientBoostingClassifier(**GP).fit(Xc, yc).predict_proba(Xd)[:, 1]
    RUN[T] = (q_cv, q_dv)
    print(f"runaway >= {T:>6}/gen  base rate cv={yc.mean():.4f} dev={yd.mean():.4f}"
          f"  AUC cv={roc_auc_score(yc,q_cv):.3f} dev={roc_auc_score(yd,q_dv):.3f}")
    for pct in (0.05, 0.10, 0.20):
        thr = np.quantile(q_cv, 1 - pct)
        rec_c = yc[q_cv >= thr].sum() / max(yc.sum(), 1)
        thr_d = np.quantile(q_dv, 1 - pct)
        rec_d = yd[q_dv >= thr_d].sum() / max(yd.sum(), 1)
        print(f"    veto top {pct:.0%}: recall cv={rec_c:.2f} dev={rec_d:.2f}"
              f"  precision cv={yc[q_cv>=thr].mean():.3f}")
np.savez("reports/lab/b03_runaway.npz", **{f"q{T}_{s}": RUN[T][i]
         for T in RUN for i, s in enumerate(("cv", "dv"))})

# ---------------- transforms -----------------------------------------------
def mk(beta=0.0, veto_T=None, veto_pct=0.0, tiers=("premium",)):
    qc, qd = (RUN[veto_T] if veto_T else (None, None))
    thr = np.quantile(qc, 1 - veto_pct) if veto_T else None
    def tr(lab_, a, ps, pc, tier):
        if tier not in tiers:
            return ps, pc
        ps = ps.copy(); pc = pc.copy()
        isc = len(a["idx"]) == len(ci)
        if beta > 0:
            d2 = (1 - beta) * (ps[:, 2] - ps[:, 1]) + beta * (r_cv if isc else r_dv)
            ps[:, 2] = np.clip(ps[:, 1] + d2, 0, 1)
        if veto_T:
            q = qc if isc else qd
            ps[q >= thr, 2] = 0.0            # remove k1 from the menu of flagged items
        return ps, pc
    return tr

base = B.run(lab, cv, arr, DEPLOYED_CFG, label="A0 legoof base")
out = {"A0": base}
for b in (0.25, 0.5, 0.75, 1.0):
    out[f"d2b{b}"] = B.run(lab, cv, arr, DEPLOYED_CFG, transform=mk(beta=b),
                           label=f"new d2 head beta={b} (prem)")
out["d2all"] = B.run(lab, cv, arr, DEPLOYED_CFG, transform=mk(beta=0.5, tiers=TIERS),
                     label="new d2 head beta=0.5 (all tiers)")
for T in (2000, 5000):
    for p in (0.05, 0.10, 0.20):
        out[f"v{T}_{p}"] = B.run(lab, cv, arr, DEPLOYED_CFG, transform=mk(veto_T=T, veto_pct=p),
                                 label=f"runaway veto T={T} top{p:.0%} (prem)")
print("\npremium detail:")
for k, r in out.items():
    d = r["det"]["premium"]; t = r["dev_tiers"]["premium"]
    print(f"  {k:<12} premEV={d['ev']:.4f} bust={d['bust']*100:5.2f}% safety={r['safety']['premium']:.3f}"
          f"  devPrem={t['score']:.4f}/r{t['ratio']:.3f}  EV={r['EV']:.6f} dev={r['dev']:.6f}")
