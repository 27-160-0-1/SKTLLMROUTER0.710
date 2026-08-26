# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""A fast-tier-specialised binary head: light-vs-mid only.

The fast tier never buys k1 (measured worth 0.0000 in _09), so its whole decision
is one binary comparison.  This trains a dedicated head on exactly that target,
stacked on top of the honest OOF predictions, and scores it with the decision
statistic (within-family AUC of d1) and with the realised frontier at 1.25x.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b01lib", HERE / "b01-fast-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP  # noqa
import bench2 as B
import protocol as P

lab = Lab(); MF = 1.25
cv0, arr0 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
new = np.array([L.classify_v3(t) for t in lab.texts])
GP = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=30,
          l2_regularization=3.0, early_stopping=True, validation_fraction=0.15, random_state=11)


def block(a):
    r = a["idx"]
    return np.hstack([lab.dense[r], lab.fam_onehot[r], a["legacy"], a["lin"], a["knn"],
                      a["meta"], a["gain"], a["rank_eff"]])


Xc, Xd = block(cv0), block(arr0)
ic, idv = cv0["idx"], arr0["idx"]
d1_true = lab.true_s[:, 1] - lab.true_s[:, 0]
yc = (d1_true[ic] > 0).astype(int)


def nested(X, y, model, folds=5, seed=5):
    f = np.random.default_rng(seed).integers(0, folds, size=len(y))
    out = np.zeros(len(y))
    for k in range(folds):
        m = model().fit(X[f != k], y[f != k])
        out[f == k] = (m.predict_proba(X[f == k])[:, 1] if hasattr(m, "predict_proba")
                       else m.predict(X[f == k]))
    return out


def auc(x, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(x); n1 = y.sum(); n0 = (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def wf(x, y, fam):
    num = den = 0.0
    for f in np.unique(fam):
        s = fam == f
        yy = y[s]
        if yy.all() or (~yy).all():
            continue
        w = yy.sum() * (~yy).sum(); num += auc(x[s], yy) * w; den += w
    return num / den


HEADS = {
    "binary  1[s_m>s_l]": (lambda: HistGradientBoostingClassifier(**GP), yc),
    "regress d1": (lambda: HistGradientBoostingRegressor(**GP), d1_true[ic]),
}
res = {}
for nm, (mk, tgt) in HEADS.items():
    oof = nested(Xc, tgt, mk)
    m = mk().fit(Xc, tgt)
    dv = (m.predict_proba(Xd)[:, 1] if hasattr(m, "predict_proba") else m.predict(Xd))
    res[nm] = (oof, dv)
    print(f"{nm:20s} AUC_wf train-OOF={wf(oof, d1_true[ic] > 0, new[ic]):.4f} "
          f"dev={wf(dv, d1_true[idv] > 0, new[idv]):.4f}   "
          f"AUC_pool train={auc(oof, d1_true[ic] > 0):.4f} dev={auc(dv, d1_true[idv] > 0):.4f}",
          flush=True)

ps_c, pc_c = lab.compose(cv0, DEPLOYED_CFG, "fast")
ps_d, pc_d = lab.compose(arr0, DEPLOYED_CFG, "fast")
print(f"{'deployed d1':20s} AUC_wf train-OOF={wf(ps_c[:,1]-ps_c[:,0], d1_true[ic]>0, new[ic]):.4f} "
      f"dev={wf(ps_d[:,1]-ps_d[:,0], d1_true[idv]>0, new[idv]):.4f}")

GRID = np.arange(0.60, 1.201, 0.005)


def frontier(ps, pc, rows):
    ts = lab.true_s[rows]; tc = lab.true_c[rows]; r = np.arange(len(rows))
    best = (-1, None, None)
    for g in GRID:
        pk = P.exact_allocate(ps, pc, MF, float(g))
        rt = tc[r, pk].sum() / tc[:, 0].sum(); sc = ts[r, pk].mean()
        if rt <= MF + 1e-12 and sc > best[0]:
            best = (sc, rt, g)
    return best


def make(ps, d1, ban=True):
    s = ps.copy()
    s[:, 1] = np.clip(s[:, 0] + d1, 0, 1)
    if ban:
        s[:, 2] = -1e9
    return s


print("\n=== frontier at ratio<=1.25 (k1 banned everywhere) ===")
print(f"{'gain used':34s} {'TRAIN-OOF':>10s} {'DEV':>10s}")
b = frontier(make(ps_c, ps_c[:, 1] - ps_c[:, 0]), pc_c, ic)
b2 = frontier(make(ps_d, ps_d[:, 1] - ps_d[:, 0]), pc_d, idv)
print(f"{'deployed d1 (baseline)':34s} {b[0]:10.4f} {b2[0]:10.4f}")
for nm, (oof, dv) in res.items():
    # map the head output onto the gain scale with train-only constants
    if "binary" in nm:
        a_, b_ = np.polyfit(oof, d1_true[ic], 1)
        gc, gd = a_ * oof + b_, a_ * dv + b_
    else:
        gc, gd = oof, dv
    for lam in (1.0, 0.5):
        g1 = (1 - lam) * (ps_c[:, 1] - ps_c[:, 0]) + lam * gc
        g2 = (1 - lam) * (ps_d[:, 1] - ps_d[:, 0]) + lam * gd
        f1 = frontier(make(ps_c, g1), pc_c, ic); f2 = frontier(make(ps_d, g2), pc_d, idv)
        print(f"{nm + f'  lam={lam}':34s} {f1[0]:10.4f} {f2[0]:10.4f}")
