# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b04 - end-to-end value of a LEARNED long-k1-output veto, under bench2.

a11 measured the ORACLE veto frontier (+0.0041 EV at a 5k-token threshold) and
left the detector as an open question.  _16_tailclf.py measures what a detector
can reach.  This closes the loop: the detector is fitted out-of-fold on Train,
the veto is applied to BOTH the OOF rows (so the safety ratio is chosen for the
vetoed pipeline) and to Dev, and bench2 scores it.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")
from harness import DEPLOYED_EXP, DEPLOYED_CFG, TIERS
import bench2 as B
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

BASE_EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)
OUT = Path("reports/lab/b04_veto.json")
GP = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20,
          l2_regularization=3.0, early_stopping=True, validation_fraction=0.15, random_state=11)


def detector_scores(lab, thr):
    """Out-of-fold scores on Train, train-fitted scores on Dev.  Never sees a row's
    own label, and never sees Dev at fit time."""
    tr, dv = lab.train_idx, lab.dev_idx
    opg = lab.otok[:, 2] / np.maximum(lab.ngen[:, 2], 1)
    y = (opg >= thr).astype(int)
    Xd = np.hstack([lab.dense, lab.fam_onehot])
    out = np.zeros(lab.n)
    fold = np.random.default_rng(3).integers(0, 5, size=len(tr))
    for k in range(5):
        f, h = tr[fold != k], tr[fold == k]
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear").fit(lab.X[f], y[f])
        s = lr.decision_function(lab.X)
        inner = np.random.default_rng(5).integers(0, 4, size=len(f))
        o = np.zeros(len(f))
        for j in range(4):
            g = f[inner != j]
            o[inner == j] = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear").fit(
                lab.X[g], y[g]).decision_function(lab.X[f[inner == j]])
        m = HistGradientBoostingClassifier(**GP).fit(np.hstack([Xd[f], o[:, None]]), y[f])
        out[h] = m.decision_function(np.hstack([Xd[h], s[h][:, None]]))
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear").fit(lab.X[tr], y[tr])
    s = lr.decision_function(lab.X)
    inner = np.random.default_rng(5).integers(0, 4, size=len(tr))
    o = np.zeros(len(tr))
    for j in range(4):
        g = tr[inner != j]
        o[inner == j] = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear").fit(
            lab.X[g], y[g]).decision_function(lab.X[tr[inner == j]])
    m = HistGradientBoostingClassifier(**GP).fit(np.hstack([Xd[tr], o[:, None]]), y[tr])
    out[dv] = m.decision_function(np.hstack([Xd[dv], s[dv][:, None]]))
    return out, y


def make_veto(score, width, oracle_y=None):
    def tr(_lab, arr, ps, pc, tier):
        rows = arr["idx"]
        if oracle_y is not None:
            bad = oracle_y[rows] > 0
        else:
            s = score[rows]
            k = int(round(width * len(rows)))
            bad = np.zeros(len(rows), bool)
            bad[np.argsort(-s)[:k]] = True
        ps = ps.copy()
        ps[bad, 2] = -1e6
        return ps, pc
    return tr


if __name__ == "__main__":
    lab = lib.MemLab(verbose=False)
    cv, arr = B.stage(lab, BASE_EXP, tag="b04_R1")
    res = {}
    r = B.run(lab, cv, arr, DEPLOYED_CFG, label="R1 (no veto)")
    res["base"] = {k: v for k, v in r.items() if k != "curves"}
    for thr in (5_000, 10_000):
        sc, y = detector_scores(lab, thr)
        r = B.run(lab, cv, arr, DEPLOYED_CFG, transform=make_veto(None, None, oracle_y=y),
                  label=f"ORACLE veto >={thr//1000}k")
        res[f"oracle{thr}"] = {k: v for k, v in r.items() if k != "curves"}
        for w in (0.05, 0.10, 0.15, 0.30):
            r = B.run(lab, cv, arr, DEPLOYED_CFG, transform=make_veto(sc, w),
                      label=f"learned veto >={thr//1000}k, width {int(w*100)}%")
            res[f"learn{thr}_{int(w*100)}"] = {k: v for k, v in r.items() if k != "curves"}
        rng = np.random.default_rng(9)
        for w in (0.15, 0.30):
            r = B.run(lab, cv, arr, DEPLOYED_CFG, transform=make_veto(rng.random(lab.n), w),
                      label=f"RANDOM veto width {int(w*100)}%")
            res[f"rand_{int(w*100)}"] = {k: v for k, v in r.items() if k != "curves"}
    OUT.write_text(json.dumps(res, indent=1, default=float))
