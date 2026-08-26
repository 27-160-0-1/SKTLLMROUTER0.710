# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b06 forest-view: is the oracle partition RECOVERABLE from the prompt?

Script 06 showed that an oracle K-way partition of the true 6-vector, used as a
pure lookup table of K x 6 constants, beats the whole 129k-parameter stack
(9-way dev 0.717, 18-way 0.740).  That is an upper bound built from labels.

Here the partition is made honest end to end:
  * k-means centroids fitted on TRAIN targets only,
  * a GBM classifier maps the meta features -> cluster, fitted out-of-fold for
    the Train-OOF rows and on all of Train for Dev,
  * the prediction is the (soft or hard) cluster-mean target vector.
Nothing reads a Dev label.  If the honest version lands near the family-mean
rung, the partition idea is dead; if it lands well above the full stack, it is
the cheapest remaining route.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP, _gbm_params
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
import bench2 as B

OUT = Path("reports/lab/b06_partition.json")


def feat(lab, a):
    return np.hstack([lab.dense[a["idx"]], lab.fam_onehot[a["idx"]],
                      a["legacy"], a["lin"], a["knn"]])


def build(lab, cv, arr, K, channel="full", seed=0, soft=True):
    """Returns pred6 arrays for the cv rows and the dev rows, plus diagnostics."""
    tr = cv["idx"]                                   # the 1,760 Train rows
    T = lab.targets
    if channel == "full":
        Z = T[tr].copy()
    elif channel == "gain":                          # decision-relevant channel only
        Z = np.column_stack([T[tr, 1] - T[tr, 0], T[tr, 2] - T[tr, 1],
                             T[tr, 4] - T[tr, 3], T[tr, 5] - T[tr, 4]])
    Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-9)
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(Z)
    lab_tr = km.labels_
    cmean = np.vstack([T[tr][lab_tr == g].mean(axis=0) for g in range(K)])

    Xc = feat(lab, cv); Xd = feat(lab, arr)
    gp = _gbm_params(DEPLOYED_EXP)
    fold = np.random.default_rng(9).integers(0, 5, size=len(tr))
    Pc = np.zeros((len(tr), K))
    for f in range(5):
        m = HistGradientBoostingClassifier(**gp).fit(Xc[fold != f], lab_tr[fold != f])
        pr = m.predict_proba(Xc[fold == f])
        Pc[np.ix_(fold == f, m.classes_)] = pr
    mall = HistGradientBoostingClassifier(**gp).fit(Xc, lab_tr)
    Pd = np.zeros((len(arr["idx"]), K))
    Pd[:, mall.classes_] = mall.predict_proba(Xd)
    acc = float((Pc.argmax(1) == lab_tr).mean())
    if not soft:
        Pc = np.eye(K)[Pc.argmax(1)]; Pd = np.eye(K)[Pd.argmax(1)]
    return Pc @ cmean, Pd @ cmean, acc, lab_tr, cmean


def mk_t(pred_cv, pred_dev, ncv):
    def t(lab, a, ps, pc, tier):
        P6 = pred_cv if len(a["idx"]) == ncv else pred_dev
        p = np.clip(P6[:, :3], 0, 1)
        c = np.exp(np.clip(P6[:, 3:6], -50, 50))
        c[:, 1] = np.maximum(c[:, 1], c[:, 0] * (1 + 1e-12))
        c[:, 2] = np.maximum(c[:, 2], c[:, 1] * (1 + 1e-12))
        return p, c
    return t


def mk_blend(pred_cv, pred_dev, ncv, w):
    """Blend the honest partition prediction into the deployed composition."""
    def t(lab, a, ps, pc, tier):
        P6 = pred_cv if len(a["idx"]) == ncv else pred_dev
        p = np.clip((1 - w) * ps + w * np.clip(P6[:, :3], 0, 1), 0, 1)
        c = np.exp((1 - w) * np.log(np.maximum(pc, 1e-300)) + w * np.clip(P6[:, 3:6], -50, 50))
        c[:, 1] = np.maximum(c[:, 1], c[:, 0] * (1 + 1e-12))
        c[:, 2] = np.maximum(c[:, 2], c[:, 1] * (1 + 1e-12))
        return p, c
    return t


def main():
    lab = Lab()
    cv, arr = B.stage(lab, DEPLOYED_EXP, tag="base")
    ncv = len(cv["idx"])
    base = B.run(lab, cv, arr, None, label="BASE full stack")
    rep = {"base": dict(EV=base["EV"], dev=base["dev"]), "runs": []}

    for channel in ("full", "gain"):
        for K in (9, 18, 40):
            t0 = time.perf_counter()
            pc_, pd_, acc, ltr, cm = build(lab, cv, arr, K, channel=channel)
            r = B.run(lab, cv, arr, None, transform=mk_t(pc_, pd_, ncv),
                      label=f"honest {channel} K={K} (acc {acc:.3f})")
            rep["runs"].append(dict(channel=channel, K=K, acc=acc, mode="lookup",
                                    EV=r["EV"], dev=r["dev"],
                                    safety={t: r["safety"][t] for t in TIERS}))
            for w in (0.25, 0.5):
                rb = B.run(lab, cv, arr, None, transform=mk_blend(pc_, pd_, ncv, w),
                           label=f"   blend w={w} into full stack")
                rep["runs"].append(dict(channel=channel, K=K, acc=acc, mode=f"blend{w}",
                                        EV=rb["EV"], dev=rb["dev"]))
            print(f"   ({time.perf_counter()-t0:.0f}s)", flush=True)

    OUT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
