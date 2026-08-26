# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a04 / step 8 -- are the runaway generations (the cost tail that busts a tier)
predictable from the prompt?

Target: output_tokens/generation above a high threshold (a degenerate/looping
generation).  Tested with (i) a hand rule (long integer literal / family) and
(ii) a HistGradientBoosting classifier on the dense features, train -> dev.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import learned_router, similarity
from ossp_router.protocol import load_input
from sklearn.ensemble import HistGradientBoostingClassifier

tr, dv = load_split("train"), load_split("dev")
famtr = np.array([similarity.classify_family(t) for t in tr.texts])
famdv = np.array([similarity.classify_family(t) for t in dv.texts])
INT = re.compile(r"\d+")

def hand(texts):
    out = []
    for t in texts:
        digs = [len(m.group()) for m in INT.finditer(t[:4000])]
        out.append([max(digs) if digs else 0, len(digs), len(t)])
    return np.asarray(out, dtype=float)

Htr, Hdv = hand(tr.texts), hand(dv.texts)

print("=" * 100)
print("STEP 8a  prevalence of runaway generations")
print("=" * 100)
for sp, nm in ((tr, "train"), (dv, "dev")):
    o = sp.otok / sp.ngen
    for j in range(3):
        print(f"  {nm:5s} {MODEL_IDS[j]:11s} otok/gen p50={np.median(o[:,j]):6.0f} p99={np.percentile(o[:,j],99):7.0f} "
              f"max={o[:,j].max():7.0f} | frac>4000={np.mean(o[:,j]>4000):.4f} frac>8000={np.mean(o[:,j]>8000):.4f}")

print()
print("=" * 100)
print("STEP 8b  hand rule: longest integer literal in the prompt")
print("=" * 100)
for j in (1, 2):
    thr = 4000 if j == 1 else 12000
    for sp, H, fam, nm in ((tr, Htr, famtr, "train"), (dv, Hdv, famdv, "dev")):
        y = (sp.otok[:, j] / sp.ngen[:, j]) > thr
        for cut in (6, 8, 10):
            m = H[:, 0] >= cut
            if m.sum() == 0:
                continue
            print(f"  {nm:5s} {MODEL_IDS[j]:11s} otok/gen>{thr}: n_pos={y.sum():3d} | rule maxdigits>={cut}: "
                  f"n={m.sum():4d} precision={y[m].mean():.3f} recall={y[m].sum()/max(y.sum(),1):.3f}")
    print()

print("=" * 100)
print("STEP 8c  learned blow-up classifier (dense features, train -> dev)")
print("=" * 100)
ep_tr = list(load_input(ROOT / "data/materialized/train/inputs.json").episodes)
ep_dv = list(load_input(ROOT / "data/materialized/dev/inputs.json").episodes)
Dtr = np.array([learned_router.raw_dense_features(e) for e in ep_tr])
Ddv = np.array([learned_router.raw_dense_features(e) for e in ep_dv])
FAMS = list(similarity.FAMILY_NAMES)
Otr = np.stack([(famtr == f).astype(float) for f in FAMS], 1)
Odv = np.stack([(famdv == f).astype(float) for f in FAMS], 1)
Xtr = np.hstack([Dtr, Otr, Htr]); Xdv = np.hstack([Ddv, Odv, Hdv])
GBM = dict(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=15,
           l2_regularization=3.0, random_state=11)
for j in (1, 2):
    for thr in ((4000, 8000) if j == 1 else (12000, 20000)):
        ytr = ((tr.otok[:, j] / tr.ngen[:, j]) > thr).astype(int)
        ydv = ((dv.otok[:, j] / dv.ngen[:, j]) > thr).astype(int)
        if ytr.sum() < 5:
            print(f"  {MODEL_IDS[j]:11s} thr={thr}: only {ytr.sum()} train positives -- skipped")
            continue
        m = HistGradientBoostingClassifier(**GBM).fit(Xtr, ytr)
        p = m.predict_proba(Xdv)[:, 1]
        o = np.argsort(-p)
        prec_at = {k: ydv[o[:k]].sum() / k for k in (10, 25, 50)}
        rec_at = {k: ydv[o[:k]].sum() / max(ydv.sum(), 1) for k in (10, 25, 50)}
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(ydv, p) if 0 < ydv.sum() < len(ydv) else float("nan")
        print(f"  {MODEL_IDS[j]:11s} thr={thr:6d}: train_pos={ytr.sum():3d} dev_pos={ydv.sum():3d} AUC={auc:.3f} "
              f"| prec@10={prec_at[10]:.2f} @25={prec_at[25]:.2f} @50={prec_at[50]:.2f} "
              f"| recall@50={rec_at[50]:.2f}")

print()
print("=" * 100)
print("STEP 8d  how much of the tier cost do the blow-ups carry?")
print("=" * 100)
for sp, nm in ((tr, "train"), (dv, "dev")):
    LT = sp.cost[:, 0].sum()
    for j in (1, 2):
        o = sp.otok[:, j] / sp.ngen[:, j]
        top = np.argsort(-o)[:5]
        share_if_all = (sp.cost[:, j].sum() - sp.cost[:, 0].sum()) / LT
        print(f"  {nm:5s} {MODEL_IDS[j]:11s}: top-5 items carry "
              f"{(sp.cost[top,j]-sp.cost[top,0]).sum()/LT:.3f} of the light budget; "
              f"upgrading everything costs {share_if_all:.2f}")
