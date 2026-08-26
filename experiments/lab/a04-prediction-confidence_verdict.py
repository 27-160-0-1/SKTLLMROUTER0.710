# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a04 / step 14 -- the fair head-to-head the control demands.

The guard's win at FIXED deployed safety could be nothing but "spend less".
Compare, under the same score-head jitter on the real (un-resampled) dev set:
   (A) baseline at deployed safety
   (B) baseline at a LOWER safety chosen to remove the same bust risk
   (C) guard at deployed safety
If (B) >= (C) the guard adds nothing over the global scalar (E32/E39 verdict holds).
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import learned_router, similarity
from ossp_router.protocol import load_input
from sklearn.ensemble import HistGradientBoostingClassifier

tr, dv = load_split("train"), load_split("dev")
n = len(dv)
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
famtr = np.array([similarity.classify_family(t) for t in tr.texts])
famdv = np.array([similarity.classify_family(t) for t in dv.texts])
INT = re.compile(r"\d+")
hand = lambda ts: np.asarray([[max([len(m.group()) for m in INT.finditer(t[:4000])] or [0]),
                               len(INT.findall(t[:4000])), len(t)] for t in ts], dtype=float)
FAMS = list(similarity.FAMILY_NAMES)
ep_tr = list(load_input(ROOT / "data/materialized/train/inputs.json").episodes)
ep_dv = list(load_input(ROOT / "data/materialized/dev/inputs.json").episodes)
Xtr = np.hstack([np.array([learned_router.raw_dense_features(e) for e in ep_tr]),
                 np.stack([(famtr == f).astype(float) for f in FAMS], 1), hand(tr.texts)])
Xdv = np.hstack([np.array([learned_router.raw_dense_features(e) for e in ep_dv]),
                 np.stack([(famdv == f).astype(float) for f in FAMS], 1), hand(dv.texts)])
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                     min_samples_leaf=15, l2_regularization=3.0,
                                     random_state=11).fit(
    Xtr, ((tr.otok[:, 2] / tr.ngen[:, 2]) > 12000).astype(int))
order = np.argsort(-clf.predict_proba(Xdv)[:, 1])


def cost_of(t, K=0, infl=1.0):
    pc = P[f"cost_{t}"].copy()
    if K:
        pc[order[:K], 1] *= infl; pc[order[:K], 2] *= infl
    pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * 1.0000001)
    pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * 1.0000001)
    return pc


sig = np.array([np.sqrt(((P['score_fast'][:, j] - dv.score[:, j]) ** 2).mean()) for j in range(3)])
rng = np.random.default_rng(31)
JIT = {sc: [rng.normal(0, 1, (n, 3)) * sig * sc for _ in range(200)] for sc in (0.0, 0.05, 0.10, 0.25)}


def ev(safe, K, infl, scale):
    tot = 0.0; det = []
    for t in TIERS:
        pc = cost_of(t, K, infl)
        vals, busts = [], 0
        for z in JIT[scale]:
            S = np.clip(P[f"score_{t}"] + z, 0, 1)
            r = tier_result(S, pc, dv, t, safe[t])
            vals.append(r["tier_score"]); busts += (not r["passed"])
        tot += TIER_WEIGHT[t] * np.mean(vals)
        det.append(f"{t[:1]}{np.mean(vals):.4f}/{100*busts/len(JIT[scale]):.0f}%")
    return tot, " ".join(det)


print("=" * 100)
print("STEP 14  real dev (no resampling), score head jittered, 200 draws")
print("=" * 100)
CONF = [("A baseline @ deployed .98/.87/.85", SAFE, 0, 1.0),
        ("B baseline @ .96/.86/.84", {"fast": .96, "balanced": .86, "premium": .84}, 0, 1.0),
        ("B baseline @ .94/.85/.82", {"fast": .94, "balanced": .85, "premium": .82}, 0, 1.0),
        ("B baseline @ .92/.84/.80", {"fast": .92, "balanced": .84, "premium": .80}, 0, 1.0),
        ("C guard K=25 x20 @ deployed", SAFE, 25, 20.0),
        ("C guard K=150 x20 @ deployed", SAFE, 150, 20.0),
        ("D guard K=25 x20 @ .94/.85/.82", {"fast": .94, "balanced": .85, "premium": .82}, 25, 20.0)]
print(f"{'config':34s} " + " ".join(f"{'x%.2f' % s:>10s}" for s in (0.0, 0.05, 0.10, 0.25)) + "   detail@x0.10")
for name, safe, K, infl in CONF:
    row = []; det = ""
    for sc in (0.0, 0.05, 0.10, 0.25):
        v, d = ev(safe, K, infl, sc)
        row.append(v)
        if sc == 0.10:
            det = d
    print(f"{name:34s} " + " ".join(f"{v:10.4f}" for v in row) + f"   {det}")
