# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a04 / step 9 -- evaluate the targeted tail guard.

Proposal: train a *rare-event* classifier for "this prompt makes the model run
away" (otok/gen above a high threshold), and inflate the predicted cost of ONLY
the top-flagged ~1-3% of items.  Unlike E32 (global heteroscedastic kappa) and
E37-CP (conformal upper bound on every item) this leaves ~98% of the decisions
untouched, so the score loss should be ~0 while the bust tail shrinks.

Measured here on dev: (i) score/ratio at deployed and best safety, (ii) bust
probability under score jitter, (iii) how much safety headroom the guard buys.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import learned_router, similarity
from ossp_router.protocol import load_input
from sklearn.ensemble import HistGradientBoostingClassifier

tr, dv = load_split("train"), load_split("dev")
n = len(dv); IDX = np.arange(n)
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
famtr = np.array([similarity.classify_family(t) for t in tr.texts])
famdv = np.array([similarity.classify_family(t) for t in dv.texts])
INT = re.compile(r"\d+")

def hand(texts):
    out = []
    for t in texts:
        d = [len(m.group()) for m in INT.finditer(t[:4000])]
        out.append([max(d) if d else 0, len(d), len(t)])
    return np.asarray(out, dtype=float)

ep_tr = list(load_input(ROOT / "data/materialized/train/inputs.json").episodes)
ep_dv = list(load_input(ROOT / "data/materialized/dev/inputs.json").episodes)
FAMS = list(similarity.FAMILY_NAMES)
Xtr = np.hstack([np.array([learned_router.raw_dense_features(e) for e in ep_tr]),
                 np.stack([(famtr == f).astype(float) for f in FAMS], 1), hand(tr.texts)])
Xdv = np.hstack([np.array([learned_router.raw_dense_features(e) for e in ep_dv]),
                 np.stack([(famdv == f).astype(float) for f in FAMS], 1), hand(dv.texts)])
GBM = dict(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=15,
           l2_regularization=3.0, random_state=11)

print("=" * 100)
print("STEP 9a  cross-model correlation of runaway behaviour (train)")
print("=" * 100)
o_tr = tr.otok / tr.ngen
print("  rank corr of otok/gen across models (train):")
R = np.vstack([np.argsort(np.argsort(o_tr[:, j])) for j in range(3)])
print("   ", np.round(np.corrcoef(R), 3))
print(f"  P(mid otok/gen>4000 | k1 otok/gen>12000) = "
      f"{np.mean(o_tr[o_tr[:,2]>12000,1]>4000):.3f}  base {np.mean(o_tr[:,1]>4000):.4f}")

# one runaway head: P(k1 otok/gen > 12000), used as the flag for BOTH models
y = ((tr.otok[:, 2] / tr.ngen[:, 2]) > 12000).astype(int)
clf = HistGradientBoostingClassifier(**GBM).fit(Xtr, y)
flag = clf.predict_proba(Xdv)[:, 1]
order = np.argsort(-flag)
print(f"  dev flag: item 677 rank {int(np.where(order==677)[0][0])}/880 (p={flag[677]:.3f}); "
      f"item 7 rank {int(np.where(order==7)[0][0])} (p={flag[7]:.3f}); "
      f"item 292 rank {int(np.where(order==292)[0][0])} (p={flag[292]:.3f})")

print()
print("=" * 100)
print("STEP 9b  guard: for the top-K flagged items, inflate the predicted cost of mid and k1")
print("=" * 100)
def guarded(t, K, infl):
    pc = P[f"cost_{t}"].copy()
    top = order[:K]
    pc[top, 1] *= infl
    pc[top, 2] *= infl
    pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * 1.0000001)
    pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * 1.0000001)
    return pc

def best_safety(t, pc):
    best = None
    for sf in np.arange(0.60, 1.201, 0.005):
        r = tier_result(P[f"score_{t}"], pc, dv, t, float(sf))
        if r["passed"] and (best is None or r["score"] > best[0]):
            best = (r["score"], float(sf), r["ratio"])
    return best

print(f"  {'K':>4s} {'infl':>6s} | " + " | ".join(f"{t:>28s}" for t in TIERS) + " | final@dep  final@best")
for K, infl in [(0, 1.0), (10, 5.0), (10, 20.0), (25, 5.0), (25, 20.0), (50, 5.0), (50, 20.0), (100, 5.0)]:
    cells = []; f_dep = 0.0; f_best = 0.0
    for t in TIERS:
        pc = guarded(t, K, infl)
        r = tier_result(P[f"score_{t}"], pc, dv, t, SAFE[t])
        b = best_safety(t, pc)
        f_dep += TIER_WEIGHT[t] * r["tier_score"]; f_best += TIER_WEIGHT[t] * b[0]
        cells.append(f"{r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else '!'} b{b[0]:.4f}@{b[1]:.2f}")
    print(f"  {K:4d} {infl:6.1f} | " + " | ".join(f"{c:>28s}" for c in cells) + f" | {f_dep:.4f}    {f_best:.4f}")

print()
print("=" * 100)
print("STEP 9c  does the guard remove the jitter fragility?")
print("=" * 100)
rng = np.random.default_rng(5)
sig = np.array([np.sqrt(((P['score_fast'][:, j] - dv.score[:, j]) ** 2).mean()) for j in range(3)])
for t in TIERS:
    for K, infl in [(0, 1.0), (25, 20.0), (50, 20.0)]:
        pc = guarded(t, K, infl)
        for scale in (0.05, 0.25):
            busts = 0; scs = []
            for _ in range(200):
                S = np.clip(P[f"score_{t}"] + rng.normal(0, 1, (n, 3)) * sig * scale, 0, 1)
                r = tier_result(S, pc, dv, t, SAFE[t])
                busts += (not r["passed"]); scs.append(r["tier_score"])
            print(f"  {t:9s} K={K:3d} infl={infl:5.1f} noise x{scale:<5.2f}: bust {busts:3d}/200  EV={np.mean(scs):.4f}")
    print()
