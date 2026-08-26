# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a04 / step 10 -- does the targeted tail guard unblock score-side improvement?

E42's blocker: a better score head moves the allocator onto items whose cost is
under-predicted, so the bust probability rises and the safety factor has to fall,
eating the gain.  If that mechanism is carried by the small runaway-cost tail,
then guarding that tail should let score-head improvements convert at (or near)
the fixed deployed safety factors.

Test: blend the predicted score toward the truth (diag6's lam ladder, an
idealised "better score head") with and without the guard, at the DEPLOYED
safety factors and at the dev-best safety factors.
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

def cost_of(t, K, infl):
    pc = P[f"cost_{t}"].copy()
    if K:
        pc[order[:K], 1] *= infl
        pc[order[:K], 2] *= infl
    pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * 1.0000001)
    pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * 1.0000001)
    return pc

print("=" * 108)
print("STEP 10  score-head quality ladder x tail guard   (S = (1-lam)*pred + lam*true)")
print("=" * 108)
print(f"{'guard':>14s} {'lam':>5s} | {'final @ deployed safety':>26s} | {'final @ dev-best safety':>24s}")
for K, infl in [(0, 1.0), (25, 20.0), (50, 20.0)]:
    for lam in (0.0, 0.05, 0.10, 0.25):
        dep_tot = 0.0; best_tot = 0.0; dep_parts = []
        for t in TIERS:
            pc = cost_of(t, K, infl)
            S = (1 - lam) * P[f"score_{t}"] + lam * dv.score
            r = tier_result(S, pc, dv, t, SAFE[t])
            dep_tot += TIER_WEIGHT[t] * r["tier_score"]
            dep_parts.append(f"{t[:1]}{r['score']:.3f}{'' if r['passed'] else '!'}")
            best = None
            for sf in np.arange(0.60, 1.201, 0.005):
                rr = tier_result(S, pc, dv, t, float(sf))
                if rr["passed"] and (best is None or rr["score"] > best[0]):
                    best = (rr["score"], float(sf))
            best_tot += TIER_WEIGHT[t] * best[0]
        g = "none" if K == 0 else f"K={K},x{infl:.0f}"
        print(f"{g:>14s} {lam:5.2f} | {dep_tot:10.4f}  {' '.join(dep_parts):>14s} | {best_tot:24.4f}")
    print()

print("=" * 108)
print("STEP 10b  the same ladder, but the safety factors are FROZEN at the deployed values")
print("  and the score head is perturbed instead of improved (robustness of the gain)")
print("=" * 108)
rng = np.random.default_rng(9)
sig = np.array([np.sqrt(((P['score_fast'][:, j] - dv.score[:, j]) ** 2).mean()) for j in range(3)])
for K, infl in [(0, 1.0), (25, 20.0)]:
    for lam in (0.0, 0.10):
        tot = np.zeros(120)
        for rep in range(120):
            s = 0.0
            for t in TIERS:
                pc = cost_of(t, K, infl)
                S = (1 - lam) * P[f"score_{t}"] + lam * dv.score
                S = np.clip(S + rng.normal(0, 1, (n, 3)) * sig * 0.10, 0, 1)
                s += TIER_WEIGHT[t] * tier_result(S, pc, dv, t, SAFE[t])["tier_score"]
            tot[rep] = s
        g = "none" if K == 0 else f"K={K},x{infl:.0f}"
        print(f"  guard {g:>12s} lam={lam:.2f}: EV over 120 jittered heads = {tot.mean():.4f} "
              f"(sd {tot.std():.4f}, p5 {np.percentile(tot,5):.4f}, frac<0.65 {np.mean(tot<0.65):.2f})")
