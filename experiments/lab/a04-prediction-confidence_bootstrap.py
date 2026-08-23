# SPDX-License-Identifier: Apache-2.0
"""a04 / step 11 -- the project's own judgement harness: 880-item bootstrap EV,
with and without the targeted runaway-cost guard, at the deployed safety factors
and on the safety grid.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT
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


def allocate(ps, pc, mult, safety):
    lt = pc[:, 0].sum(); cap = lt * max(1.0, mult * safety)
    def choose(pen):
        u = ps - pen * pc / lt
        pick = u.argmax(1)
        return pick, pc[np.arange(len(pick)), pick].sum()
    pick, tot = choose(0.0)
    if tot > cap:
        lo, hi = 0.0, 1.0
        pick, tot = choose(hi)
        while tot > cap and hi < 2 ** 60:
            lo, hi = hi, hi * 2
            pick, tot = choose(hi)
        for _ in range(40):
            mid = (lo + hi) / 2
            c2, t2 = choose(mid)
            if t2 <= cap:
                hi, pick, tot = mid, c2, t2
            else:
                lo = mid
    if tot > cap:
        pick = np.zeros(len(ps), dtype=int)
    return pick


rng = np.random.default_rng(7)
SAMPLES = [rng.integers(0, n, size=880) for _ in range(400)]
print("=" * 100)
print("STEP 11  880-item bootstrap over dev (400 draws), deployed E43 predictions")
print("=" * 100)
print(f"{'guard':>12s} {'tier':9s} {'safety':>7s} {'EV':>8s} {'bust%':>7s} {'score|pass':>11s}")
for K, infl in [(0, 1.0), (25, 20.0), (50, 20.0)]:
    tot = {}
    for t in TIERS:
        pc = cost_of(t, K, infl)
        ps = P[f"score_{t}"]
        for sf in (SAFE[t], SAFE[t] + 0.05, SAFE[t] + 0.10):
            evs, busts, sc = [], 0, []
            for s in SAMPLES:
                pick = allocate(ps[s], pc[s], TIER_MULT[t], float(sf))
                r = np.arange(len(s))
                ratio = dv.cost[s][r, pick].sum() / dv.cost[s][:, 0].sum()
                if ratio > TIER_MULT[t]:
                    busts += 1; evs.append(0.0)
                else:
                    v = dv.score[s][r, pick].mean(); evs.append(v); sc.append(v)
            g = "none" if K == 0 else f"K={K},x{infl:.0f}"
            print(f"{g:>12s} {t:9s} {sf:7.3f} {np.mean(evs):8.4f} {100*busts/len(SAMPLES):6.1f}% "
                  f"{(np.mean(sc) if sc else 0):11.4f}")
            if abs(sf - SAFE[t]) < 1e-9:
                tot[t] = np.mean(evs)
    print(f"{'':12s} weighted EV at deployed safety = {sum(TIER_WEIGHT[t]*tot[t] for t in TIERS):.4f}\n")
