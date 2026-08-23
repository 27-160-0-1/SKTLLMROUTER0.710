# SPDX-License-Identifier: Apache-2.0
"""a04 / step 13 -- the control E32/E39 demand: does the targeted tail guard beat
the global safety scalar, or is it just "spend less" in disguise?

For every configuration we sweep the safety factor per tier and report the
bootstrap-EV-optimal point.  If the guard's optimum is above the baseline's
optimum, the guard carries information the scalar cannot.
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
prob = clf.predict_proba(Xdv)[:, 1]
order = np.argsort(-prob)


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


def draws(seed, m=300):
    r = np.random.default_rng(seed)
    return [r.integers(0, n, size=880) for _ in range(m)]
SAMPLES = {s: draws(s) for s in (7, 17, 23)}
GRID = {"fast": np.arange(0.80, 1.061, 0.02), "balanced": np.arange(0.70, 1.001, 0.02),
        "premium": np.arange(0.60, 1.001, 0.02)}


def tier_ev(pc, t, sf, seed):
    ps = P[f"score_{t}"]; evs = []; busts = 0
    for s in SAMPLES[seed]:
        pick = allocate(ps[s], pc[s], TIER_MULT[t], float(sf))
        r = np.arange(len(s))
        ratio = dv.cost[s][r, pick].sum() / dv.cost[s][:, 0].sum()
        if ratio > TIER_MULT[t]:
            busts += 1; evs.append(0.0)
        else:
            evs.append(dv.score[s][r, pick].mean())
    return float(np.mean(evs)), busts / len(SAMPLES[seed])


def mkcost(kind, t):
    pc = P[f"cost_{t}"].copy()
    if kind[0] == "hard":
        K, infl = kind[1], kind[2]
        pc[order[:K], 1] *= infl; pc[order[:K], 2] *= infl
    elif kind[0] == "soft":
        m = 1.0 + kind[1] * prob
        pc[:, 1] *= m; pc[:, 2] *= m
    elif kind[0] == "flatmid":      # control: inflate mid+k1 for EVERY item (no information)
        pc[:, 1] *= kind[1]; pc[:, 2] *= kind[1]
    pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * 1.0000001)
    pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * 1.0000001)
    return pc


CFG = [("baseline", ("none",)),
       ("hard K=150 x20", ("hard", 150, 20.0)),
       ("hard K=100 x20", ("hard", 100, 20.0)),
       ("soft gamma=20", ("soft", 20.0)),
       ("soft gamma=10", ("soft", 10.0)),
       ("flat x1.2 (no info)", ("flatmid", 1.2)),
       ("flat x1.5 (no info)", ("flatmid", 1.5)),
       ("flat x2.0 (no info)", ("flatmid", 2.0))]

print("=" * 108)
print("STEP 13  safety-optimised bootstrap EV per configuration (300 draws x 3 seeds)")
print("  Each tier's safety factor is re-optimised for EVERY configuration -- this is the")
print("  control E32/E39 require: does the guard beat what the global scalar can already do?")
print("=" * 108)
print(f"{'config':22s} | " + " | ".join(f"{t:>26s}" for t in TIERS) + " |   weighted")
for name, kind in CFG:
    tot = 0.0; cells = []
    for t in TIERS:
        pc = mkcost(kind, t)
        best = None
        for sf in GRID[t]:
            evs = [tier_ev(pc, t, sf, s) for s in (7, 17, 23)]
            m = float(np.mean([e[0] for e in evs])); b = float(np.mean([e[1] for e in evs]))
            if best is None or m > best[0]:
                best = (m, float(sf), b)
        tot += TIER_WEIGHT[t] * best[0]
        cells.append(f"EV{best[0]:.4f}@{best[1]:.2f} bust{100*best[2]:4.1f}%")
    print(f"{name:22s} | " + " | ".join(f"{c:>26s}" for c in cells) + f" |   {tot:.4f}")
