# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a04 / step 12 -- shape of the guard: hard top-K vs soft probability inflation,
which model column to inflate, and unimodality in K.  Judged by the 880-item
bootstrap EV at the deployed safety factors (400 draws, seeds 7/17/23).
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


def _draws(seed, m=400):
    r = np.random.default_rng(seed)
    return [r.integers(0, n, size=880) for _ in range(m)]


SAMPLES = {s: _draws(s) for s in (7, 17, 23)}


def ev(mkcost, seed):
    tot = 0.0; parts = []
    for t in TIERS:
        pc = mkcost(t); ps = P[f"score_{t}"]
        evs, busts = [], 0
        for s in SAMPLES[seed]:
            pick = allocate(ps[s], pc[s], TIER_MULT[t], SAFE[t])
            r = np.arange(len(s))
            ratio = dv.cost[s][r, pick].sum() / dv.cost[s][:, 0].sum()
            if ratio > TIER_MULT[t]:
                busts += 1; evs.append(0.0)
            else:
                evs.append(dv.score[s][r, pick].mean())
        tot += TIER_WEIGHT[t] * np.mean(evs)
        parts.append(f"{t[:1]}{np.mean(evs):.4f}/{100*busts/400:.0f}%")
    return tot, " ".join(parts)


def hardK(K, infl, cols=(1, 2)):
    def f(t):
        pc = P[f"cost_{t}"].copy()
        for c in cols:
            pc[order[:K], c] *= infl
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * 1.0000001)
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * 1.0000001)
        return pc
    return f


def soft(gamma, cols=(1, 2)):
    def f(t):
        pc = P[f"cost_{t}"].copy()
        m = 1.0 + gamma * prob
        for c in cols:
            pc[:, c] *= m
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * 1.0000001)
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * 1.0000001)
        return pc
    return f


print("=" * 100)
print("STEP 12  guard shape sweep, 880-bootstrap EV at deployed safety, 3 seeds")
print("=" * 100)
CFG = [("baseline", hardK(0, 1.0))]
for K in (10, 25, 50, 75, 100, 150, 250):
    CFG.append((f"hard K={K} x20 (mid+k1)", hardK(K, 20.0)))
for K in (25, 50, 100):
    CFG.append((f"hard K={K} x20 (k1 only)", hardK(K, 20.0, cols=(2,))))
    CFG.append((f"hard K={K} x3  (mid+k1)", hardK(K, 3.0)))
for g in (2.0, 5.0, 10.0, 20.0):
    CFG.append((f"soft gamma={g:.0f}", soft(g)))
print(f"{'config':28s} {'s7':>8s} {'s17':>8s} {'s23':>8s} {'mean':>8s}  {'delta':>8s}   detail(s7)")
base = None
for name, f in CFG:
    vals = []; det = ""
    for s in (7, 17, 23):
        v, d = ev(f, s)
        vals.append(v)
        if s == 7:
            det = d
    m = float(np.mean(vals))
    if base is None:
        base = m
    print(f"{name:28s} {vals[0]:8.4f} {vals[1]:8.4f} {vals[2]:8.4f} {m:8.4f}  {m-base:+8.4f}   {det}")
