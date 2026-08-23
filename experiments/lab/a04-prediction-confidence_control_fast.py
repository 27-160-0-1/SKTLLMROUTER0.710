# SPDX-License-Identifier: Apache-2.0
"""a04 / step 13-fast -- cheap version of the E32/E39 control (150 draws, 1 seed,
coarse safety grid) so a verdict exists even if the full run is still going.
Same question: does the targeted guard beat what the global safety scalar can
already do, and does it beat an information-free uniform inflation?
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
        pick = (ps - pen * pc / lt).argmax(1)
        return pick, pc[np.arange(len(pick)), pick].sum()
    pick, tot = choose(0.0)
    if tot > cap:
        lo, hi = 0.0, 1.0
        pick, tot = choose(hi)
        while tot > cap and hi < 2 ** 60:
            lo, hi = hi, hi * 2
            pick, tot = choose(hi)
        for _ in range(30):
            mid = (lo + hi) / 2
            c2, t2 = choose(mid)
            if t2 <= cap:
                hi, pick, tot = mid, c2, t2
            else:
                lo = mid
    if tot > cap:
        pick = np.zeros(len(ps), dtype=int)
    return pick


r0 = np.random.default_rng(7)
SAMPLES = [r0.integers(0, n, size=880) for _ in range(150)]
GRID = {"fast": np.arange(0.84, 1.041, 0.04), "balanced": np.arange(0.74, 1.001, 0.04),
        "premium": np.arange(0.62, 1.001, 0.04)}


def mkcost(kind, t):
    pc = P[f"cost_{t}"].copy()
    if kind[0] == "hard":
        pc[order[:kind[1]], 1] *= kind[2]; pc[order[:kind[1]], 2] *= kind[2]
    elif kind[0] == "soft":
        m = 1.0 + kind[1] * prob
        pc[:, 1] *= m; pc[:, 2] *= m
    elif kind[0] == "flat":
        pc[:, 1] *= kind[1]; pc[:, 2] *= kind[1]
    elif kind[0] == "rand":            # information-free control: inflate a RANDOM K items
        rr = np.random.default_rng(kind[3]).permutation(n)[:kind[1]]
        pc[rr, 1] *= kind[2]; pc[rr, 2] *= kind[2]
    pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * 1.0000001)
    pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * 1.0000001)
    return pc


CFG = [("baseline", ("none",)),
       ("hard K=150 x20", ("hard", 150, 20.0)),
       ("soft gamma=20", ("soft", 20.0)),
       ("flat x1.3 (no info)", ("flat", 1.3)),
       ("flat x1.8 (no info)", ("flat", 1.8)),
       ("flat x2.5 (no info)", ("flat", 2.5)),
       ("rand K=150 x20 s1", ("rand", 150, 20.0, 1)),
       ("rand K=150 x20 s2", ("rand", 150, 20.0, 2))]

print("=" * 104)
print("STEP 13-fast  safety-optimised bootstrap EV (150 draws, seed 7, coarse grid)")
print("=" * 104)
print(f"{'config':22s} | " + " | ".join(f"{t:>24s}" for t in TIERS) + " | weighted", flush=True)
for name, kind in CFG:
    tot = 0.0; cells = []
    for t in TIERS:
        pc = mkcost(kind, t); ps = P[f"score_{t}"]
        best = None
        for sf in GRID[t]:
            evs = []; busts = 0
            for s in SAMPLES:
                pick = allocate(ps[s], pc[s], TIER_MULT[t], float(sf))
                r = np.arange(len(s))
                if dv.cost[s][r, pick].sum() / dv.cost[s][:, 0].sum() > TIER_MULT[t]:
                    busts += 1; evs.append(0.0)
                else:
                    evs.append(dv.score[s][r, pick].mean())
            m = float(np.mean(evs))
            if best is None or m > best[0]:
                best = (m, float(sf), busts / len(SAMPLES))
        tot += TIER_WEIGHT[t] * best[0]
        cells.append(f"EV{best[0]:.4f}@{best[1]:.2f} b{100*best[2]:4.1f}%")
    print(f"{name:22s} | " + " | ".join(f"{c:>24s}" for c in cells) + f" | {tot:.4f}", flush=True)
