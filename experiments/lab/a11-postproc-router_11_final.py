# SPDX-License-Identifier: Apache-2.0
"""a11 step 11 - how much of the measured risk rests on a handful of items, and
what do the recommended constants score (deterministic dev + bootstrap EV)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
import importlib.util
_spec = importlib.util.spec_from_file_location("a11path", HERE / "a11-postproc-router_5_path.py")
a11path = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(a11path)

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv)
DEP = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}


def path_for(tier, kappa):
    pc = P[f"cost_{tier}"] * np.asarray(kappa)[None, :]
    pa = a11path.path_arrays(P[f"score_{tier}"], pc, dv.score, dv.cost)
    pa["bp"] = pc[:, 0]
    return pa


def ev(tier, pa, s, samples, pool=None):
    mult = TIER_MULT[tier]
    tot = 0.0; bust = 0; cond = 0.0
    for S in samples:
        m = len(S)
        bp = pa["bp"][S].sum(); bt = dv.cost[S, 0].sum(); bs = dv.score[S, 0].sum()
        v = pa["valid"][S].ravel()
        o = np.argsort(-pa["eff"][S].ravel()[v], kind="stable")
        cp = np.concatenate([[0.0], np.cumsum(pa["dc_p"][S].ravel()[v][o])])
        ct = np.concatenate([[0.0], np.cumsum(pa["dc_t"][S].ravel()[v][o])])
        st = np.concatenate([[0.0], np.cumsum(pa["ds_t"][S].ravel()[v][o])])
        cap = bp * (max(1.0, mult * s) - 1.0)
        k = int(np.searchsorted(cp, cap, side="right")) - 1
        ratio = (bt + ct[k]) / bt
        sc = (bs + st[k]) / m
        cond += sc
        if ratio > mult:
            bust += 1
        else:
            tot += sc
    R = len(samples)
    return tot / R, bust / R, cond / R


def samples(seed, pool, size=880, R=400, replace=True):
    g = np.random.default_rng(seed)
    return [g.choice(pool, size=size, replace=replace) for _ in range(R)]


full = np.arange(n)
order = np.argsort(-dv.cost[:, 2])
print("=== (a) how much of the bust risk rests on the few costliest items? ===")
print(f"{'pool':28s} {'tier':9s} {'s':>5s} {'EV':>7s} {'bust':>6s} {'cond':>7s}")
for pname, pool, kw in (("full dev (880, with repl.)", full, {}),
                        ("drop top-1 k1 cost", np.delete(full, order[:1]), {}),
                        ("drop top-3", np.delete(full, order[:3]), {}),
                        ("drop top-10", np.delete(full, order[:10]), {}),
                        ("subsample 440 w/o repl.", full, dict(size=440, replace=False))):
    for tier in TIERS:
        pa = path_for(tier, (1, 1, 1))
        e = np.zeros(3); b = np.zeros(3); c = np.zeros(3)
        for i, sd in enumerate((7, 17, 23)):
            e[i], b[i], c[i] = ev(tier, pa, DEP[tier], samples(sd, pool, **kw))
        print(f"{pname:28s} {tier:9s} {DEP[tier]:5.2f} {e.mean():7.4f} {b.mean():6.3f} {c.mean():7.4f}")
    print()

print("=== (b) deterministic dev: deployed vs kappa-corrected, matched realised ratio ===")
KAPS = [(1.0, 1.0, 1.0), (1.0, 0.93, 1.24), (1.0, 1.0, 1.33)]
best = {}
for tier in TIERS:
    print(f"-- {tier}")
    for kap in KAPS:
        pc = P[f"cost_{tier}"] * np.asarray(kap)[None, :]
        for s in np.arange(0.60, 1.201, 0.005):
            r = tier_result(P[f"score_{tier}"], pc, dv, tier, float(s))
            key = (tier, kap)
            if r["passed"] and (key not in best or r["score"] > best[key][1]):
                best[key] = (float(s), r["score"], r["ratio"])
        s, sc, ra = best[(tier, kap)]
        print(f"   kappa={str(kap):18s} best passing s={s:.3f} score={sc:.4f} ratio={ra:.4f}")

print("\n=== (c) bootstrap EV of candidate deployments (3 seeds x 400) ===")
CAND = {
    "deployed .98/.87/.85, kappa=1": ((1, 1, 1), {"fast": .98, "balanced": .87, "premium": .85}),
    "kappa=1, EV-tuned .95/.82/.75": ((1, 1, 1), {"fast": .95, "balanced": .82, "premium": .75}),
    "kappa=(1,.93,1.24), .93/.83/.79": ((1, .93, 1.24), {"fast": .93, "balanced": .83, "premium": .79}),
    "kappa=(1,.93,1.24), dep-s .98/.87/.85": ((1, .93, 1.24), {"fast": .98, "balanced": .87, "premium": .85}),
    "kappa=(1,1,1.33), .94/.84/.80": ((1, 1, 1.33), {"fast": .94, "balanced": .84, "premium": .80}),
}
for name, (kap, sf) in CAND.items():
    tot = 0.0; parts = []
    for tier in TIERS:
        pa = path_for(tier, kap)
        e = np.zeros(3); b = np.zeros(3); c = np.zeros(3)
        for i, sd in enumerate((7, 17, 23)):
            e[i], b[i], c[i] = ev(tier, pa, sf[tier], samples(sd, full))
        tot += TIER_WEIGHT[tier] * e.mean()
        parts.append(f"{tier[:4]}={e.mean():.4f}/b{b.mean():.3f}")
    print(f"  {name:40s} weighted EV={tot:.4f}   " + " ".join(parts))

print("\n=== (d) deterministic dev final score of the candidates ===")
for name, (kap, sf) in CAND.items():
    tot = 0.0; parts = []
    for tier in TIERS:
        pc = P[f"cost_{tier}"] * np.asarray(kap)[None, :]
        r = tier_result(P[f"score_{tier}"], pc, dv, tier, sf[tier])
        tot += TIER_WEIGHT[tier] * r["tier_score"]
        parts.append(f"{tier[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else ' BUST'}")
    print(f"  {name:40s} dev final={tot:.4f}   " + " ".join(parts))
