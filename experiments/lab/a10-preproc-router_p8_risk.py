# SPDX-License-Identifier: Apache-2.0
"""a10 P8 - the safety scalar measured on HELD-OUT predictions.

E39/E43 chose .98/.87/.85 from a bust curve computed on 2,640 CV predictions.
E43 then observed a real held-out BUST at premium .88 (ratio 4.06) - i.e. the
CV curve put ~0 mass where reality landed.  This script recomputes the bust
curve on the honest held-out object: the train-only E43 predictions for the
880 dev episodes, bootstrapped to 880.

Also: does the train-derived per-model Duan factor (the only preprocessing
change that survived P5 across 3 seeds) move the curve?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, allocate
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev"); tr = load_split("train")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv); ar = np.arange(n)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
# train-derived per-model Duan factors (recomputed in P5; hard-coded here to
# keep this script cheap -- P5 prints them)
GFAC = np.array([1.168, 1.129, 1.448])

print("=" * 88)
print("(1) held-out point estimates: predicted vs realised budget ratio")
for t in TIERS:
    S, C = P[f"score_{t}"], P[f"cost_{t}"]
    sel = allocate(S, C, dv.cost, TIER_MULT[t], SAFE[t])
    rp = C[ar, sel].sum() / C[:, 0].sum()
    rt = dv.cost[ar, sel].sum() / dv.cost[:, 0].sum()
    print(f"  {t:9s} safety={SAFE[t]:.2f}  predicted ratio={rp:.4f}  realised={rt:.4f}"
          f"  cap={TIER_MULT[t]:.2f}  headroom={TIER_MULT[t]/rt:.4f}  K={rt/rp:.4f}")

print()
print("=" * 88)
print("(2) bust curve on held-out preds, 880-bootstrap, B=800, seeds 7 & 17 averaged")


def curve(mkC, tier, grid):
    S = P[f"score_{tier}"]
    C = mkC(tier)
    mult = TIER_MULT[tier]
    res = []
    for sf in grid:
        ev = []; nb = 0; nn = 0
        for seed in (7, 17):
            rng = np.random.default_rng(seed)
            for _ in range(400):
                rows = rng.integers(0, n, n)
                sel = allocate(S[rows], C[rows], dv.cost[rows], mult, float(sf))
                tc = dv.cost[rows]
                ok = tc[ar, sel].sum() / tc[:, 0].sum() <= mult + 1e-15
                ev.append(dv.score[rows][ar, sel].mean() if ok else 0.0)
                nb += 0 if ok else 1; nn += 1
        ev = np.array(ev)
        res.append((float(sf), nb / nn, float(ev.mean()),
                    float(ev[ev > 0].mean()) if (ev > 0).any() else 0.0))
    return res


raw = lambda t: P[f"cost_{t}"]
duan = lambda t: P[f"cost_{t}"] * GFAC[None, :]

for tier, grid in (("premium", np.round(np.arange(0.66, 0.941, 0.02), 3)),
                   ("fast", np.round(np.arange(0.88, 1.041, 0.02), 3)),
                   ("balanced", np.round(np.arange(0.74, 0.941, 0.02), 3))):
    print(f"  --- {tier}  (deployed safety {SAFE[tier]})")
    print(f"      {'safety':>7s} | {'raw cost head':^28s} | {'x per-model Duan (train)':^28s}")
    print(f"      {'':>7s} | {'bust':>6s} {'EV':>8s} {'E[s|pass]':>10s} | "
          f"{'bust':>6s} {'EV':>8s} {'E[s|pass]':>10s}")
    a = curve(raw, tier, grid); b = curve(duan, tier, grid)
    for (sf, p1, e1, s1), (_, p2, e2, s2) in zip(a, b):
        print(f"      {sf:7.2f} | {p1:6.3f} {e1:8.4f} {s1:10.4f} | "
              f"{p2:6.3f} {e2:8.4f} {s2:10.4f}")
    print()

print("=" * 88)
print("(3) weighted final EV of safety triples (raw cost head, B=800)")


def triple_ev(safety, mkC=raw, B=400):
    tot = 0.0; det = []
    for t in TIERS:
        S = P[f"score_{t}"]; C = mkC(t); mult = TIER_MULT[t]
        ev = []; nb = 0; nn = 0
        for seed in (7, 17):
            rng = np.random.default_rng(seed)
            for _ in range(B):
                rows = rng.integers(0, n, n)
                sel = allocate(S[rows], C[rows], dv.cost[rows], mult, safety[t])
                tc = dv.cost[rows]
                ok = tc[ar, sel].sum() / tc[:, 0].sum() <= mult + 1e-15
                ev.append(dv.score[rows][ar, sel].mean() if ok else 0.0)
                nb += 0 if ok else 1; nn += 1
        m = float(np.mean(ev)); tot += TIER_WEIGHT[t] * m
        det.append(f"{t[:4]}={m:.4f}(bust {nb/nn:.3f})")
    return tot, det


for name, sfy, mk in (
        ("deployed E43 .98/.87/.85", dict(fast=.98, balanced=.87, premium=.85), raw),
        ("E39 robust    .985/.875/.81", dict(fast=.985, balanced=.875, premium=.81), raw),
        ("a10 held-out  .95/.82/.75", dict(fast=.95, balanced=.82, premium=.75), raw),
        ("a10 held-out  .94/.80/.72", dict(fast=.94, balanced=.80, premium=.72), raw),
        ("a10 + Duan    .94/.84/.82", dict(fast=.94, balanced=.84, premium=.82), duan),
        ("a10 + Duan    .96/.86/.86", dict(fast=.96, balanced=.86, premium=.86), duan)):
    tot, det = triple_ev(sfy, mk)
    print(f"  {name:30s} EV={tot:.4f}   " + "  ".join(det))

print()
print("=" * 88)
print("(4) is the premium bust driven by a few items?  drop the k most expensive")
print("    TRUE premium costs from the pool and recompute the bust rate at .85")
S, C = P["score_premium"], P["cost_premium"]
sel0 = allocate(S, C, dv.cost, 4.0, SAFE["premium"])
order = np.argsort(-dv.cost[ar, sel0])
for k in (0, 3, 10, 25):
    keep = np.setdiff1d(ar, order[:k])
    m = len(keep)
    rng = np.random.default_rng(7); nb = 0
    for _ in range(400):
        rows = keep[rng.integers(0, m, m)]
        sel = allocate(S[rows], C[rows], dv.cost[rows], 4.0, SAFE["premium"])
        tc = dv.cost[rows]
        nb += 0 if tc[np.arange(m), sel].sum() / tc[:, 0].sum() <= 4.0 else 1
    print(f"    drop top {k:2d}: pool={m}  bust={nb/400:.3f}")
