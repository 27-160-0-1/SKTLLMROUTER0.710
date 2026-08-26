# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a10 P9 - cost-tail guard + confirmation of the recommended input set.

P8(4): dropping the 3 most expensive realised premium selections cuts the
880-bootstrap bust rate from 0.140 to 0.022.  So the premium budget risk is
carried by a handful of very expensive k1 selections.  E08 rejected a per-item
cost CAP, but that was measured on the CV harness where the bust probability
was already ~0.  Re-test on held-out preds: forbid k1 for items whose
PREDICTED k1 cost exceeds tau x (mean light cost).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev"); tr = load_split("train")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv); ar = np.arange(n)
GFAC = np.array([1.168, 1.129, 1.448])          # train-derived per-model Duan
fam = np.array([classify_family(t) for t in dv.texts])


def allocate2(pred_score, util_cost, ledger_cost, mult, safety, mask=None):
    N = len(pred_score); a = np.arange(N)
    cap = ledger_cost[:, 0].sum() * max(1.0, mult * safety)
    U0 = pred_score if mask is None else np.where(mask, pred_score, -1e18)
    denom = util_cost[:, 0].sum()

    def choose(pen):
        return (U0 - pen * util_cost / denom).argmax(axis=1)
    sel = choose(0.0); tot = ledger_cost[a, sel].sum()
    if tot > cap:
        low, high = 0.0, 1.0
        sel = choose(high); tot = ledger_cost[a, sel].sum()
        while tot > cap and high < 2 ** 60:
            low, high = high, high * 2.0
            sel = choose(high); tot = ledger_cost[a, sel].sum()
        for _ in range(40):
            mid = (low + high) / 2.0
            cand = choose(mid); ct = ledger_cost[a, cand].sum()
            if ct <= cap:
                high, sel, tot = mid, cand, ct
            else:
                low = mid
    if tot > cap:
        sel = np.zeros(N, dtype=int)
    return sel


GRID = np.round(np.arange(0.60, 1.101, 0.02), 3)


def evaluate(cfg, batches, grid=GRID):
    out = {}
    for t in TIERS:
        s, uc, lc, mk = cfg(t)
        mult = TIER_MULT[t]
        best = None
        for sf in grid:
            vals = np.empty(len(batches)); nb = 0
            for bi, rows in enumerate(batches):
                sel = allocate2(s[rows], uc[rows], lc[rows], mult, float(sf),
                                None if mk is None else mk[rows])
                tc = dv.cost[rows]
                ok = tc[ar, sel].sum() / tc[:, 0].sum() <= mult + 1e-15
                vals[bi] = dv.score[rows][ar, sel].mean() if ok else 0.0
                nb += 0 if ok else 1
            ev = float(vals.mean())
            if best is None or ev > best[1]:
                best = (float(sf), ev, nb / len(batches))
        sel = allocate2(s, uc, lc, mult, best[0], mk)
        ratio = dv.cost[ar, sel].sum() / dv.cost[:, 0].sum()
        out[t] = dict(safety=best[0], ev=best[1], bust=best[2],
                      dev_score=float(dv.score[ar, sel].mean()), dev_ratio=float(ratio),
                      dev_ok=ratio <= mult + 1e-15)
    out["ev"] = sum(TIER_WEIGHT[t] * out[t]["ev"] for t in TIERS)
    out["dev"] = sum(TIER_WEIGHT[t] * (out[t]["dev_score"] if out[t]["dev_ok"] else 0.0)
                     for t in TIERS)
    return out


def base(t):
    return P[f"score_{t}"], P[f"cost_{t}"], P[f"cost_{t}"], None


def duan(t):
    C = P[f"cost_{t}"] * GFAC[None, :]
    return P[f"score_{t}"], C, C, None


def mk_guard(tau, with_duan=False):
    def cfg(t):
        C = P[f"cost_{t}"] * (GFAC[None, :] if with_duan else 1.0)
        thr = tau * C[:, 0].mean()
        mask = np.ones((n, 3), bool)
        mask[:, 2] = C[:, 2] <= thr
        return P[f"score_{t}"], C, C, mask
    return cfg


CONFIGS = [("base", base),
           ("per-model Duan (train) both", duan),
           ("guard: k1 forbidden if pred c_k1 > 30x meanlight", mk_guard(30)),
           ("guard tau=60", mk_guard(60)),
           ("guard tau=120", mk_guard(120)),
           ("Duan + guard tau=60", mk_guard(60, True)),
           ("Duan + guard tau=120", mk_guard(120, True))]

print("share of dev items with predicted premium k1 cost above tau x mean light:")
Cp = P["cost_premium"]
for tau in (30, 60, 120):
    m = Cp[:, 2] > tau * Cp[:, 0].mean()
    print(f"   tau={tau:4d}: {int(m.sum()):4d} items  families="
          + str({k: int(v) for k, v in zip(*np.unique(fam[m], return_counts=True))}))

for SEED in (7, 17, 23):
    rng = np.random.default_rng(SEED)
    batches = [rng.integers(0, n, n) for _ in range(250)]
    print()
    print("=" * 96)
    print(f"paired bootstrap seed={SEED} B=250, EV-optimal safety per config")
    print(f"{'config':48s} {'EV':>7s} {'dEV':>7s} {'dev':>7s}  {'safety f/b/p':>14s}"
          f"  {'bust f/b/p':>14s}")
    ref = None
    for name, cfg in CONFIGS:
        r = evaluate(cfg, batches)
        if ref is None:
            ref = r["ev"]
        print(f"{name:48s} {r['ev']:7.4f} {r['ev']-ref:+7.4f} {r['dev']:7.4f}  "
              + "/".join(f"{r[t]['safety']:.2f}" for t in TIERS) + "  "
              + "/".join(f"{r[t]['bust']:.3f}" for t in TIERS)
              + "  tiers " + " ".join(f"{r[t]['dev_score']:.4f}" for t in TIERS))
