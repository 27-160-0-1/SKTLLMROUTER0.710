# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a10 P7 - family-level preprocessing of the allocator's inputs.

C1  family re-centring of the SCORE matrix to the train family means
    (keeps the model's within-family ranking, replaces its family-level
    calibration with the train-measured one)
C2  family re-centring of the LOG-COST matrix so that each family x model
    predicted SUM matches the train family x model arithmetic mean cost
C3  C1 + C2
C4  per-item ratio as the utility cost (the one non-no-op reparameterisation)
C5  two-level allocation: family budget shares fixed from TRAIN (using the
    TRUE train labels, i.e. an optimistic upper bound for the idea), then a
    within-family Lagrangian on the batch
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_MULT, TIER_WEIGHT
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev"); tr = load_split("train")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv); ar = np.arange(n)
famdv = np.array([classify_family(t) for t in dv.texts])
famtr = np.array([classify_family(t) for t in tr.texts])
fams = sorted(set(famtr) | set(famdv))
FI = {f: i for i, f in enumerate(fams)}
fid = np.array([FI[f] for f in famdv]); fidtr = np.array([FI[f] for f in famtr])
NF = len(fams)

TR_SC = np.stack([tr.score[fidtr == k].mean(0) for k in range(NF)])      # (F,3)
TR_C = np.stack([tr.cost[fidtr == k].mean(0) for k in range(NF)])        # (F,3)


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


# ---- two-level: family budget shares from TRAIN with true labels (optimistic)
def train_family_shares(mult, safety):
    """flat Lagrangian on TRAIN with true labels -> share of the light total
    each family ends up consuming."""
    sel = allocate2(tr.score, tr.cost, tr.cost, mult, safety)
    a = np.arange(len(tr))
    Lt = tr.cost[:, 0].sum()
    sh = np.array([tr.cost[a, sel][fidtr == k].sum() / Lt for k in range(NF)])
    lsh = np.array([tr.cost[fidtr == k, 0].sum() / Lt for k in range(NF)])
    return sh, lsh          # spend share, light share


def allocate_twolevel(pred_score, cost, mult, safety, shares, lshares):
    """within-family Lagrangian; family f gets shares[f]/lshares[f] * its own
    light total (so the family multiplier is transferred from train)."""
    N = len(pred_score)
    sel = np.zeros(N, dtype=int)
    for k in range(NF):
        m = np.where(fid_batch == k)[0]
        if len(m) == 0:
            continue
        fm = shares[k] / max(lshares[k], 1e-12)          # family-level multiplier
        sel[m] = allocate2(pred_score[m], cost[m], cost[m], fm, safety)
    return sel


GRID = np.round(np.arange(0.62, 1.161, 0.02), 3)


def evaluate(cfg, batches, grid=GRID, twolevel=None):
    global fid_batch
    out = {}
    for t in TIERS:
        s, uc, lc, mk = cfg(t)
        mult = TIER_MULT[t]
        sh = lsh = None
        if twolevel is not None:
            sh, lsh = twolevel[t]
        best = None
        for sf in grid:
            vals = np.empty(len(batches)); nb = 0
            for bi, rows in enumerate(batches):
                fid_batch = fid[rows]
                if twolevel is None:
                    sel = allocate2(s[rows], uc[rows], lc[rows], mult, float(sf),
                                    None if mk is None else mk[rows])
                else:
                    sel = allocate_twolevel(s[rows], uc[rows], mult, float(sf), sh, lsh)
                tc = dv.cost[rows]
                ok = tc[ar, sel].sum() / tc[:, 0].sum() <= mult + 1e-15
                vals[bi] = dv.score[rows][ar, sel].mean() if ok else 0.0
                nb += 0 if ok else 1
            ev = float(vals.mean())
            if best is None or ev > best[1]:
                best = (float(sf), ev, nb / len(batches))
        fid_batch = fid
        if twolevel is None:
            sel = allocate2(s, uc, lc, mult, best[0], mk)
        else:
            sel = allocate_twolevel(s, uc, mult, best[0], sh, lsh)
        ratio = dv.cost[ar, sel].sum() / dv.cost[:, 0].sum()
        ok = ratio <= mult + 1e-15
        out[t] = dict(safety=best[0], ev=best[1], bust=best[2],
                      dev_score=float(dv.score[ar, sel].mean()),
                      dev_ratio=float(ratio), dev_ok=ok)
    out["ev"] = sum(TIER_WEIGHT[t] * out[t]["ev"] for t in TIERS)
    out["dev"] = sum(TIER_WEIGHT[t] * (out[t]["dev_score"] if out[t]["dev_ok"] else 0.0)
                     for t in TIERS)
    return out


def base(t):
    return P[f"score_{t}"], P[f"cost_{t}"], P[f"cost_{t}"], None


def c1(t):
    S = P[f"score_{t}"].copy()
    for k in range(NF):
        m = fid == k
        S[m] += TR_SC[k] - S[m].mean(0)
    return S, P[f"cost_{t}"], P[f"cost_{t}"], None


def _c2cost(t):
    C = P[f"cost_{t}"].copy()
    for k in range(NF):
        m = fid == k
        C[m] *= TR_C[k] / C[m].mean(0)
    return C


def c2(t):
    C = _c2cost(t)
    return P[f"score_{t}"], C, C, None


def c3(t):
    S = c1(t)[0]; C = _c2cost(t)
    return S, C, C, None


def c4(t):
    C = P[f"cost_{t}"]
    return P[f"score_{t}"], C / C[:, :1], C, None


CONFIGS = [("base", base, None),
           ("C1 family re-centred SCORES (train)", c1, None),
           ("C2 family re-centred COSTS (train)", c2, None),
           ("C3 both", c3, None),
           ("C4 ratio utility cost", c4, None)]

for SEED in (7, 17):
    rng = np.random.default_rng(SEED)
    batches = [rng.integers(0, n, n) for _ in range(200)]
    print("=" * 92)
    print(f"paired bootstrap seed={SEED} B=200, EV-optimal safety per config")
    print(f"{'config':38s} {'EV':>7s} {'dEV':>7s} {'dev':>7s}  {'safety f/b/p':>14s}"
          f"  {'bust f/b/p':>14s}")
    ref = None
    for name, cfg, tl in CONFIGS:
        r = evaluate(cfg, batches, twolevel=tl)
        if ref is None:
            ref = r["ev"]
        print(f"{name:38s} {r['ev']:7.4f} {r['ev']-ref:+7.4f} {r['dev']:7.4f}  "
              + "/".join(f"{r[t]['safety']:.2f}" for t in TIERS) + "  "
              + "/".join(f"{r[t]['bust']:.3f}" for t in TIERS)
              + "  tiers " + " ".join(f"{r[t]['dev_score']:.4f}" for t in TIERS))
    # two-level (needs its own share table per tier, at the tier's own safety)
    tl = {t: train_family_shares(TIER_MULT[t], 1.0) for t in TIERS}
    r = evaluate(base, batches, twolevel=tl)
    print(f"{'C5 two-level (train ORACLE shares)':38s} {r['ev']:7.4f} {r['ev']-ref:+7.4f} "
          f"{r['dev']:7.4f}  " + "/".join(f"{r[t]['safety']:.2f}" for t in TIERS) + "  "
          + "/".join(f"{r[t]['bust']:.3f}" for t in TIERS)
          + "  tiers " + " ".join(f"{r[t]['dev_score']:.4f}" for t in TIERS))
    print()
