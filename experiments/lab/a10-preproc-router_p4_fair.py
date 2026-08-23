# SPDX-License-Identifier: Apache-2.0
"""a10 P4 - fair comparison of allocator-input designs.

Every config is given its OWN EV-optimal safety triple (paired 880-bootstrap,
identical batches across configs), then compared at that optimum.  This is the
only fair way to compare: any change to the ledger rescales the safety axis.

NOTE ON THE HARNESS: bootstraps are drawn from the 880 held-out dev episodes
using the deployed E43 train-only predictions.  That is a *different and
harsher* basis than the project's 2,640-item CV EV harness (E39/E43), because
dev turned out to be a high-cost sample.  Absolute EV levels here are NOT
comparable to the log's numbers; differences between configs inside this
script are.
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
dv = load_split("dev")
tr = load_split("train")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
n = len(dv)
ar = np.arange(n)
famdv = np.array([classify_family(t) for t in dv.texts])
famtr = np.array([classify_family(t) for t in tr.texts])
fams = sorted(set(famtr) | set(famdv))
fid = np.array([fams.index(f) for f in famdv])
FI = {f: i for i, f in enumerate(fams)}


def allocate2(pred_score, util_cost, ledger_cost, mult, safety, mask=None):
    N = len(pred_score)
    a = np.arange(N)
    cap = ledger_cost[:, 0].sum() * max(1.0, mult * safety)
    U0 = pred_score if mask is None else np.where(mask, pred_score, -1e18)
    denom = util_cost[:, 0].sum()

    def choose(pen):
        return (U0 - pen * util_cost / denom).argmax(axis=1)

    sel = choose(0.0)
    tot = ledger_cost[a, sel].sum()
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


# shared bootstrap batches (paired across every config)
def make_batches(B, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(B)]


GRID = np.round(np.arange(0.60, 1.121, 0.02), 3)


def evaluate(cfg, batches, grid=GRID):
    """cfg(tier) -> (score, util_cost, ledger_cost, mask).  Returns per-tier
    (best_safety, EV, bust_rate, mean_pass_score) and the dev point estimate."""
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
                ratio = tc[ar, sel].sum() / tc[:, 0].sum()
                ok = ratio <= mult + 1e-15
                vals[bi] = dv.score[rows][ar, sel].mean() if ok else 0.0
                nb += 0 if ok else 1
            ev = float(vals.mean())
            if best is None or ev > best[1]:
                best = (float(sf), ev, nb / len(batches))
        # dev point estimate at that safety
        sel = allocate2(s, uc, lc, mult, best[0], mk)
        ratio = dv.cost[ar, sel].sum() / dv.cost[:, 0].sum()
        ok = ratio <= mult + 1e-15
        out[t] = dict(safety=best[0], ev=best[1], bust=best[2],
                      dev_score=float(dv.score[ar, sel].mean()),
                      dev_ratio=float(ratio), dev_ok=ok)
    out["ev"] = sum(TIER_WEIGHT[t] * out[t]["ev"] for t in TIERS)
    out["dev"] = sum(TIER_WEIGHT[t] * (out[t]["dev_score"] if out[t]["dev_ok"] else 0.0)
                     for t in TIERS)
    return out


# ------------------------------------------------------------------- configs
def base(t):
    return P[f"score_{t}"], P[f"cost_{t}"], P[f"cost_{t}"], None


half = ar % 2


def fam_factors(rows_fit, tier):
    pc = P[f"cost_{tier}"]
    g = dv.cost[rows_fit].sum(0) / pc[rows_fit].sum(0)
    fac = np.tile(g, (len(fams), 1))
    for k in range(len(fams)):
        m = rows_fit[fid[rows_fit] == k]
        if len(m) >= 8:
            fac[k] = dv.cost[m].sum(0) / pc[m].sum(0)
    return fac


def ledger_smear(t):
    pc = P[f"cost_{t}"]
    led = pc.copy()
    for h in (0, 1):
        fit = ar[half == 1 - h]; app = ar[half == h]
        led[app] = pc[app] * fam_factors(fit, t)[fid[app]]
    return P[f"score_{t}"], pc, led, None


def mk_nok1(families, tiers_on=TIERS):
    def cfg(t):
        mask = np.ones((n, 3), bool)
        if t in tiers_on:
            for f in families:
                mask[fid == FI[f], 2] = False
        return P[f"score_{t}"], P[f"cost_{t}"], P[f"cost_{t}"], mask
    return cfg


def mk_clamp(families):
    def cfg(t):
        mask = np.ones((n, 3), bool)
        for f in families:
            mask[fid == FI[f], 1:] = False
        return P[f"score_{t}"], P[f"cost_{t}"], P[f"cost_{t}"], mask
    return cfg


def mk_shrink(tau):
    """ledger cost shrunk toward the family geometric mean (aggregation test)."""
    def cfg(t):
        pc = P[f"cost_{t}"]
        lg = np.log(pc)
        fm = np.zeros((len(fams), 3))
        for k in range(len(fams)):
            fm[k] = lg[fid == k].mean(0)
        led = np.exp((1 - tau) * lg + tau * fm[fid])
        return P[f"score_{t}"], pc, led, None
    return cfg


def mk_fastpair(t):
    mask = np.ones((n, 3), bool)
    if t == "fast":
        mask[:, 2] = False
    return P[f"score_{t}"], P[f"cost_{t}"], P[f"cost_{t}"], mask


CONFIGS = [
    ("base (deployed inputs)", base),
    ("ledger = family-smeared (cross-fit)", ledger_smear),
    ("ledger shrunk to family gmean tau=.25", mk_shrink(0.25)),
    ("ledger shrunk to family gmean tau=.50", mk_shrink(0.50)),
    ("ledger shrunk to family gmean tau=1.0", mk_shrink(1.00)),
    ("no k1 for code", mk_nok1(["code"])),
    ("no k1 for code+aime", mk_nok1(["code", "aime"])),
    ("no k1 for code+dmmath", mk_nok1(["code", "dmmath"])),
    ("no k1 anywhere but bal/prem allow", mk_nok1(["code"], tiers_on=("premium",))),
    ("clamp longdoc+hrmcr to light", mk_clamp(["longdoc", "hrmcr"])),
    ("fast: no k1 (pairwise tier)", mk_fastpair),
]

for SEED, B in ((7, 300),):
    batches = make_batches(B, SEED)
    print("=" * 96)
    print(f"paired bootstrap, seed={SEED}, B={B}, EV-optimal safety per config per tier")
    print(f"{'config':40s} {'EV':>7s} {'dEV':>7s} {'dev':>7s}  "
          f"{'safety f/b/p':>16s}  {'bust f/b/p':>16s}")
    ref = None
    for name, cfg in CONFIGS:
        r = evaluate(cfg, batches)
        if ref is None:
            ref = r["ev"]
        print(f"{name:40s} {r['ev']:7.4f} {r['ev']-ref:+7.4f} {r['dev']:7.4f}  "
              + "/".join(f"{r[t]['safety']:.2f}" for t in TIERS)
              + "  " + "/".join(f"{r[t]['bust']:.3f}" for t in TIERS)
              + "   tiers " + " ".join(f"{r[t]['dev_score']:.4f}" for t in TIERS))
