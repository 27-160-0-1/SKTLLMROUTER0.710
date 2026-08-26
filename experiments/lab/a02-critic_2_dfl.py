# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a02-critic #2.  Decision-focused re-parameterisation of the EXISTING predictions.

Question: E37 proved the allocator is optimal *given* the predictions.  But the
predictions were trained with squared-error-ish losses on (score, log cost).
How much final score is left on the table purely by feeding the allocator
badly-shaped numbers -- with zero new information?

Protocol (honest): random half-split of dev (440/440), stratified by family.
All free parameters (including the per-tier safety ratios) are fitted on half A
by maximising a 440-item bootstrap EV objective, then applied unchanged to half
B.  Both directions, averaged over SPLITS random splits.  A configuration that
only re-shapes the same 6 numbers per item cannot smuggle in new information,
so the A->B delta is an honest lower bound on decision-focused headroom.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, TIER_MULT
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
fam = np.array([classify_family(t) for t in dv.texts])
N = len(dv)

SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NBOOT = 60


# ------------------------------------------------------------------ allocator
def alloc_batch(S, C, cap):
    """S,C: (B,m,3) predictions; cap: (B,). returns sel (B,m) int."""
    B, m, _ = S.shape
    L = C[:, :, 0].sum(1)                      # (B,)

    def choose(pen):
        util = S - pen[:, None, None] * C / L[:, None, None]
        return util.argmax(2)

    ar = np.arange(B)[:, None]
    pen = np.zeros(B)
    sel = choose(pen)
    tot = np.take_along_axis(C, sel[:, :, None], 2)[:, :, 0].sum(1)
    over = tot > cap
    if not over.any():
        return sel
    low = np.zeros(B)
    high = np.ones(B)
    for _ in range(24):
        s2 = choose(high)
        t2 = np.take_along_axis(C, s2[:, :, None], 2)[:, :, 0].sum(1)
        bad = (t2 > cap) & over
        if not bad.any():
            break
        low = np.where(bad, high, low)
        high = np.where(bad, high * 2.0, high)
    for _ in range(30):
        mid = (low + high) / 2.0
        cand = choose(mid)
        ct = np.take_along_axis(C, cand[:, :, None], 2)[:, :, 0].sum(1)
        ok = ct <= cap
        high = np.where(ok, mid, high)
        low = np.where(ok, low, mid)
    sel_final = choose(high)
    ct = np.take_along_axis(C, sel_final[:, :, None], 2)[:, :, 0].sum(1)
    sel = np.where(over[:, None], sel_final, sel)
    tot = np.where(over, ct, tot)
    # give-up rule of the deployed allocator: all-light
    give = tot > cap
    if give.any():
        sel = np.where(give[:, None], 0, sel)
    return sel


def evaluate(S, C, idx, tier, safety, true_s, true_c):
    """idx: (B,m) resample indices into the item pool.  returns (score, passed)."""
    Sb, Cb = S[idx], C[idx]
    ts, tc = true_s[idx], true_c[idx]
    cap = Cb[:, :, 0].sum(1) * max(1.0, TIER_MULT[tier] * safety)
    sel = alloc_batch(Sb, Cb, cap)
    got_s = np.take_along_axis(ts, sel[:, :, None], 2)[:, :, 0].mean(1)
    got_c = np.take_along_axis(tc, sel[:, :, None], 2)[:, :, 0].sum(1)
    ratio = got_c / tc[:, :, 0].sum(1)
    passed = ratio <= TIER_MULT[tier] + 1e-15
    return got_s, passed, ratio


# ------------------------------------------------------------- transformations
def transform(S, C, th):
    """th = dict of decision-focused shape parameters.  NO new information."""
    S2 = S.copy()
    C2 = C.copy()
    # per-model log-cost offsets (relative pricing)
    C2 = C2 * np.exp(np.array([0.0, th.get("b1", 0.0), th.get("b2", 0.0)]))[None, :]
    # gain sharpening: s_j -> s_0 + g*(s_j - s_0)
    g = th.get("gain", 1.0)
    S2 = S2[:, [0]] + g * (S2 - S2[:, [0]])
    # extra additive premium on the top model's gain
    S2[:, 2] = S2[:, 2] + th.get("a2", 0.0)
    S2[:, 1] = S2[:, 1] + th.get("a1", 0.0)
    return S2, C2


BASE_TH = dict(b1=0.0, b2=0.0, gain=1.0, a1=0.0, a2=0.0)
GRIDS = {
    "b1": np.arange(-0.30, 0.301, 0.05),
    "b2": np.arange(-0.40, 0.401, 0.05),
    "gain": np.arange(0.5, 2.01, 0.1),
    "a1": np.arange(-0.04, 0.041, 0.01),
    "a2": np.arange(-0.04, 0.041, 0.01),
}
SAFE_GRID = {"fast": np.arange(0.86, 1.21, 0.01),
             "balanced": np.arange(0.76, 1.11, 0.01),
             "premium": np.arange(0.74, 1.09, 0.01)}


def fit_on(pool, params, rng, tier_list=TIERS, rounds=2):
    """coordinate descent over `params` + safety, maximising bootstrap EV on `pool`."""
    m = len(pool)
    idx = rng.integers(0, m, size=(NBOOT, m))
    ts, tc = dv.score[pool], dv.cost[pool]
    th = dict(BASE_TH)
    safety = dict(SAFE)

    def ev(tier, th_, sf):
        S, C = transform(P[f"score_{tier}"][pool], P[f"cost_{tier}"][pool], th_)
        s, p, _ = evaluate(S, C, idx, tier, sf, ts, tc)
        return float(np.mean(s * p))

    # start from a safety that is sane for the untransformed predictions
    for t in tier_list:
        safety[t] = max(SAFE_GRID[t], key=lambda sf: ev(t, th, float(sf)))
    for _ in range(rounds):
        for k in params:
            best, bv = th[k], -1
            for v in GRIDS[k]:
                th2 = dict(th); th2[k] = float(v)
                tot = sum(TIER_WEIGHT[t] * ev(t, th2, safety[t]) for t in tier_list)
                if tot > bv:
                    bv, best = tot, float(v)
            th[k] = best
        for t in tier_list:
            best, bv = safety[t], -1
            for sf in SAFE_GRID[t]:
                v = ev(t, th, float(sf))
                if v > bv:
                    bv, best = v, float(sf)
            safety[t] = best
    return th, safety


def score_on(pool, th, safety):
    """deterministic final score of one 440-item batch."""
    ts, tc = dv.score[pool], dv.cost[pool]
    idx = np.arange(len(pool))[None, :]
    tot = 0.0
    parts = {}
    for t in TIERS:
        S, C = transform(P[f"score_{t}"][pool], P[f"cost_{t}"][pool], th)
        s, p, r = evaluate(S, C, idx, t, safety[t], ts, tc)
        tot += TIER_WEIGHT[t] * float(s[0] * p[0])
        parts[t] = (float(s[0]), bool(p[0]), float(r[0]))
    return tot, parts


def stratified_split(rng):
    a, b = [], []
    for f in set(fam):
        ii = np.where(fam == f)[0]
        rng.shuffle(ii)
        h = len(ii) // 2
        a.extend(ii[:h]); b.extend(ii[h:])
    return np.array(sorted(a)), np.array(sorted(b))


CONFIGS = {
    "deployed constants (no fitting)":        ([], False),
    "safety only refit":                      ([], True),
    "+ relative cost pricing (b1,b2)":        (["b1", "b2"], True),
    "+ gain sharpening (g)":                  (["gain"], True),
    "+ model offsets (a1,a2)":                (["a1", "a2"], True),
    "FULL decision-focused (b,g,a) ":         (["b1", "b2", "gain", "a1", "a2"], True),
}

print(f"honest half-split cross-fit, {SPLITS} splits x 2 directions, "
      f"bootstrap-EV objective ({NBOOT} resamples) on the fitting half")
print(f"{'configuration':40s} {'held-out half':>13s} {'fit half':>10s}  delta_vs_deployed")
res = {}
t0 = time.time()
for name, (params, refit_safety) in CONFIGS.items():
    outs, ins = [], []
    for sp in range(SPLITS):
        rng = np.random.default_rng(1000 + sp)
        A, B = stratified_split(rng)
        for fitpool, evalpool in ((A, B), (B, A)):
            if not params and not refit_safety:
                th, sf = dict(BASE_TH), dict(SAFE)
            else:
                th, sf = fit_on(fitpool, params, np.random.default_rng(7 + sp))
            o, _ = score_on(evalpool, th, sf)
            i, _ = score_on(fitpool, th, sf)
            outs.append(o); ins.append(i)
    res[name] = np.array(outs)
    d = np.mean(outs) - np.mean(res["deployed constants (no fitting)"])
    print(f"{name:40s} {np.mean(outs):13.4f} {np.mean(ins):10.4f}  {d:+.4f}"
          f"   (sd of held-out mean {np.std(outs)/np.sqrt(len(outs)):.4f})")
print(f"[{time.time()-t0:.0f}s]")

print("\npaired deltas vs 'safety only refit' (same splits, same fitting seeds):")
base = res["safety only refit"]
for name, v in res.items():
    d = v - base
    print(f"  {name:40s} paired mean {d.mean():+.4f}  sd {d.std(ddof=1)/np.sqrt(len(d)):.4f}"
          f"  wins {int((d>0).sum())}/{len(d)}")
