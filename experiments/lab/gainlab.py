# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Gain-axis experiment bench.

a03 established the axis that matters: the allocator is invariant to a constant
added to all three predicted scores of an item, so only the *gains*
d1 = s_mid - s_light and d2 = s_k1 - s_mid carry decision information (moving
the level to the truth buys +0.009, moving the gains to the truth buys +0.078).

This module caches the expensive stage once (train OOF predictions + the
train-fit dev predictions) and then evaluates arbitrary *transforms* of the
composed (score, cost) rows under the honest protocol:
    safety ratios chosen by 3-seed bootstrap EV on the train OOF rows only,
    dev scored once with those ratios.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP  # noqa: E402
import protocol as P  # noqa: E402

CACHE = Path("reports/lab/gainlab_stage1.pkl")

GRIDS = {"fast": np.arange(0.90, 1.061, 0.01),
         "balanced": np.arange(0.80, 1.041, 0.01),
         "premium": np.arange(0.78, 1.041, 0.01)}


def stage1(lab, exp=None, force=False):
    """(cv arrays over Train, arr over Dev) for one expensive configuration."""
    key = repr(sorted((exp or DEPLOYED_EXP).items()))
    if CACHE.exists() and not force:
        blob = pickle.loads(CACHE.read_bytes())
        if blob.get("key") == key:
            return blob["cv"], blob["arr"]
    t0 = time.perf_counter()
    cv = P.cv_arrays(lab, exp)
    arr = lab.fit_predict(lab.train_idx, lab.dev_idx, exp)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps({"key": key, "cv": cv, "arr": arr}))
    print(f"[gainlab] stage1 built in {time.perf_counter()-t0:.0f}s", flush=True)
    return cv, arr


def identity(lab, arr, ps, pc, tier):
    return ps, pc


def evaluate(lab, cv, arr, cfg=None, transform=identity, seeds=(7, 17, 23), nboot=150,
             grids=None, label="", verbose=True, return_curves=False):
    cfg = dict(DEPLOYED_CFG, **(cfg or {}))
    grids = grids or GRIDS
    ts = lab.true_s[cv["idx"]]; tc = lab.true_c[cv["idx"]]
    m = len(cv["idx"])
    safety, detail = {}, {}
    for t in TIERS:
        ps, pc = lab.compose(cv, cfg, t)
        ps, pc = transform(lab, cv, ps, pc, t)
        curve = np.zeros(len(grids[t])); bust_c = np.zeros(len(grids[t]))
        for s in seeds:
            smp = np.asarray(lab.samples_for(m, s, nboot, 880))
            ev, bs, _raw = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MULTS[t], grids[t])
            curve += ev / len(seeds)
            bust_c += bs / len(seeds)
        gi = int(np.argmax(curve))
        safety[t] = float(grids[t][gi])
        detail[t] = dict(ev=float(curve[gi]), bust=float(bust_c[gi]),
                         curve=curve.tolist(), grid=[float(x) for x in grids[t]])
    cv_ev = sum(W[t] * detail[t]["ev"] for t in TIERS)

    idx = arr["idx"]; total = 0.0; tiers = {}
    for t in TIERS:
        ps, pc = lab.compose(arr, cfg, t)
        ps, pc = transform(lab, arr, ps, pc, t)
        pick = lab.allocate(ps, pc, MULTS[t], safety[t])
        r = np.arange(len(idx))
        ratio = lab.true_c[idx][r, pick].sum() / lab.true_c[idx][:, 0].sum()
        sc = lab.true_s[idx][r, pick].mean()
        passed = ratio <= MULTS[t] + 1e-15
        tiers[t] = dict(score=float(sc), ratio=float(ratio), passed=bool(passed),
                        counts=np.bincount(pick, minlength=3).tolist())
        total += W[t] * (sc if passed else 0.0)
    res = dict(label=label, dev=float(total), cv_ev=cv_ev, safety=safety, tiers=tiers,
               detail=detail if return_curves else {t: {k: v for k, v in detail[t].items()
                                                        if k not in ("curve", "grid")}
                                                   for t in TIERS})
    if verbose:
        s = " ".join(f"{t[:4]}={tiers[t]['score']:.4f}/r{tiers[t]['ratio']:.3f}"
                     f"{'' if tiers[t]['passed'] else '!BUST'}@{safety[t]:.2f}" for t in TIERS)
        print(f"{label:38s} cvEV={cv_ev:.6f}  dev={res['dev']:.6f}  {s}", flush=True)
    return res


# --------------------------------------------------------------- transforms
def gains_of(ps):
    return ps[:, 0].copy(), ps[:, 1] - ps[:, 0], ps[:, 2] - ps[:, 1]


def rebuild(s0, d1, d2):
    return np.clip(np.column_stack([s0, s0 + d1, s0 + d1 + d2]), 0.0, 1.0)


def pair_scale(a1=1.0, a2=1.0, per_tier=None):
    """Scale the two upgrade gains independently (only a2/a1 changes decisions)."""
    def f(lab, arr, ps, pc, tier):
        x1, x2 = (per_tier or {}).get(tier, (a1, a2))
        s0, d1, d2 = gains_of(ps)
        return rebuild(s0, x1 * d1, x2 * d2), pc
    return f


def gain_shrink(w1=0.0, w2=0.0, toward="family"):
    """Shrink each gain toward the family (or global) mean gain by weight w."""
    def f(lab, arr, ps, pc, tier):
        s0, d1, d2 = gains_of(ps)
        fam = lab.fam_arr[arr["idx"]]
        for d, w in ((d1, w1), (d2, w2)):
            if w == 0.0:
                continue
            if toward == "family":
                tgt = np.zeros_like(d)
                for fv in np.unique(fam):
                    msk = fam == fv
                    tgt[msk] = d[msk].mean()
            else:
                tgt = np.full_like(d, d.mean())
            d *= (1 - w)
            d += w * tgt
        return rebuild(s0, d1, d2), pc
    return f


def compose_transforms(*fs):
    def f(lab, arr, ps, pc, tier):
        for g in fs:
            ps, pc = g(lab, arr, ps, pc, tier)
        return ps, pc
    return f


# ---------------------------------------------------------------- robust safety
# The plain bootstrap resamples items inside Train and therefore only sees
# sampling noise.  Dev busts show that the binding risk is *calibration
# transfer*: a configuration whose predicted cost ratio is slightly optimistic on
# a fresh sample.  These scenarios add family-mix shift and cost-tail inflation,
# which is what E39 found premium to be vulnerable to.
def _family_block_samples(lab, idx, seed, nboot, size=880):
    fam = lab.fam_arr[idx]
    groups = {f: np.where(fam == f)[0] for f in np.unique(fam)}
    names = list(groups)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nboot):
        pick = rng.choice(len(names), size=len(names), replace=True)
        pool = np.concatenate([groups[names[p]] for p in pick])
        out.append(rng.choice(pool, size=size, replace=True))
    return np.asarray(out)


def robust_curve(lab, cv, cfg, tier, grid, transform=identity, seeds=(7, 17, 23), nboot=300,
                 scenarios=("iid", "block", "tail", "cost")):
    ps, pc = lab.compose(cv, cfg, tier)
    ps, pc = transform(lab, cv, ps, pc, tier)
    idx = cv["idx"]
    ts = lab.true_s[idx]; tc = lab.true_c[idx]
    m = len(idx)
    ev_by = {}
    bust_by = {}
    for sc in scenarios:
        ev_t = np.zeros(len(grid)); bu_t = np.zeros(len(grid))
        for s in seeds:
            if sc == "block":
                smp = _family_block_samples(lab, idx, s + 1000, nboot)
            else:
                smp = np.asarray(lab.samples_for(m, s, nboot, 880))
            TC = tc[smp].copy()
            if sc == "cost":                      # think outputs 25% longer
                TC[:, :, 2] *= 1.25
                TC[:, :, 1] *= 1.10
            if sc == "tail":                      # over-weight the expensive-k1 half
                q = np.quantile(tc[:, 2], 0.5)
                heavy = np.where(tc[:, 2] > q)[0]
                rng = np.random.default_rng(s + 77)
                rep = rng.integers(0, len(heavy), size=(nboot, 880 // 4))
                smp = smp.copy(); smp[:, :880 // 4] = heavy[rep]
                TC = tc[smp]
            ev, bu, _ = P.safety_curve(ps[smp], pc[smp], ts[smp], TC, MULTS[tier], grid)
            ev_t += ev / len(seeds); bu_t += bu / len(seeds)
        ev_by[sc] = ev_t; bust_by[sc] = bu_t
    return ev_by, bust_by


def robust_evaluate(lab, cv, arr, cfg=None, transform=identity, seeds=(7, 17, 23), nboot=300,
                    grids=None, label="", tau=0.02, verbose=True,
                    scenarios=("iid", "block", "tail", "cost")):
    """Safety = largest EV point whose worst-case scenario bust rate is <= tau."""
    cfg = dict(DEPLOYED_CFG, **(cfg or {}))
    grids = grids or GRIDS
    safety, detail = {}, {}
    for t in TIERS:
        g = grids[t]
        ev_by, bust_by = robust_curve(lab, cv, cfg, t, g, transform, seeds, nboot, scenarios)
        worst = np.max(np.stack([bust_by[s] for s in scenarios]), axis=0)
        ok = worst <= tau
        base = ev_by["iid"]
        cand = np.where(ok, base, -1.0)
        gi = int(np.argmax(cand)) if ok.any() else int(np.argmin(worst))
        safety[t] = float(g[gi])
        detail[t] = dict(ev=float(base[gi]), bust_iid=float(bust_by["iid"][gi]),
                         bust_worst=float(worst[gi]),
                         ev_mean=float(np.mean([ev_by[s][gi] for s in scenarios])))
    cv_ev = sum(W[t] * detail[t]["ev"] for t in TIERS)
    cv_ev_rob = sum(W[t] * detail[t]["ev_mean"] for t in TIERS)
    idx = arr["idx"]; total = 0.0; tiers = {}
    for t in TIERS:
        ps, pc = lab.compose(arr, cfg, t)
        ps, pc = transform(lab, arr, ps, pc, t)
        pick = lab.allocate(ps, pc, MULTS[t], safety[t])
        r = np.arange(len(idx))
        ratio = lab.true_c[idx][r, pick].sum() / lab.true_c[idx][:, 0].sum()
        sc = lab.true_s[idx][r, pick].mean()
        passed = ratio <= MULTS[t] + 1e-15
        tiers[t] = dict(score=float(sc), ratio=float(ratio), passed=bool(passed))
        total += W[t] * (sc if passed else 0.0)
    res = dict(label=label, dev=float(total), cv_ev=cv_ev, cv_ev_rob=cv_ev_rob,
               safety=safety, tiers=tiers, detail=detail)
    if verbose:
        s = " ".join(f"{t[:4]}={tiers[t]['score']:.4f}/r{tiers[t]['ratio']:.3f}"
                     f"{'' if tiers[t]['passed'] else '!BUST'}@{safety[t]:.2f}" for t in TIERS)
        print(f"{label:38s} cvEV={cv_ev:.6f} rob={cv_ev_rob:.6f} dev={res['dev']:.6f}  {s}", flush=True)
    return res
