# SPDX-License-Identifier: Apache-2.0
"""The honest end-to-end evaluation protocol used by every lab experiment.

    1. 5-fold CV inside Train (1,760) -> out-of-fold predictions
    2. bootstrap EV over 880-item resamples of those OOF rows -> per-tier safety ratio
       (this is the only place a safety ratio may be chosen: dev is never consulted)
    3. refit on all of Train -> predict Dev (880) -> exact final score

Nothing here reads dev outcomes before step 3, so the reported dev number is a
true held-out number for any cfg/exp that was itself chosen from step 2.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, W, MULTS, DEPLOYED_CFG, DEPLOYED_EXP  # noqa: E402


def cv_arrays(lab, exp=None, targets=None, sample_weight=None, folds=5, seed=123, n_jobs=5,
              idx=None):
    """Out-of-fold predictions over `idx` (default: Train)."""
    idx = lab.train_idx if idx is None else np.asarray(idx)
    fold_of = np.random.default_rng(seed).integers(0, folds, size=len(idx))
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(lab.fit_predict)(idx[fold_of != f], idx[fold_of == f], exp, targets, sample_weight)
        for f in range(folds)
    )
    order = np.concatenate([p["idx"] for p in parts])
    out = {"idx": order}
    for k in ("lin", "legacy", "fam", "knn", "meta", "gain", "rank_eff"):
        out[k] = np.vstack([p[k] for p in parts])
    out["floors"] = parts[0]["floors"]
    return out


def batch_allocate(ps, pc, mult, safety):
    """Vectorised Lagrangian allocation over a batch of bootstrap samples.

    ps, pc: (B, m, 3).  Returns picks (B, m) and totals (B,).
    Identical decisions to harness.Lab.allocate, run for every sample at once.
    """
    B = ps.shape[0]
    lt = pc[:, :, 0].sum(axis=1)                      # (B,)
    cap = lt * np.maximum(1.0, mult * safety)
    tb = np.array([2e-12, 1e-12, 0.0])

    def choose(pen):                                   # pen: (B,)
        u = ps - pen[:, None, None] * pc / lt[:, None, None] + tb
        pick = np.argmax(u, axis=2)                    # (B, m)
        tot = np.take_along_axis(pc, pick[:, :, None], axis=2)[:, :, 0].sum(axis=1)
        return pick, tot

    zero = np.zeros(B)
    pick, tot = choose(zero)
    need = tot > cap
    if need.any():
        lo = np.zeros(B); hi = np.ones(B)
        p2, t2 = choose(hi)
        pick = np.where(need[:, None], p2, pick); tot = np.where(need, t2, tot)
        for _ in range(64):
            grow = need & (tot > cap) & (hi < 2.0 ** 60)
            if not grow.any():
                break
            lo = np.where(grow, hi, lo)
            hi = np.where(grow, hi * 2.0, hi)
            p2, t2 = choose(hi)
            pick = np.where(grow[:, None], p2, pick); tot = np.where(grow, t2, tot)
        for _ in range(40):
            mid = (lo + hi) / 2.0
            p2, t2 = choose(mid)
            ok = need & (t2 <= cap)
            hi = np.where(ok, mid, hi)
            lo = np.where(need & ~ok, mid, lo)
            pick = np.where(ok[:, None], p2, pick); tot = np.where(ok, t2, tot)
    bad = tot > cap
    if bad.any():
        pick = np.where(bad[:, None], 0, pick)
    return pick


def choose_safety(lab, cv, cfg, seeds=(7, 17, 23), nboot=200, grids=None):
    """Per-tier safety ratio = argmax of the multi-seed mean bootstrap EV."""
    safety, detail = {}, {}
    grids = grids or {"fast": np.arange(0.90, 1.021, 0.01),
                      "balanced": np.arange(0.80, 0.981, 0.01),
                      "premium": np.arange(0.78, 0.961, 0.01)}
    curves = {t: np.zeros(len(grids[t])) for t in TIERS}
    busts = {t: np.zeros(len(grids[t])) for t in TIERS}
    ts = lab.true_s[cv["idx"]]; tc = lab.true_c[cv["idx"]]
    m = len(cv["idx"])
    for t in TIERS:
        ps, pc = lab.compose(cv, cfg, t)
        for s in seeds:
            smp = np.asarray(lab.samples_for(m, s, nboot, 880))       # (B, 880)
            PS = ps[smp]; PC = pc[smp]
            TS = ts[smp]; TC = tc[smp]
            light = TC[:, :, 0].sum(axis=1)
            for gi, g in enumerate(grids[t]):
                pick = batch_allocate(PS, PC, MULTS[t], float(g))
                real = np.take_along_axis(TC, pick[:, :, None], axis=2)[:, :, 0].sum(axis=1)
                sc = np.take_along_axis(TS, pick[:, :, None], axis=2)[:, :, 0].mean(axis=1)
                bust = (real / light) > MULTS[t]
                curves[t][gi] += float(np.mean(np.where(bust, 0.0, sc))) / len(seeds)
                busts[t][gi] += float(np.mean(bust)) / len(seeds)
        gi = int(np.argmax(curves[t]))
        safety[t] = float(grids[t][gi])
        detail[t] = dict(ev=float(curves[t][gi]), bust=float(busts[t][gi]),
                         grid=[float(x) for x in grids[t]],
                         curve=[float(x) for x in curves[t]],
                         bust_curve=[float(x) for x in busts[t]])
    cv_ev = sum(W[t] * detail[t]["ev"] for t in TIERS)
    return safety, detail, cv_ev


def honest(lab, cfg=None, exp=None, targets=None, sample_weight=None, extra=None,
           seeds=(7, 17, 23), nboot=200, label="", n_jobs=5, grids=None, verbose=True):
    cfg = dict(DEPLOYED_CFG, **(cfg or {}))
    exp = dict(DEPLOYED_EXP, **(exp or {}))
    t0 = time.perf_counter()
    if extra is not None:
        lab.set_extra_features(extra)
    cv = cv_arrays(lab, exp, targets, sample_weight, n_jobs=n_jobs)
    safety, detail, cv_ev = choose_safety(lab, cv, cfg, seeds=seeds, nboot=nboot, grids=grids)
    arr = lab.fit_predict(lab.train_idx, lab.dev_idx, exp, targets, sample_weight)
    rep = lab.score(arr, cfg, safety, verbose=False)
    if verbose:
        tiers = " ".join(f"{t[:4]}={rep['tiers'][t]['score']:.4f}/r{rep['tiers'][t]['ratio']:.3f}"
                         f"{'' if rep['tiers'][t]['passed'] else '!BUST'}@{safety[t]:.2f}"
                         for t in TIERS)
        print(f"[honest] {label:28s} dev={rep['final']:.6f}  cvEV={cv_ev:.6f}  {tiers}  "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return dict(label=label, dev=rep["final"], cv_ev=cv_ev, safety=safety, report=rep,
                detail=detail, cv=cv, arr=arr, cfg=cfg, exp=exp)


if __name__ == "__main__":
    lab = Lab()
    r = honest(lab, label="E43 deployed baseline")
    Path("reports/lab").mkdir(parents=True, exist_ok=True)
    Path("reports/lab/baseline_honest.json").write_text(json.dumps(
        {k: v for k, v in r.items() if k in ("label", "dev", "cv_ev", "safety")} |
        {"tiers": {t: {kk: vv for kk, vv in r["report"]["tiers"][t].items() if kk != "pick"}
                   for t in TIERS},
         "bust": {t: r["detail"][t]["bust"] for t in TIERS}}, indent=2), encoding="utf-8")
    print(json.dumps({t: r["detail"][t]["bust"] for t in TIERS}, indent=2))


# --------------------------------------------------------------------------
# Exact allocator via the per-item concave cost/score envelope.
#
# The Lagrangian solution at penalty pi takes, for each item, every segment of
# its upper concave envelope whose slope exceeds pi.  Sorting all segments of
# all items by slope descending therefore traces the whole safety curve at once:
# the solution for any cap is the longest prefix that fits.  Identical output to
# the 40-step bisection in learned_router.select_models, ~100x faster, and it
# evaluates a whole batch of bootstrap samples in one pass.
def envelope_segments(ps, pc):
    """ps, pc: (..., 3) -> slopes (..., 2), dcost (..., 2), level (..., 2)."""
    s0, s1, s2 = ps[..., 0], ps[..., 1], ps[..., 2]
    c0, c1, c2 = pc[..., 0], pc[..., 1], pc[..., 2]
    d1c = np.maximum(c1 - c0, 1e-300); d2c = np.maximum(c2 - c1, 1e-300)
    m1 = (s1 - s0) / d1c
    m2 = (s2 - s1) / d2c
    merge = m2 > m1                       # mid lies below the light->k1 chord
    dcm = np.maximum(c2 - c0, 1e-300)
    mm = (s2 - s0) / dcm
    slope = np.stack([np.where(merge, mm, m1), np.where(merge, -np.inf, m2)], axis=-1)
    dcost = np.stack([np.where(merge, dcm, d1c), np.where(merge, 0.0, d2c)], axis=-1)
    level = np.stack([np.where(merge, 2, 1), np.full(merge.shape, 2)], axis=-1)
    return slope, dcost, level


def exact_allocate(ps, pc, mult, safety):
    """Batched exact allocation.  ps, pc: (B, m, 3) or (m, 3).  Returns picks."""
    single = ps.ndim == 2
    if single:
        ps = ps[None]; pc = pc[None]
    B, m, _ = ps.shape
    slope, dcost, level = envelope_segments(ps, pc)
    slope = slope.reshape(B, 2 * m); dcost = dcost.reshape(B, 2 * m)
    level = level.reshape(B, 2 * m)
    item = np.tile(np.repeat(np.arange(m), 2)[None, :], (B, 1))
    slope = np.where(slope > 0, slope, -np.inf)          # never buy a non-positive gain
    order = np.argsort(-slope, axis=1, kind="stable")
    ss = np.take_along_axis(slope, order, axis=1)
    dc = np.take_along_axis(dcost, order, axis=1)
    lv = np.take_along_axis(level, order, axis=1)
    it = np.take_along_axis(item, order, axis=1)
    dc = np.where(np.isfinite(ss), dc, 0.0)
    base = pc[:, :, 0].sum(axis=1)
    cap = base * np.maximum(1.0, mult * safety)
    cum = base[:, None] + np.cumsum(dc, axis=1)
    take = (cum <= cap[:, None] + 1e-15) & np.isfinite(ss)
    picks = np.zeros((B, m), dtype=np.int64)
    # a later segment of an item can only be taken if the earlier one was, so a
    # simple maximum over taken segments recovers the chosen model
    flat = picks.reshape(-1)
    idx = (np.arange(B)[:, None] * m + it)[take]
    np.maximum.at(flat, idx, lv[take])
    picks = flat.reshape(B, m)
    return picks[0] if single else picks


def safety_curve(ps, pc, ts, tc, mult, safeties):
    """Whole safety curve from ONE sort per bootstrap sample.

    ps, pc: predicted (B, m, 3).  ts, tc: realised (B, m, 3).  Returns
    (mean_ev, bust_rate, mean_score_when_passed) arrays over `safeties`.
    """
    B, m, _ = ps.shape
    slope, dcost, level = envelope_segments(ps, pc)
    # true deltas for the same segments
    prev = np.stack([np.zeros_like(level[..., 0]), level[..., 0]], axis=-1)
    tci = np.take_along_axis(tc, level, axis=2) - np.take_along_axis(tc, prev, axis=2)
    tsi = np.take_along_axis(ts, level, axis=2) - np.take_along_axis(ts, prev, axis=2)
    slope = np.where(slope > 0, slope, -np.inf).reshape(B, 2 * m)
    dcost = dcost.reshape(B, 2 * m); tci = tci.reshape(B, 2 * m); tsi = tsi.reshape(B, 2 * m)
    order = np.argsort(-slope, axis=1, kind="stable")
    fin = np.isfinite(np.take_along_axis(slope, order, axis=1))
    dc = np.where(fin, np.take_along_axis(dcost, order, axis=1), 0.0)
    dtc = np.where(fin, np.take_along_axis(tci, order, axis=1), 0.0)
    dts = np.where(fin, np.take_along_axis(tsi, order, axis=1), 0.0)
    base_pred = pc[:, :, 0].sum(axis=1)
    base_tc = tc[:, :, 0].sum(axis=1)
    base_ts = ts[:, :, 0].sum(axis=1)
    cum_p = np.concatenate([np.zeros((B, 1)), np.cumsum(dc, axis=1)], axis=1) + base_pred[:, None]
    cum_tc = np.concatenate([np.zeros((B, 1)), np.cumsum(dtc, axis=1)], axis=1) + base_tc[:, None]
    cum_ts = np.concatenate([np.zeros((B, 1)), np.cumsum(dts, axis=1)], axis=1) + base_ts[:, None]
    evs, busts, raws = [], [], []
    for sf in safeties:
        cap = base_pred * max(1.0, mult * float(sf))
        k = np.array([np.searchsorted(cum_p[b], cap[b] + 1e-15, side="right") - 1 for b in range(B)])
        r = np.arange(B)
        ratio = cum_tc[r, k] / base_tc
        sc = cum_ts[r, k] / m
        bust = ratio > mult
        evs.append(float(np.mean(np.where(bust, 0.0, sc))))
        busts.append(float(np.mean(bust)))
        raws.append(float(np.mean(sc)))
    return np.array(evs), np.array(busts), np.array(raws)
