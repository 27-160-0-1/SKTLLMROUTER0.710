# SPDX-License-Identifier: Apache-2.0
"""Definitive comparison protocol.

For every configuration:
  1. 10-fold OOF predictions over Train only (legacy head refitted per fold, so
     no component is in-sample on the rows the criterion is computed from).
  2. Per-tier safety ratio = argmax of the 3-seed x 400-sample bootstrap EV over
     880-item resamples of those OOF rows, on a wide grid.  Dev is never read.
  3. Report BOTH
       EV   - the honest expected final score (bust priced in), and
       DEV  - the single-sample held-out dev score at the chosen safety,
     because the competition pays the single sample but a rational choice
     maximises the expectation.
"""
from __future__ import annotations
import pickle, sys, time
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP
import protocol as P

GRIDS = {"fast": np.arange(0.80, 1.041, 0.005),
         "balanced": np.arange(0.60, 1.001, 0.005),
         "premium": np.arange(0.55, 1.001, 0.005)}
FOLDS = 10


def cv_arrays(lab, exp=None, folds=FOLDS, seed=123, n_jobs=6, **kw):
    idx = lab.train_idx
    fold_of = np.random.default_rng(seed).integers(0, folds, size=len(idx))
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(lab.fit_predict)(idx[fold_of != f], idx[fold_of == f], exp, **kw)
        for f in range(folds))
    out = {"idx": np.concatenate([p["idx"] for p in parts])}
    for k in ("lin", "legacy", "fam", "knn", "meta", "gain", "rank_eff"):
        out[k] = np.vstack([p[k] for p in parts])
    out["floors"] = parts[0]["floors"]
    return out


def stage(lab, exp=None, tag="base", force=False, **kw):
    f = Path(f"reports/lab/stage_{tag}.pkl")
    if f.exists() and not force:
        b = pickle.loads(f.read_bytes())
        return b["cv"], b["arr"]
    t0 = time.perf_counter()
    cv = cv_arrays(lab, exp, **kw)
    arr = lab.fit_predict(lab.train_idx, lab.dev_idx, exp, **kw)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps({"cv": cv, "arr": arr}))
    print(f"[bench2] stage '{tag}' built in {time.perf_counter()-t0:.0f}s", flush=True)
    return cv, arr


def ident(lab, arr, ps, pc, tier):
    return ps, pc


def stress_costs(lab, idx, smp, kind, rng_seed=0):
    """Return the realised-cost tensor for one stress scenario.

    b01 showed the fast tier's budget is decided by a single runaway upgrade
    (dev ep 2437: mid emitted 33,459 output tokens, 6.3% of the whole light
    baseline).  Train contains four such items per 1,760 rows, so a fresh 880
    batch carries about two - but which item is not predictable from the prompt.
    The honest way to price that is to inject one into every resample rather
    than to hope the bootstrap happens to draw one.
    """
    TC = lab.true_c[idx][smp].copy()
    if kind == "plain":
        return TC
    if kind == "inflate":                       # think 25% longer, mid 10% longer
        TC[:, :, 2] *= 1.25
        TC[:, :, 1] *= 1.10
        return TC
    if kind == "runaway":
        light_total = TC[:, :, 0].sum(axis=1)
        fam = lab.fam_arr[idx][smp]
        mathy = np.isin(fam, ("dmmath", "gsm8k_or_other", "aime", "code"))
        delta = TC[:, :, 1] - TC[:, :, 0]
        big = np.where(mathy, delta, np.inf)
        j = np.argmin(big, axis=1)
        b = np.arange(TC.shape[0])
        hit = np.isfinite(big[b, j])
        TC[b[hit], j[hit], 1] = TC[b[hit], j[hit], 0] + 0.065 * light_total[hit]
        TC[b[hit], j[hit], 2] = np.maximum(TC[b[hit], j[hit], 2], TC[b[hit], j[hit], 1])
        return TC
    raise ValueError(kind)


def run(lab, cv, arr, cfg=None, transform=ident, label="", seeds=(7, 17, 23), nboot=400,
        grids=None, verbose=True, fixed_safety=None, keep_curves=False,
        scenarios=("plain", "runaway", "inflate")):
    cfg = dict(DEPLOYED_CFG, **(cfg or {}))
    grids = grids or GRIDS
    ci = cv["idx"]; ts = lab.true_s[ci]; tc = lab.true_c[ci]; m = len(ci)
    safety, det, curves = {}, {}, {}
    for t in TIERS:
        ps, pc = lab.compose(cv, cfg, t)
        ps, pc = transform(lab, cv, ps, pc, t)
        g = grids[t]
        ev = np.zeros(len(g)); bu = np.zeros(len(g)); raw = np.zeros(len(g))
        nsc = len(scenarios)
        for s in seeds:
            smp = np.asarray(lab.samples_for(m, s, nboot, 880))
            for sc in scenarios:
                TC = stress_costs(lab, ci, smp, sc)
                e, b, r = P.safety_curve(ps[smp], pc[smp], ts[smp], TC, MULTS[t], g)
                ev += e / (len(seeds) * nsc)
                bu += b / (len(seeds) * nsc)
                raw += r / (len(seeds) * nsc)
        gi = int(np.argmax(ev)) if fixed_safety is None else int(np.argmin(np.abs(g - fixed_safety[t])))
        safety[t] = float(g[gi])
        det[t] = dict(ev=float(ev[gi]), bust=float(bu[gi]), raw=float(raw[gi]))
        if keep_curves:
            curves[t] = dict(grid=g.tolist(), ev=ev.tolist(), bust=bu.tolist(), raw=raw.tolist())
    EV = sum(W[t] * det[t]["ev"] for t in TIERS)
    di = arr["idx"]; dev = 0.0; dt = {}
    for t in TIERS:
        ps, pc = lab.compose(arr, cfg, t)
        ps, pc = transform(lab, arr, ps, pc, t)
        pick = lab.allocate(ps, pc, MULTS[t], safety[t])
        r = np.arange(len(di))
        sc = lab.true_s[di][r, pick].mean()
        ratio = lab.true_c[di][r, pick].sum() / lab.true_c[di][:, 0].sum()
        ok = ratio <= MULTS[t] + 1e-15
        dt[t] = dict(score=float(sc), ratio=float(ratio), passed=bool(ok),
                     margin=float(MULTS[t] / ratio - 1.0))
        dev += W[t] * (sc if ok else 0.0)
    res = dict(label=label, EV=float(EV), dev=float(dev), safety=safety, det=det,
               dev_tiers=dt, cfg=cfg, curves=curves)
    if verbose:
        s = " ".join(f"{t[:4]}={dt[t]['score']:.4f}/r{dt[t]['ratio']:.3f}"
                     f"{'' if dt[t]['passed'] else '!BUST'}@{safety[t]:.3f}" for t in TIERS)
        bb = "/".join(f"{det[t]['bust']*100:.1f}" for t in TIERS)
        print(f"{label:34s} EV={EV:.6f} bust%={bb}  dev={res['dev']:.6f}  {s}", flush=True)
    return res
