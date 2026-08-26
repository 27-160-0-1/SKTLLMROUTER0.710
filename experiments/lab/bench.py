# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Stable comparison bench.

Selection metric and confirmation metric are deliberately separated:
  * CV:  10-fold out-of-fold predictions over Train, scored at a FIXED safety
         vector so configurations are compared at equal risk appetite, plus the
         880-bootstrap bust rate at that point (a config that games the cost
         model shows up there).
  * DEV: Train-only fit, Dev scored at the E43 deployed safety vector - the same
         number the project has always reported (baseline 0.701648).
Safety itself is optimised only once, at the end, on the chosen configuration.
"""
from __future__ import annotations
import pickle, sys, time
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP, DEPLOYED_SAFETY
import protocol as P

CV_SAFETY = {"fast": 0.94, "balanced": 0.84, "premium": 0.80}   # fixed, common risk point
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
    print(f"[bench] stage '{tag}' built in {time.perf_counter()-t0:.0f}s", flush=True)
    return cv, arr


def ident(lab, arr, ps, pc, tier):
    return ps, pc


def bench(lab, cv, arr, cfg=None, transform=ident, label="", cv_safety=None,
          dev_safety=None, seeds=(7, 17, 23), nboot=300, verbose=True):
    cfg = dict(DEPLOYED_CFG, **(cfg or {}))
    cvs = cv_safety or CV_SAFETY
    dvs = dev_safety or DEPLOYED_SAFETY
    ci = cv["idx"]; ts = lab.true_s[ci]; tc = lab.true_c[ci]; m = len(ci)
    cv_score = 0.0; cv_bust = {}; cv_tiers = {}
    for t in TIERS:
        ps, pc = lab.compose(cv, cfg, t)
        ps, pc = transform(lab, cv, ps, pc, t)
        pick = lab.allocate(ps, pc, MULTS[t], cvs[t])
        r = np.arange(m)
        sc = ts[r, pick].mean(); ratio = tc[r, pick].sum() / tc[:, 0].sum()
        b = 0.0
        for s in seeds:
            smp = np.asarray(lab.samples_for(m, s, nboot, 880))
            _e, bu, _r = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MULTS[t],
                                        [cvs[t]])
            b += float(bu[0]) / len(seeds)
        cv_bust[t] = b; cv_tiers[t] = dict(score=float(sc), ratio=float(ratio))
        cv_score += W[t] * sc
    di = arr["idx"]; dev = 0.0; dev_tiers = {}
    for t in TIERS:
        ps, pc = lab.compose(arr, cfg, t)
        ps, pc = transform(lab, arr, ps, pc, t)
        pick = lab.allocate(ps, pc, MULTS[t], dvs[t])
        r = np.arange(len(di))
        sc = lab.true_s[di][r, pick].mean()
        ratio = lab.true_c[di][r, pick].sum() / lab.true_c[di][:, 0].sum()
        ok = ratio <= MULTS[t] + 1e-15
        dev_tiers[t] = dict(score=float(sc), ratio=float(ratio), passed=bool(ok))
        dev += W[t] * (sc if ok else 0.0)
    res = dict(label=label, cv=float(cv_score), dev=float(dev), cv_bust=cv_bust,
               cv_tiers=cv_tiers, dev_tiers=dev_tiers, cfg=cfg)
    if verbose:
        s = " ".join(f"{t[:4]}={dev_tiers[t]['score']:.4f}/r{dev_tiers[t]['ratio']:.3f}"
                     f"{'' if dev_tiers[t]['passed'] else '!BUST'}" for t in TIERS)
        bb = "/".join(f"{cv_bust[t]:.3f}" for t in TIERS)
        print(f"{label:36s} CV={cv_score:.6f} bust={bb}  DEV={dev:.6f}  {s}", flush=True)
    return res
