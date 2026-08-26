# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""E44 — cost re-transformation / calibration.

Measured motivation (diag1/diag5): the runtime exponentiates a log-cost
prediction, so the predicted cost SUMS are 0.82 / 0.88 / 0.66 of the truth for
light / mid / k1.  The bias is model-dependent, which distorts the Lagrangian
efficiency ranking, and it forces the safety ratio down.

E36 tried Duan smearing and lost, but it re-optimised the safety ratio on the
same CV grid and never separated "fix the per-model bias" from "fix the global
scale".  Here every variant is estimated strictly out-of-fold on Train and then
evaluated by the honest protocol (CV-chosen safety, dev scored once).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, DEPLOYED_CFG, DEPLOYED_EXP, MULTS, W  # noqa: E402
import protocol as P  # noqa: E402


def cost_factors(lab, cv, cfg, mode="model"):
    """Out-of-fold multiplicative cost correction, per tier."""
    idx = cv["idx"]
    out = {}
    for t in TIERS:
        _ps, pc = lab.compose(cv, cfg, t)
        tc = lab.true_c[idx]
        if mode == "model":
            k = tc.sum(0) / pc.sum(0)
            out[t] = ("model", k / k[0])          # normalise: only relative matters
        elif mode == "model_abs":
            out[t] = ("model", tc.sum(0) / pc.sum(0))
        elif mode == "family":
            fam = lab.fam_arr[idx]
            table = {}
            for f in sorted(set(fam)):
                m = fam == f
                if m.sum() >= 20:
                    table[f] = tc[m].sum(0) / pc[m].sum(0)
            glob = tc.sum(0) / pc.sum(0)
            out[t] = ("family", table, glob)
        elif mode == "family_rel":
            fam = lab.fam_arr[idx]
            table = {}
            glob = tc.sum(0) / pc.sum(0)
            for f in sorted(set(fam)):
                m = fam == f
                if m.sum() >= 20:
                    kk = tc[m].sum(0) / pc[m].sum(0)
                    table[f] = kk / kk[0]
            out[t] = ("family", table, glob / glob[0])
    return out


def apply_factors(lab, arr, cfg, t, fac):
    ps, pc = lab.compose(arr, cfg, t)
    spec = fac[t]
    if spec[0] == "model":
        pc = pc * spec[1][None, :]
    else:
        _kind, table, glob = spec
        fam = lab.fam_arr[arr["idx"]]
        K = np.tile(glob, (len(fam), 1))
        for f, kk in table.items():
            K[fam == f] = kk
        pc = pc * K
    pc = pc.copy()
    pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
    pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
    return ps, pc


def eval_with_factors(lab, cv, arr, cfg, fac, seeds=(7, 17, 23), nboot=200, grids=None):
    grids = grids or {"fast": np.arange(0.90, 1.061, 0.01),
                      "balanced": np.arange(0.80, 1.041, 0.01),
                      "premium": np.arange(0.78, 1.041, 0.01)}
    ts = lab.true_s[cv["idx"]]; tc = lab.true_c[cv["idx"]]
    m = len(cv["idx"])
    safety, detail = {}, {}
    for t in TIERS:
        ps, pc = apply_factors(lab, cv, cfg, t, fac)
        curve = np.zeros(len(grids[t])); bust_c = np.zeros(len(grids[t]))
        for s in seeds:
            smp = np.asarray(lab.samples_for(m, s, nboot, 880))
            PS, PC, TS, TC = ps[smp], pc[smp], ts[smp], tc[smp]
            light = TC[:, :, 0].sum(axis=1)
            for gi, g in enumerate(grids[t]):
                pick = P.batch_allocate(PS, PC, MULTS[t], float(g))
                real = np.take_along_axis(TC, pick[:, :, None], axis=2)[:, :, 0].sum(axis=1)
                sc = np.take_along_axis(TS, pick[:, :, None], axis=2)[:, :, 0].mean(axis=1)
                bust = (real / light) > MULTS[t]
                curve[gi] += float(np.mean(np.where(bust, 0.0, sc))) / len(seeds)
                bust_c[gi] += float(np.mean(bust)) / len(seeds)
        gi = int(np.argmax(curve))
        safety[t] = float(grids[t][gi])
        detail[t] = dict(ev=float(curve[gi]), bust=float(bust_c[gi]))
    cv_ev = sum(W[t] * detail[t]["ev"] for t in TIERS)
    # score dev
    idx = arr["idx"]; total = 0.0; tiers = {}
    for t in TIERS:
        ps, pc = apply_factors(lab, arr, cfg, t, fac)
        pick = lab.allocate(ps, pc, MULTS[t], safety[t])
        r = np.arange(len(idx))
        ratio = lab.true_c[idx][r, pick].sum() / lab.true_c[idx][:, 0].sum()
        sc = lab.true_s[idx][r, pick].mean()
        passed = ratio <= MULTS[t] + 1e-15
        tiers[t] = dict(score=float(sc), ratio=float(ratio), passed=bool(passed))
        total += W[t] * (sc if passed else 0.0)
    return dict(dev=float(total), cv_ev=cv_ev, safety=safety, detail=detail, tiers=tiers)


def show(name, r):
    t = " ".join(f"{k[:4]}={v['score']:.4f}/r{v['ratio']:.3f}{'' if v['passed'] else '!BUST'}"
                 f"@{r['safety'][k]:.2f}" for k, v in r["tiers"].items())
    print(f"{name:34s} dev={r['dev']:.6f}  cvEV={r['cv_ev']:.6f}  {t}", flush=True)


if __name__ == "__main__":
    lab = Lab()
    cfg = dict(DEPLOYED_CFG)
    cv = P.cv_arrays(lab, DEPLOYED_EXP)
    arr = lab.fit_predict(lab.train_idx, lab.dev_idx, DEPLOYED_EXP)

    ident = {t: ("model", np.ones(3)) for t in TIERS}
    results = {}
    results["baseline (no calibration)"] = eval_with_factors(lab, cv, arr, cfg, ident)
    for mode in ("model", "model_abs", "family", "family_rel"):
        fac = cost_factors(lab, cv, cfg, mode)
        results[f"cost calibration: {mode}"] = eval_with_factors(lab, cv, arr, cfg, fac)
        if mode == "model":
            print("  OOF per-model relative factors:",
                  {t: np.round(fac[t][1], 4).tolist() for t in TIERS})
    for k, v in results.items():
        show(k, v)
    Path("reports/lab").mkdir(parents=True, exist_ok=True)
    Path("reports/lab/e44_costcal.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "detail"} for k, v in results.items()},
        indent=2, default=float), encoding="utf-8")
