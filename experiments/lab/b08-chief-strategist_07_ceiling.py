# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b08 - (1) the no-bust envelope of today's predictions (is 0.72 reachable at ALL?)
       (2) the oracle ladder on dev (true cost / true score)
       (3) an untested lever: replace the BUDGET DENOMINATOR with an exact estimate
       (4) batch-size regret matrix for the safety triple
       (5) what a seed-variance allowance does to the triple
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG            # noqa: E402
import bench2 as B                                                # noqa: E402
import protocol as P                                              # noqa: E402
from importlib import import_module                               # noqa: E402
RK = import_module("b08-chief-strategist_06_risk")

R = 16
KEYS = RK.KEYS


def ratio_curve_fac(ps, pc, tc, mult, safeties, fac):
    Bn, m, _ = ps.shape
    slope, dcost, level = P.envelope_segments(ps, pc)
    prev = np.stack([np.zeros_like(level[..., 0]), level[..., 0]], axis=-1)
    tci = np.take_along_axis(tc, level, axis=2) - np.take_along_axis(tc, prev, axis=2)
    slope = np.where(slope > 0, slope, -np.inf).reshape(Bn, 2 * m)
    dcost = dcost.reshape(Bn, 2 * m); tci = tci.reshape(Bn, 2 * m)
    order = np.argsort(-slope, axis=1, kind="stable")
    fin = np.isfinite(np.take_along_axis(slope, order, axis=1))
    dc = np.where(fin, np.take_along_axis(dcost, order, axis=1), 0.0)
    dtc = np.where(fin, np.take_along_axis(tci, order, axis=1), 0.0)
    bp = pc[:, :, 0].sum(axis=1); bt = tc[:, :, 0].sum(axis=1)
    cum_p = np.concatenate([np.zeros((Bn, 1)), np.cumsum(dc, axis=1)], axis=1) + bp[:, None]
    cum_t = np.concatenate([np.zeros((Bn, 1)), np.cumsum(dtc, axis=1)], axis=1) + bt[:, None]
    out = np.zeros((Bn, len(safeties)))
    for gi, sf in enumerate(safeties):
        cap = bp * np.maximum(1.0, mult * float(sf) * fac)
        k = np.array([np.searchsorted(cum_p[b], cap[b] + 1e-15, side="right") - 1 for b in range(Bn)])
        out[:, gi] = cum_t[np.arange(Bn), k] / bt
    return out


def tier_stats_den(lab, arrs, cfg, tier, grid, exact_den=False, nboot=120, seed=7):
    A = len(arrs); G = len(grid)
    mu = np.zeros((A, G)); raw = np.zeros((A, G)); sdi = np.zeros((A, G))
    for ai, arr in enumerate(arrs):
        idx = arr["idx"]; ts = lab.true_s[idx]; tc = lab.true_c[idx]; m = len(idx)
        ps, pc = lab.compose(arr, cfg, tier)
        f = float(tc[:, 0].sum() / pc[:, 0].sum()) if exact_den else 1.0
        r = ratio_curve_fac(ps[None], pc[None], tc[None], MULTS[tier], grid, np.array([f]))[0]
        mu[ai] = np.log(r)
        for gi, g in enumerate(grid):
            pick = lab.allocate(ps, pc, MULTS[tier], float(g) * f)
            raw[ai, gi] = ts[np.arange(m), pick].mean()
        smp = np.asarray(lab.samples_for(m, seed, nboot, 880))
        PS, PC, TC = ps[smp], pc[smp], tc[smp]
        fb = (TC[:, :, 0].sum(axis=1) / PC[:, :, 0].sum(axis=1)) if exact_den else np.ones(len(smp))
        rb = ratio_curve_fac(PS, PC, TC, MULTS[tier], grid, fb)
        sdi[ai] = np.log(rb).std(axis=0)
    sd_tot = mu.std(axis=0, ddof=1); sd_item = sdi.mean(axis=0)
    return dict(m_log=mu.mean(axis=0), sd_tot=sd_tot, sd_item=sd_item,
                sd_fit=np.sqrt(np.maximum(sd_tot ** 2 - sd_item ** 2, 0.0)),
                raw=raw.mean(axis=0))


def best(lab, arrs, cfg, label, exact_den=False, n=880, seed_sd=None):
    tot = 0.0; sfy = {}; parts = []
    for t in TIERS:
        g = B.GRIDS[t]
        st = tier_stats_den(lab, arrs, cfg, t, g, exact_den)
        sd = np.sqrt(st["sd_fit"] ** 2 + st["sd_item"] ** 2 * (880.0 / n)
                     + (0.0 if seed_sd is None else seed_sd[t] ** 2))
        pp = norm.cdf((np.log(MULTS[t]) - st["m_log"]) / np.maximum(sd, 1e-9))
        ev = st["raw"] * pp
        gi = int(np.argmax(ev))
        sfy[t] = float(g[gi]); tot += W[t] * ev[gi]
        parts.append(f"{t[:4]} s={g[gi]:.3f} raw={st['raw'][gi]:.4f} Pb={100*(1-pp[gi]):.1f}%")
    print(f"{label:34s} EV{n}={tot:.6f}  s=" + "/".join(f"{sfy[t]:.3f}" for t in TIERS)
          + "  " + " | ".join(parts), flush=True)
    return tot, sfy


def ev_of_triple(lab, arrs, cfg, sfy, n):
    tot = 0.0
    for t in TIERS:
        g = np.array([sfy[t]])
        st = RK.tier_stats(lab, arrs, cfg, t, g)
        sd = np.sqrt(st["sd_fit"] ** 2 + st["sd_item"] ** 2 * (880.0 / n))
        pp = norm.cdf((np.log(MULTS[t]) - st["m_log"]) / np.maximum(sd, 1e-9))
        tot += W[t] * st["raw"][0] * pp[0]
    return tot


if __name__ == "__main__":
    t0 = time.perf_counter()
    lab = Lab()
    res = pickle.loads(Path("reports/lab/b08_rot_arr.pkl").read_bytes())
    cvB, arrB = B.stage(lab, None, tag="base")
    cvL, arrL = B.stage(lab, dict(legacy_oof_meta=True), tag="legoof")
    ARRS = {"base": [res["base"][s] for s in range(R)] + [{k: arrB[k] for k in KEYS}],
            "C1": [res["C1"][s] for s in range(R)] + [{k: arrL[k] for k in KEYS}]}

    print("\n=== 1. NO-BUST ENVELOPE: best weighted raw score at ANY safety ===")
    for name, arrs in ARRS.items():
        tot = 0.0; parts = []
        for t in TIERS:
            g = np.arange(0.55, 1.601, 0.01)
            st = RK.tier_stats(lab, arrs, DEPLOYED_CFG, t, g, nboot=8)
            gi = int(np.argmax(st["raw"]))
            tot += W[t] * st["raw"][gi]
            parts.append(f"{t[:4]}={st['raw'][gi]:.4f}@{g[gi]:.2f}(r{np.exp(st['m_log'][gi]):.3f})")
        print(f"  {name:5s} envelope(rotation mean, budget IGNORED) = {tot:.6f}   " + " ".join(parts))

    print("\n=== 2. oracle ladder on dev (C1 stage) ===")
    di = arrL["idx"]; ts = lab.true_s[di]; tc = lab.true_c[di]
    def sc(psf, pcf, sfy):
        tot = 0.0; det = []
        for t in TIERS:
            ps, pc = lab.compose(arrL, DEPLOYED_CFG, t)
            ps = psf(ps); pc = pcf(pc)
            pick = lab.allocate(ps, pc, MULTS[t], sfy[t])
            r = np.arange(len(di))
            ratio = tc[r, pick].sum() / tc[:, 0].sum(); s = ts[r, pick].mean()
            ok = ratio <= MULTS[t] + 1e-15
            tot += W[t] * (s if ok else 0.0); det.append(f"{t[:4]}={s:.4f}/r{ratio:.3f}{'' if ok else '!'}")
        return tot, " ".join(det)
    ident = lambda x: x
    one = dict(fast=1.0, balanced=1.0, premium=1.0)
    for lab_, psf, pcf, sfy in (
            ("pred s, pred c, C1 rot-optimal", ident, ident, dict(fast=.920, balanced=.860, premium=.765)),
            ("pred s, TRUE c, safety 1.0", ident, lambda p: tc, one),
            ("TRUE s, pred c, rot-optimal", lambda p: ts, ident, dict(fast=.920, balanced=.860, premium=.765)),
            ("TRUE s, TRUE c, safety 1.0", lambda p: ts, lambda p: tc, one)):
        v, d = sc(psf, pcf, sfy)
        print(f"  {lab_:34s} dev={v:.6f}   {d}")

    print("\n=== 3. exact budget DENOMINATOR (oracle upper bound on a light-total estimator) ===")
    for name, arrs in ARRS.items():
        for ed in (False, True):
            best(lab, arrs, DEPLOYED_CFG, f"{name} exact_den={ed}", exact_den=ed)

    print("\n=== 4. batch-size regret of the safety triple (C1) ===")
    tri = {}
    for n in (880, 1760, 2640):
        _, s = best(lab, ARRS["C1"], DEPLOYED_CFG, f"C1 optimum for n={n}", n=n)
        tri[n] = s
    print("   rows = triple chosen for n, cols = realised n")
    for nc, s in tri.items():
        row = "  chose@%-5d" % nc
        for n in (880, 1760, 2640):
            row += f"  n={n}: {ev_of_triple(lab, ARRS['C1'], DEPLOYED_CFG, s, n):.6f}"
        print(row)

    print("\n=== 5. adding an across-SEED ratio-variance allowance (a07 s3.6 / b04 P1) ===")
    # a07: premium ratio sd 0.195 on a mean of 3.763 -> 0.0518 in log units
    # b04: fast ratio sd 0.0183 on a mean of 1.2393 -> 0.0148 in log units
    for lbl, ssd in (("no seed allowance", None),
                     ("seed sd f/b/p = .015/.025/.052",
                      dict(fast=0.0148, balanced=0.025, premium=0.0518))):
        best(lab, ARRS["C1"], DEPLOYED_CFG, f"C1 {lbl}", seed_sd=ssd)
    print(f"[b08] done ({time.perf_counter()-t0:.0f}s)")
