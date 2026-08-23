# SPDX-License-Identifier: Apache-2.0
"""b08 - the corrected honest risk model, and the candidate table under it.

Script 05 double-counted item variance (it bootstrapped items INSIDE replicates
that already differ by their row set).  Here the budget ratio's variance is
decomposed properly:

    Var(log R over replicates)  =  Var_fit  +  Var_item(880)

`Var_item` is measured by bootstrapping items inside each replicate; `Var_fit`
is the residual.  For an evaluation batch of size n,

    sd_n = sqrt(Var_fit + Var_item(880) * 880/n)
    P(pass) = Phi( (log mult - mean log R) / sd_n )
    EV(n)   = mean raw tier score * P(pass)

That is the quantity the competition pays, for ONE fit and ONE fresh batch.
bench2's EV sets Var_fit = 0 and is therefore optimistic; script 05's rotEV
used Var_fit + 2*Var_item and is pessimistic.
"""
from __future__ import annotations
import sys, time, pickle, json
from pathlib import Path
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG            # noqa: E402
import bench2 as B                                                # noqa: E402
import protocol as P                                              # noqa: E402

R = 16
KEYS = ("idx", "lin", "legacy", "fam", "knn", "meta", "gain", "rank_eff", "floors")
CACHE = Path("reports/lab/b08_rot_arr.pkl")


def ratio_curve(ps, pc, tc, mult, safeties):
    """Realised budget ratio for a batch of samples over a safety grid. (B, G)."""
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
        cap = bp * max(1.0, mult * float(sf))
        k = np.array([np.searchsorted(cum_p[b], cap[b] + 1e-15, side="right") - 1 for b in range(Bn)])
        out[:, gi] = cum_t[np.arange(Bn), k] / bt
    return out


def tier_stats(lab, arrs, cfg, tier, grid, transform=None, nboot=120, seed=7):
    G = len(grid); A = len(arrs)
    mu = np.zeros((A, G)); raw = np.zeros((A, G)); sdi = np.zeros((A, G))
    for ai, arr in enumerate(arrs):
        idx = arr["idx"]; ts = lab.true_s[idx]; tc = lab.true_c[idx]; m = len(idx)
        ps, pc = lab.compose(arr, cfg, tier)
        if transform is not None:
            ps, pc = transform(lab, arr, ps, pc, tier)
        r = ratio_curve(ps[None], pc[None], tc[None], MULTS[tier], grid)[0]
        mu[ai] = np.log(r)
        for gi, g in enumerate(grid):
            pick = lab.allocate(ps, pc, MULTS[tier], float(g))
            raw[ai, gi] = ts[np.arange(m), pick].mean()
        smp = np.asarray(lab.samples_for(m, seed, nboot, 880))
        rb = ratio_curve(ps[smp], pc[smp], tc[smp], MULTS[tier], grid)
        sdi[ai] = np.log(rb).std(axis=0)
    m_log = mu.mean(axis=0)
    sd_tot = mu.std(axis=0, ddof=1)
    sd_item = sdi.mean(axis=0)
    sd_fit = np.sqrt(np.maximum(sd_tot ** 2 - sd_item ** 2, 0.0))
    return dict(m_log=m_log, sd_tot=sd_tot, sd_item=sd_item, sd_fit=sd_fit,
                raw=raw.mean(axis=0), ratios=np.exp(mu))


def ev_at(st, tier, grid, n=880):
    sd = np.sqrt(st["sd_fit"] ** 2 + st["sd_item"] ** 2 * (880.0 / n))
    z = (np.log(MULTS[tier]) - st["m_log"]) / np.maximum(sd, 1e-9)
    ppass = norm.cdf(z)
    return st["raw"] * ppass, ppass


def evaluate(lab, arrs, arrdev, cfg, label, transform=None, grids=None, n=880, verbose=True):
    grids = grids or B.GRIDS
    safety, det = {}, {}
    for t in TIERS:
        st = tier_stats(lab, arrs, cfg, t, grids[t], transform)
        ev, pp = ev_at(st, t, grids[t], n)
        gi = int(np.argmax(ev))
        safety[t] = float(grids[t][gi])
        det[t] = dict(ev=float(ev[gi]), raw=float(st["raw"][gi]), ppass=float(pp[gi]),
                      sd_fit=float(st["sd_fit"][gi]), sd_item=float(st["sd_item"][gi]),
                      sd_tot=float(st["sd_tot"][gi]), rmean=float(np.exp(st["m_log"][gi])))
    EV = sum(W[t] * det[t]["ev"] for t in TIERS)
    dev = 0.0; devr = {}
    for t in TIERS:
        ps, pc = lab.compose(arrdev, cfg, t)
        if transform is not None:
            ps, pc = transform(lab, arrdev, ps, pc, t)
        di = arrdev["idx"]
        pick = lab.allocate(ps, pc, MULTS[t], safety[t])
        rr = np.arange(len(di))
        ratio = lab.true_c[di][rr, pick].sum() / lab.true_c[di][:, 0].sum()
        sc = lab.true_s[di][rr, pick].mean()
        devr[t] = ratio
        dev += W[t] * (sc if ratio <= MULTS[t] + 1e-15 else 0.0)
    if verbose:
        j = lambda f: "/".join(f(t) for t in TIERS)
        print(f"{label:36s} EV{n}={EV:.6f} dev={dev:.6f} s={j(lambda t: '%.3f' % safety[t])} "
              f"Pbust%={j(lambda t: '%.1f' % ((1 - det[t]['ppass']) * 100))} "
              f"raw={j(lambda t: '%.4f' % det[t]['raw'])} devR={j(lambda t: '%.3f' % devr[t])}",
              flush=True)
    return dict(label=label, EV=EV, dev=dev, safety=safety, det=det, devr=devr)


def mk_mult(km, kk, tiers=TIERS):
    def f(lab, arr, ps, pc, tier):
        if tier not in tiers:
            return ps, pc
        pc = pc.copy(); pc[:, 1] *= km; pc[:, 2] *= kk
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        return ps, pc
    return f


def no_k1_fast(lab, arr, ps, pc, tier):
    if tier != "fast":
        return ps, pc
    ps = ps.copy(); ps[:, 2] = -1e9
    return ps, pc


if __name__ == "__main__":
    t0 = time.perf_counter()
    lab = Lab()
    res = pickle.loads(CACHE.read_bytes())
    cvB, arrB = B.stage(lab, None, tag="base")
    cvL, arrL = B.stage(lab, dict(legacy_oof_meta=True), tag="legoof")
    ARRS = {"base": [res["base"][s] for s in range(R)] + [{k: arrB[k] for k in KEYS}],
            "C1": [res["C1"][s] for s in range(R)] + [{k: arrL[k] for k in KEYS}]}

    print("\n=== variance decomposition of log(budget ratio), 17 replicates ===")
    print("  cfg   tier      s      ratio_mean sd_tot  sd_item sd_fit  fit share")
    for name, arrs in ARRS.items():
        for t in TIERS:
            g = B.GRIDS[t]
            st = tier_stats(lab, arrs, DEPLOYED_CFG, t, g)
            for gi in range(0, len(g), 8):
                fs = st["sd_fit"][gi] ** 2 / max(st["sd_tot"][gi] ** 2, 1e-12)
                print(f"  {name:5s} {t:9s} {g[gi]:.3f}  {np.exp(st['m_log'][gi]):9.4f} "
                      f"{st['sd_tot'][gi]:.4f}  {st['sd_item'][gi]:.4f}  {st['sd_fit'][gi]:.4f}  {fs:5.1%}")

    print("\n=== candidate table, EV at batch n=880 ===")
    rows = []
    rows.append(evaluate(lab, ARRS["base"], arrB, DEPLOYED_CFG, "base (deployed arch)"))
    rows.append(evaluate(lab, ARRS["C1"], arrL, DEPLOYED_CFG, "C1"))
    rows.append(evaluate(lab, ARRS["C1"], arrL, DEPLOYED_CFG, "C1 + no-k1-in-fast", no_k1_fast))
    rows.append(evaluate(lab, ARRS["C1"], arrL, DEPLOYED_CFG, "C1 + kk1.5 balanced",
                         mk_mult(1.0, 1.5, ("balanced",))))
    rows.append(evaluate(lab, ARRS["C1"], arrL, DEPLOYED_CFG, "C1 + km0.85 global", mk_mult(0.85, 1.0)))
    for ga, rb in ((0.7, 0.0), (1.0, 0.0)):
        rows.append(evaluate(lab, ARRS["C1"], arrL, dict(gain_alpha=ga, rank_beta=rb),
                             f"C1 + ga{ga}/rb{rb}"))
    print("\n=== same, EV at batch n=1760 (a13's estimate of the private set) ===")
    for r in (("base (deployed arch)", ARRS["base"], arrB, DEPLOYED_CFG, None),
              ("C1", ARRS["C1"], arrL, DEPLOYED_CFG, None),
              ("C1 + no-k1-in-fast", ARRS["C1"], arrL, DEPLOYED_CFG, no_k1_fast)):
        evaluate(lab, r[1], r[2], r[3], r[0], r[4], n=1760)

    print("\n=== what the bench2-argmax triples are worth under this model ===")
    for name, arrs, sfy in (("base@bench2", ARRS["base"], dict(fast=.960, balanced=.840, premium=.735)),
                            ("C1@bench2", ARRS["C1"], dict(fast=.960, balanced=.825, premium=.840)),
                            ("E43 deployed", ARRS["C1"], dict(fast=.980, balanced=.870, premium=.850))):
        tot = 0.0
        parts = []
        for t in TIERS:
            g = np.array([sfy[t]])
            st = tier_stats(lab, arrs, DEPLOYED_CFG, t, g)
            ev, pp = ev_at(st, t, g, 880)
            tot += W[t] * ev[0]
            parts.append(f"{t[:4]}: raw={st['raw'][0]:.4f} Pbust={100*(1-pp[0]):.1f}%")
        print(f"  {name:14s} EV880={tot:.6f}   " + "  ".join(parts))
    print(f"[b08] done ({time.perf_counter()-t0:.0f}s)")
