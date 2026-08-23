# SPDX-License-Identifier: Apache-2.0
"""b08 - POOLED HONEST BOOTSTRAP (PHB): the risk instrument I recommend the lab adopt.

Problem with every estimator used so far:
  * bench2 bootstraps 880 items out of ONE 1,760-row train-OOF prediction set.
    One prediction set -> the estimate inherits that set's idiosyncrasies, and the
    fit is fold-averaged so across-fit dispersion is invisible.
  * a01/a09/a10/a11/a13/a14 bootstrap 880 out of the ONE 880-row dev prediction
    set.  Same defect, and dev is a measurably easy draw (mean scores
    .619/.692/.826 vs train .597/.679/.812).
  * b08 scripts 05/08 resampled items INSIDE replicates whose row sets already
    differ -> item variance counted twice.

PHB: run R independent (1,760 fit / 880 held-out) splits of the public pool, pool
all R x 880 honest out-of-sample rows into one 14,960-row prediction pool, then
bootstrap batches of size n from that pool.  Composition noise is counted once,
predictions are always out-of-sample, and the pool's mean is the population mean
rather than one row set's.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG            # noqa: E402
import bench2 as B                                                # noqa: E402
import protocol as P                                              # noqa: E402

R = 16
KEYS = ("idx", "lin", "legacy", "fam", "knn", "meta", "gain", "rank_eff", "floors")
_S = {}


def pool(lab, arrs, cfg, transform=None):
    out = {}
    for t in TIERS:
        PS, PC, TS, TC = [], [], [], []
        for a in arrs:
            ps, pc = lab.compose(a, cfg, t)
            if transform is not None:
                ps, pc = transform(lab, a, ps, pc, t)
            PS.append(ps); PC.append(pc)
            TS.append(lab.true_s[a["idx"]]); TC.append(lab.true_c[a["idx"]])
        out[t] = (np.vstack(PS), np.vstack(PC), np.vstack(TS), np.vstack(TC))
    return out


def samples(m, seed, nboot, n):
    key = (m, seed, nboot, n)
    if key not in _S:
        r = np.random.default_rng(seed)
        _S[key] = r.integers(0, m, size=(nboot, n))
    return _S[key]


def curves(pl, grids, n=880, nboot=400, seeds=(11, 29, 47)):
    out = {}
    for t in TIERS:
        PS, PC, TS, TC = pl[t]
        m = len(PS)
        g = grids[t]
        ev = np.zeros(len(g)); bu = np.zeros(len(g)); raw = np.zeros(len(g))
        for s in seeds:
            smp = samples(m, s, nboot, n)
            e, b, r = P.safety_curve(PS[smp], PC[smp], TS[smp], TC[smp], MULTS[t], g)
            ev += e / len(seeds); bu += b / len(seeds); raw += r / len(seeds)
        out[t] = dict(ev=ev, bust=bu, raw=raw, grid=g)
    return out


def final_dist(pl, sfy, n=880, nboot=1200, seeds=(11, 29, 47)):
    m = len(pl["fast"][0])
    tot = []
    for s in seeds:
        smp = samples(m, s, nboot, n)
        acc = np.zeros(nboot)
        for t in TIERS:
            PS, PC, TS, TC = pl[t]
            pick = P.exact_allocate(PS[smp], PC[smp], MULTS[t], sfy[t])
            real = np.take_along_axis(TC[smp], pick[:, :, None], axis=2)[:, :, 0].sum(axis=1)
            base = TC[smp][:, :, 0].sum(axis=1)
            sc = np.take_along_axis(TS[smp], pick[:, :, None], axis=2)[:, :, 0].mean(axis=1)
            acc += W[t] * np.where((real / base) <= MULTS[t] + 1e-15, sc, 0.0)
        tot.append(acc)
    return np.concatenate(tot)


def choose(pl, grids, n=880):
    sfy, det = {}, {}
    c = curves(pl, grids, n=n)
    for t in TIERS:
        gi = int(np.argmax(c[t]["ev"]))
        sfy[t] = float(c[t]["grid"][gi])
        det[t] = {k: float(c[t][k][gi]) for k in ("ev", "bust", "raw")}
    return sfy, det, sum(W[t] * det[t]["ev"] for t in TIERS)


def devscore(lab, arrdev, cfg, sfy, transform=None):
    di = arrdev["idx"]; tot = 0.0; rr = {}
    for t in TIERS:
        ps, pc = lab.compose(arrdev, cfg, t)
        if transform is not None:
            ps, pc = transform(lab, arrdev, ps, pc, t)
        pick = lab.allocate(ps, pc, MULTS[t], sfy[t])
        r = np.arange(len(di))
        ratio = lab.true_c[di][r, pick].sum() / lab.true_c[di][:, 0].sum()
        sc = lab.true_s[di][r, pick].mean()
        rr[t] = (sc, ratio)
        tot += W[t] * (sc if ratio <= MULTS[t] + 1e-15 else 0.0)
    return tot, rr


def mk_mult(km, kk, tiers=TIERS):
    def f(lab, arr, ps, pc, tier):
        if tier not in tiers:
            return ps, pc
        pc = pc.copy(); pc[:, 1] *= km; pc[:, 2] *= kk
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        return ps, pc
    return f


if __name__ == "__main__":
    t0 = time.perf_counter()
    lab = Lab()
    res = pickle.loads(Path("reports/lab/b08_rot_arr.pkl").read_bytes())
    cvB, arrB = B.stage(lab, None, tag="base")
    cvL, arrL = B.stage(lab, dict(legacy_oof_meta=True), tag="legoof")
    ARRS = {"base": [res["base"][s] for s in range(R)] + [{k: arrB[k] for k in KEYS}],
            "C1": [res["C1"][s] for s in range(R)] + [{k: arrL[k] for k in KEYS}]}
    G = B.GRIDS

    print("=== PHB candidate table (n=880) ===")
    cands = [("base", DEPLOYED_CFG, None, "base = deployed architecture"),
             ("C1", DEPLOYED_CFG, None, "C1 legacy-OOF meta"),
             ("C1", DEPLOYED_CFG, mk_mult(1.0, 1.5, ("balanced",)), "C1 + b02-B1 kk1.5 balanced"),
             ("C1", DEPLOYED_CFG, mk_mult(0.85, 1.0), "C1 + km0.85 global"),
             ("C1", DEPLOYED_CFG, mk_mult(1.0, 1.24), "C1 + a11-C5 kk1.24 global"),
             ("C1", dict(gain_alpha=0.7, rank_beta=0.0), None, "C1 + b05-P2 ga.7/rb0"),
             ("C1", dict(gain_alpha=1.0, rank_beta=0.0), None, "C1 + b05-P2 ga1.0/rb0"),
             ("C1", dict(fam_w=0.35), None, "C1 + fam_w .35"),
             ("C1", dict(legacy_w=1.0), None, "C1 + legacy_w 1.0"),
             ]
    store = {}
    for name, cfg, tr, lbl in cands:
        pl = pool(lab, ARRS[name], dict(DEPLOYED_CFG, **cfg), tr)
        sfy, det, EV = choose(pl, G)
        arrd = arrL if name == "C1" else arrB
        dv, rr = devscore(lab, arrd, dict(DEPLOYED_CFG, **cfg), sfy, tr)
        print(f"{lbl:32s} EV={EV:.6f} dev={dv:.6f} s={'/'.join('%.3f' % sfy[t] for t in TIERS)} "
              f"bust%={'/'.join('%.1f' % (det[t]['bust']*100) for t in TIERS)} "
              f"raw={'/'.join('%.4f' % det[t]['raw'] for t in TIERS)}", flush=True)
        store[lbl] = (pl, sfy, EV, dv)

    print("\n=== PHB safety ladder for C1 ===")
    pl = store["C1 legacy-OOF meta"][0]
    c = curves(pl, G)
    for t in TIERS:
        print(f"  -- {t} --")
        for gi in range(0, len(G[t]), 4):
            print(f"    s={G[t][gi]:.3f} raw={c[t]['raw'][gi]:.4f} bust={c[t]['bust'][gi]*100:6.2f}% "
                  f"ev={c[t]['ev'][gi]:.4f}")

    print("\n=== final-score distribution under PHB ===")
    tri = [("E43 deployed", dict(fast=.980, balanced=.870, premium=.850)),
           ("the '0.7017' triple", dict(fast=.980, balanced=.890, premium=.880)),
           ("C1 bench2 argmax", dict(fast=.960, balanced=.825, premium=.840)),
           ("base bench2 argmax", dict(fast=.960, balanced=.840, premium=.735)),
           ("PHB optimum (C1)", store["C1 legacy-OOF meta"][1]),
           ("conservative", dict(fast=.900, balanced=.820, premium=.700))]
    for lbl, sfy in tri:
        f = final_dist(pl, sfy)
        dv, _ = devscore(lab, arrL, DEPLOYED_CFG, sfy)
        print(f"  {lbl:22s} s={'/'.join('%.3f' % sfy[t] for t in TIERS)} E={f.mean():.4f} "
              f"sd={f.std():.4f} p5={np.quantile(f,.05):.4f} P>=.70={np.mean(f>=.70):.3f} "
              f"P>=.72={np.mean(f>=.72):.4f} Pbust={np.mean(f<0.60):.3f} dev={dv:.4f}", flush=True)

    print("\n=== PHB optimum at n=1760 / n=2640 ===")
    for n in (1760, 2640):
        sfy, det, EV = choose(pl, G, n=n)
        dv, _ = devscore(lab, arrL, DEPLOYED_CFG, sfy)
        print(f"  n={n}: EV={EV:.6f} dev={dv:.6f} s={'/'.join('%.3f' % sfy[t] for t in TIERS)} "
              f"bust%={'/'.join('%.1f' % (det[t]['bust']*100) for t in TIERS)}")
    print(f"[b08] done ({time.perf_counter()-t0:.0f}s)")
