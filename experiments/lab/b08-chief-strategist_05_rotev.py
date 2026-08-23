# SPDX-License-Identifier: Apache-2.0
"""b08 - the ROTATION-EV criterion.

bench2's bootstrap prices ONLY item-sampling variance around a fit-averaged
prediction set.  The realised budget ratio also varies across FITS (a07 s3.6:
premium ratio 3.763 +- 0.195 over 7 meta-GBM seeds, 1 of 7 busts; b04 P1: fast
ratio sd 0.0183 over 10 seeds).  Neither single-pool bootstrap sees it.

Here we build R independent replicates of the real deployment shape
(fit 1,760 rows of the public pool -> predict the held-out 880) and store the
whole prediction bundle, so any cfg / cost transform can be re-scored.  Risk is
then the FULL variance: replicate (fit + row-set) x item bootstrap inside the
replicate.

    rotEV(safety) = mean over replicates, over item resamples,
                    of  score * 1[realised ratio <= mult]

Dev is one further replicate of exactly the same experiment and is included.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG            # noqa: E402
import bench2 as B                                                # noqa: E402
import protocol as P                                              # noqa: E402

R = 16
EXPS = {"base": None, "C1": dict(legacy_oof_meta=True)}
CACHE = Path("reports/lab/b08_rot_arr.pkl")
KEYS = ("idx", "lin", "legacy", "fam", "knn", "meta", "gain", "rank_eff", "floors")


def one(seed, expname):
    lab = Lab(verbose=False)
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(lab.n)
    fit, hold = perm[:1760], perm[1760:]
    arr = lab.fit_predict(fit, hold, EXPS[expname])
    return seed, expname, {k: arr[k] for k in KEYS}


def build(lab):
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    jobs = [(s, e) for e in EXPS for s in range(R)]
    parts = Parallel(n_jobs=4, backend="loky")(delayed(one)(s, e) for s, e in jobs)
    res = {}
    for s, e, o in parts:
        res.setdefault(e, {})[s] = o
    CACHE.write_bytes(pickle.dumps(res))
    return res


def rot_curve(lab, arrs, cfg, tier, grid, transform=None, seeds=(7, 17), nboot=150):
    """Full fit x item risk curve over a list of prediction bundles."""
    ev = np.zeros(len(grid)); bu = np.zeros(len(grid)); raw = np.zeros(len(grid))
    pr = np.zeros((len(arrs), len(grid)))
    for ai, arr in enumerate(arrs):
        idx = arr["idx"]; ts = lab.true_s[idx]; tc = lab.true_c[idx]; m = len(idx)
        ps, pc = lab.compose(arr, cfg, tier)
        if transform is not None:
            ps, pc = transform(lab, arr, ps, pc, tier)
        for s in seeds:
            smp = np.asarray(lab.samples_for(m, s, nboot, 880))
            e, b, r = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MULTS[tier], grid)
            ev += e / (len(seeds) * len(arrs))
            bu += b / (len(seeds) * len(arrs))
            raw += r / (len(seeds) * len(arrs))
        for gi, g in enumerate(grid):
            pick = lab.allocate(ps, pc, MULTS[tier], float(g))
            pr[ai, gi] = tc[np.arange(m), pick].sum() / tc[:, 0].sum()
    return dict(ev=ev, bust=bu, raw=raw, ratio=pr)


def choose(lab, arrs, arrdev, cfg, label, transform=None, grids=None, verbose=True):
    grids = grids or B.GRIDS
    safety, det = {}, {}
    for t in TIERS:
        c = rot_curve(lab, arrs, cfg, t, grids[t], transform)
        gi = int(np.argmax(c["ev"]))
        safety[t] = float(grids[t][gi])
        det[t] = dict(ev=float(c["ev"][gi]), bust=float(c["bust"][gi]), raw=float(c["raw"][gi]),
                      rmean=float(c["ratio"][:, gi].mean()), rsd=float(c["ratio"][:, gi].std()),
                      rmax=float(c["ratio"][:, gi].max()),
                      pmax=float((c["ratio"][:, gi] > MULTS[t]).mean()))
    EV = sum(W[t] * det[t]["ev"] for t in TIERS)
    # dev point score at that triple
    dev = 0.0; devd = {}
    for t in TIERS:
        ps, pc = lab.compose(arrdev, cfg, t)
        if transform is not None:
            ps, pc = transform(lab, arrdev, ps, pc, t)
        di = arrdev["idx"]
        pick = lab.allocate(ps, pc, MULTS[t], safety[t])
        r = np.arange(len(di))
        ratio = lab.true_c[di][r, pick].sum() / lab.true_c[di][:, 0].sum()
        sc = lab.true_s[di][r, pick].mean()
        ok = ratio <= MULTS[t] + 1e-15
        devd[t] = (sc, ratio, ok)
        dev += W[t] * (sc if ok else 0.0)
    if verbose:
        j = lambda f: "/".join(f(t) for t in TIERS)
        print(f"{label:40s} rotEV={EV:.6f} dev={dev:.6f}  "
              f"s={j(lambda t: '%.3f' % safety[t])}  "
              f"rotBust%={j(lambda t: '%.1f' % (det[t]['bust'] * 100))}  "
              f"repBust%={j(lambda t: '%.0f' % (det[t]['pmax'] * 100))}  "
              f"devR={j(lambda t: '%.3f' % devd[t][1])}", flush=True)
    return dict(label=label, EV=EV, dev=dev, safety=safety, det=det, devd=devd)


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
    res = build(lab)
    print(f"[b08] rotation bundles ready ({time.perf_counter()-t0:.0f}s)", flush=True)
    cvB, arrB = B.stage(lab, None, tag="base")
    cvL, arrL = B.stage(lab, dict(legacy_oof_meta=True), tag="legoof")
    ARRS = {"base": [res["base"][s] for s in range(R)] + [{k: arrB[k] for k in KEYS}],
            "C1": [res["C1"][s] for s in range(R)] + [{k: arrL[k] for k in KEYS}]}

    print("\n=== ratio dispersion at the bench2-chosen triples (17 replicates incl. dev) ===")
    for name, sfy in (("base", dict(fast=.960, balanced=.840, premium=.735)),
                      ("C1", dict(fast=.960, balanced=.825, premium=.840))):
        for t in TIERS:
            g = np.array([sfy[t]])
            c = rot_curve(lab, ARRS[name], DEPLOYED_CFG, t, g)
            print(f"  {name:5s} {t:9s} s={sfy[t]:.3f}  ratio mean={c['ratio'][:,0].mean():.4f} "
                  f"sd={c['ratio'][:,0].std():.4f} max={c['ratio'][:,0].max():.4f} "
                  f"repbust={100*(c['ratio'][:,0]>MULTS[t]).mean():.0f}%  "
                  f"fullBust={c['bust'][0]*100:.1f}%  rotEVtier={c['ev'][0]:.4f}")

    print("\n=== rotation-EV optimal triples ===")
    out = {}
    out["base"] = choose(lab, ARRS["base"], arrB, DEPLOYED_CFG, "base")
    out["C1"] = choose(lab, ARRS["C1"], arrL, DEPLOYED_CFG, "C1")

    print("\n=== candidate composition on top of C1 (rotation-EV) ===")
    for kk in (1.24, 1.5):
        choose(lab, ARRS["C1"], arrL, DEPLOYED_CFG, f"C1 + kk={kk} balanced", mk_mult(1.0, kk, ("balanced",)))
    for kk in (1.24, 1.5, 2.0):
        choose(lab, ARRS["C1"], arrL, DEPLOYED_CFG, f"C1 + kk={kk} premium", mk_mult(1.0, kk, ("premium",)))
    for km in (0.85, 1.15):
        choose(lab, ARRS["C1"], arrL, DEPLOYED_CFG, f"C1 + km={km} global", mk_mult(km, 1.0))
    for ga, rb in ((0.5, 0.0), (0.7, 0.0), (1.0, 0.0), (0.0, 0.0)):
        choose(lab, ARRS["C1"], arrL, dict(gain_alpha=ga, rank_beta=rb),
               f"C1 + gain_alpha={ga} rank_beta={rb}")
    print(f"[b08] done ({time.perf_counter()-t0:.0f}s)")
