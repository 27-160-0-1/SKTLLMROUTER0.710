# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b08 - ROTATION PROTOCOL: is the held-out dev budget ratio an outlier, or is the
train-OOF pool optimistic?

bench2 chooses the safety triple on 10-fold OOF rows (fit size 1,584, predicted in
176-row folds) and then scores one 880-row held-out set predicted by a model fit on
1,760 rows.  Those two prediction sets are NOT interchangeable: under C1 the fast
tier's realised ratio is 1.190 on the OOF pool and 1.249 on dev at the same safety.

This script removes the confound by generating K fresh (1,760 fit / 880 held-out)
splits of the whole 2,640-row public pool - the exact shape of the real deployment -
and reporting the distribution of the realised budget ratio.  Dev is then just one
draw from that distribution and we can say whether it is typical.

Uses dev rows inside the FIT set for some replicates; that is a diagnostic, not a
selection, and the rules permit training on public Train+Dev.
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

K = 8
EXPS = {"base": None, "C1": dict(legacy_oof_meta=True)}
CACHE = Path("reports/lab/b08_rotation.pkl")


def one(seed, expname):
    lab = Lab(verbose=False)
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(lab.n)
    fit, hold = perm[:1760], perm[1760:]
    arr = lab.fit_predict(fit, hold, EXPS[expname])
    out = {}
    for t in TIERS:
        ps, pc = lab.compose(arr, DEPLOYED_CFG, t)
        g = B.GRIDS[t]
        ratio = np.zeros(len(g)); score = np.zeros(len(g))
        ts = lab.true_s[hold]; tc = lab.true_c[hold]
        for gi, s in enumerate(g):
            pick = lab.allocate(ps, pc, MULTS[t], float(s))
            r = np.arange(len(hold))
            ratio[gi] = tc[r, pick].sum() / tc[:, 0].sum()
            score[gi] = ts[r, pick].mean()
        out[t] = dict(ratio=ratio, score=score)
    return (seed, expname, out)


if __name__ == "__main__":
    t0 = time.perf_counter()
    _LAB = Lab()
    if CACHE.exists():
        res = pickle.loads(CACHE.read_bytes())
    else:
        jobs = [(s, e) for e in EXPS for s in range(K)]
        parts = Parallel(n_jobs=4, backend="loky")(delayed(one)(s, e) for s, e in jobs)
        res = {}
        for s, e, o in parts:
            res.setdefault(e, {})[s] = o
        CACHE.write_bytes(pickle.dumps(res))
        print(f"[b08] rotation built in {time.perf_counter()-t0:.0f}s", flush=True)

    # dev reference (the real held-out split)
    cvB, arrB = B.stage(_LAB, None, tag="base")
    cvL, arrL = B.stage(_LAB, dict(legacy_oof_meta=True), tag="legoof")
    devref = {}
    for name, cv, arr in (("base", cvB, arrB), ("C1", cvL, arrL)):
        devref[name] = {}
        for t in TIERS:
            g = B.GRIDS[t]
            ratio = np.zeros(len(g)); score = np.zeros(len(g))
            oofr = np.zeros(len(g))
            ps, pc = _LAB.compose(arr, DEPLOYED_CFG, t)
            di = arr["idx"]
            for gi, s in enumerate(g):
                pick = _LAB.allocate(ps, pc, MULTS[t], float(s))
                r = np.arange(len(di))
                ratio[gi] = _LAB.true_c[di][r, pick].sum() / _LAB.true_c[di][:, 0].sum()
                score[gi] = _LAB.true_s[di][r, pick].mean()
            pso, pco = _LAB.compose(cv, DEPLOYED_CFG, t)
            ci = cv["idx"]
            for gi, s in enumerate(g):
                pick = _LAB.allocate(pso, pco, MULTS[t], float(s))
                r = np.arange(len(ci))
                oofr[gi] = _LAB.true_c[ci][r, pick].sum() / _LAB.true_c[ci][:, 0].sum()
            devref[name][t] = dict(ratio=ratio, score=score, oof=oofr)

    print("\n=== realised budget ratio at fixed safety: 8 fresh 1760/880 splits vs dev vs OOF pool ===")
    for name in EXPS:
        print(f"\n-- {name} --")
        for t in TIERS:
            g = B.GRIDS[t]
            R = np.stack([res[name][s][t]["ratio"] for s in range(K)])
            print(f"  {t} (cap {MULTS[t]})")
            print("    s      rot_mean rot_sd  rot_max  rot_bust  dev     OOFpool")
            for gi, s in enumerate(g):
                if gi % 4:
                    continue
                bust = float((R[:, gi] > MULTS[t]).mean())
                print(f"    {s:.3f}  {R[:, gi].mean():7.4f} {R[:, gi].std():6.4f} "
                      f"{R[:, gi].max():7.4f}  {bust*100:6.1f}%  {devref[name][t]['ratio'][gi]:7.4f} "
                      f"{devref[name][t]['oof'][gi]:7.4f}")

    print("\n=== largest safety at which EVERY replicate (and dev) passes ===")
    for name in EXPS:
        row = []
        for t in TIERS:
            g = B.GRIDS[t]
            R = np.stack([res[name][s][t]["ratio"] for s in range(K)] + [devref[name][t]["ratio"]])
            ok = (R <= MULTS[t] + 1e-12).all(axis=0)
            idx = np.where(ok)[0]
            row.append(f"{t}={g[idx.max()]:.3f}" if len(idx) else f"{t}=none")
        print(f"  {name}: " + "  ".join(row))

    print("\n=== mean held-out tier score across replicates at a common safety ===")
    for t in TIERS:
        g = B.GRIDS[t]
        print(f"  -- {t} --")
        for gi, s in enumerate(g):
            if gi % 6:
                continue
            line = f"    {s:.3f} "
            for name in EXPS:
                S = np.stack([res[name][sd][t]["score"] for sd in range(K)])
                R = np.stack([res[name][sd][t]["ratio"] for sd in range(K)])
                eff = np.where(R[:, gi] > MULTS[t], 0.0, S[:, gi])
                line += f" {name}: raw={S[:, gi].mean():.4f} ev={eff.mean():.4f}"
            print(line)
    print(f"[b08] done ({time.perf_counter()-t0:.0f}s)")
