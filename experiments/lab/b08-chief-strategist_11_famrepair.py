# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b08 - the last open score-side candidate: a08's family-regex repair (C2/C3),
measured under the pooled honest bootstrap on top of C1.
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
from importlib import import_module                               # noqa: E402
PHB = import_module("b08-chief-strategist_10_phb")
import famrepair                                                  # noqa: E402

R = 16
KEYS = PHB.KEYS
EXP = dict(legacy_oof_meta=True)
CACHE = Path("reports/lab/b08_rot_famv3.pkl")


def labels_for(lab):
    return [famrepair.classify_v3(t) for t in lab.texts]


def one(seed):
    lab = Lab(verbose=False)
    lab.set_family(labels_for(lab))
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(lab.n)
    fit, hold = perm[:1760], perm[1760:]
    arr = lab.fit_predict(fit, hold, EXP)
    return seed, {k: arr[k] for k in KEYS}


if __name__ == "__main__":
    t0 = time.perf_counter()
    lab = Lab()
    lab0_names = list(lab.fam_names)
    v3 = labels_for(lab)
    moved = sum(1 for a, b in zip(lab0_names, v3) if a != b)
    print(f"family repair moves {moved}/{lab.n} = {moved/lab.n:.1%} of items")

    if CACHE.exists():
        res = pickle.loads(CACHE.read_bytes())
        devarr = res.pop("dev")
    else:
        parts = Parallel(n_jobs=4, backend="loky")(delayed(one)(s) for s in range(R))
        res = {s: o for s, o in parts}
        lab.set_family(v3)
        d = lab.fit_predict(lab.train_idx, lab.dev_idx, EXP)
        devarr = {k: d[k] for k in KEYS}
        CACHE.write_bytes(pickle.dumps(dict(res, dev=devarr)))
        print(f"[b08] famv3 rotation built ({time.perf_counter()-t0:.0f}s)", flush=True)

    lab.set_family(v3)
    arrs_v3 = [res[s] for s in range(R)] + [devarr]

    base = pickle.loads(Path("reports/lab/b08_rot_arr.pkl").read_bytes())
    lab_ref = Lab(verbose=False)
    cvL, arrL = B.stage(lab_ref, EXP, tag="legoof")
    arrs_c1 = [base["C1"][s] for s in range(R)] + [{k: arrL[k] for k in KEYS}]

    print("\n=== PHB, n=880 ===")
    for lbl, LB, arrs, arrd in (("C1 (deployed families)", lab_ref, arrs_c1, arrL),
                                ("C1 + famv3 repair", lab, arrs_v3, devarr)):
        pl = PHB.pool(LB, arrs, DEPLOYED_CFG)
        sfy, det, EV = PHB.choose(pl, B.GRIDS)
        dv, rr = PHB.devscore(LB, arrd, DEPLOYED_CFG, sfy)
        f = PHB.final_dist(pl, sfy)
        print(f"{lbl:26s} EV={EV:.6f} dev={dv:.6f} s={'/'.join('%.3f' % sfy[t] for t in TIERS)} "
              f"bust%={'/'.join('%.1f' % (det[t]['bust']*100) for t in TIERS)} "
              f"raw={'/'.join('%.4f' % det[t]['raw'] for t in TIERS)} "
              f"p5={np.quantile(f,.05):.4f}", flush=True)
    print(f"[b08] done ({time.perf_counter()-t0:.0f}s)")
