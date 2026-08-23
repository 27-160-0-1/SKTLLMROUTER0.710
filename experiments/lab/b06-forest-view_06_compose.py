# SPDX-License-Identifier: Apache-2.0
"""b06 forest-view: adversarial audit of the BRIEF2 s6 candidate list.

C4 (variance re-transformation), C5 (k1 relative-price correction) and C8 (hard
cost ceiling) are all monotone reshapings of the predicted cost vector.  If they
are the same mechanism, the best member of the 2-parameter per-model multiplier
family should already contain most of what any of them buys, and stacking the
ceiling on top should add ~nothing.  Measured here.

C2 and C3 are both family-partition repairs.  Their JOINT value is bounded above
by an ORACLE partition built on the true (score, cost) profile -- no regex can
beat a partition that was allowed to look at the labels.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B

OUT = Path("reports/lab/b06_compose.json")


def mk_mult(km, kk):
    def t(lab, a, ps, pc, tier):
        q = pc.copy()
        q[:, 1] *= km; q[:, 2] *= kk
        q[:, 1] = np.maximum(q[:, 1], q[:, 0] * (1 + 1e-12))
        q[:, 2] = np.maximum(q[:, 2], q[:, 1] * (1 + 1e-12))
        return ps, q
    return t


def mk_cap(q, km=1.0, kk=1.0):
    def t(lab, a, ps, pc, tier):
        c = pc.copy()
        c[:, 1] *= km; c[:, 2] *= kk
        for j in (1, 2):
            hi = np.quantile(c[:, j], q)
            c[:, j] = np.minimum(c[:, j], hi)
        c[:, 1] = np.maximum(c[:, 1], c[:, 0] * (1 + 1e-12))
        c[:, 2] = np.maximum(c[:, 2], c[:, 1] * (1 + 1e-12))
        return ps, c
    return t


def mk_oracle_partition(lab, K, seed=0):
    """Oracle upper bound on ANY family/sub-family scheme: k-means on the true
    6-vector, then use the cluster mean as the prediction."""
    from sklearn.cluster import KMeans
    Z = lab.targets.copy()
    Z = (Z - Z.mean(0)) / Z.std(0)
    lbl = KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(Z)

    def t(lab_, a, ps, pc, tier):
        idx = a["idx"]; L = lbl[idx]
        p = np.zeros_like(ps); c = np.zeros_like(pc)
        for g in np.unique(L):
            m = L == g
            p[m] = lab_.targets[idx][m][:, :3].mean(axis=0)
            c[m] = np.exp(lab_.targets[idx][m][:, 3:6].mean(axis=0))
        c[:, 1] = np.maximum(c[:, 1], c[:, 0] * (1 + 1e-12))
        c[:, 2] = np.maximum(c[:, 2], c[:, 1] * (1 + 1e-12))
        return np.clip(p, 0, 1), c
    return t


def main():
    lab = Lab()
    cv, arr = B.stage(lab, DEPLOYED_EXP, tag="base")
    rep = {}
    base = B.run(lab, cv, arr, None, label="BASE")
    rep["base"] = dict(EV=base["EV"], dev=base["dev"], safety=base["safety"])

    # -------------------------------------- C4/C5: the per-model multiplier family
    print("\n-- C4/C5 live in one 2-parameter family: pc[:,1]*=km, pc[:,2]*=kk --")
    ks = [0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
    grid = {}
    t0 = time.perf_counter()
    for km in ks:
        row = []
        for kk in ks:
            r = B.run(lab, cv, arr, None, transform=mk_mult(km, kk), verbose=False)
            row.append(dict(km=km, kk=kk, EV=r["EV"], dev=r["dev"],
                            safety={t: r["safety"][t] for t in TIERS}))
        grid[km] = row
        print(f"  km={km:<5} EV " + " ".join(f"{x['EV']:.5f}" for x in row)
              + "   dev " + " ".join(f"{x['dev']:.5f}" for x in row), flush=True)
    flat = [x for row in grid.values() for x in row]
    bEV = max(flat, key=lambda x: x["EV"]); bdev = max(flat, key=lambda x: x["dev"])
    print(f"  best EV  km={bEV['km']} kk={bEV['kk']}  EV={bEV['EV']:.6f} dev={bEV['dev']:.6f}"
          f"  (base EV {base['EV']:.6f} dev {base['dev']:.6f})")
    print(f"  best dev km={bdev['km']} kk={bdev['kk']} EV={bdev['EV']:.6f} dev={bdev['dev']:.6f}")
    rep["mult_grid"] = flat
    print(f"  ({time.perf_counter()-t0:.0f}s)")

    # -------------------------------------- C8: hard ceiling, alone and stacked
    print("\n-- C8 hard cost ceiling: alone, then stacked on the best multiplier --")
    caps = []
    for q in (0.90, 0.95, 0.98, 0.99, 0.995):
        r1 = B.run(lab, cv, arr, None, transform=mk_cap(q), verbose=False)
        r2 = B.run(lab, cv, arr, None, transform=mk_cap(q, bEV["km"], bEV["kk"]), verbose=False)
        caps.append(dict(q=q, alone_EV=r1["EV"], alone_dev=r1["dev"],
                         stacked_EV=r2["EV"], stacked_dev=r2["dev"]))
        print(f"  cap q={q:<6} alone EV={r1['EV']:.6f} ({r1['EV']-base['EV']:+.6f}) "
              f"dev={r1['dev']:.6f} ({r1['dev']-base['dev']:+.6f})  |  "
              f"stacked EV={r2['EV']:.6f} ({r2['EV']-bEV['EV']:+.6f}) "
              f"dev={r2['dev']:.6f} ({r2['dev']-bEV['dev']:+.6f})", flush=True)
    rep["caps"] = caps

    # -------------------------------------- C2/C3: oracle partition upper bound
    print("\n-- C2/C3 upper bound: ORACLE partition of the true 6-vector --")
    orc = []
    for K in (9, 18, 40, 80):
        r = B.run(lab, cv, arr, None, transform=mk_oracle_partition(lab, K),
                  label=f"  oracle {K}-way partition")
        orc.append(dict(K=K, EV=r["EV"], dev=r["dev"]))
    rep["oracle_partition"] = orc

    # -------------------------------------- C1 reproduction + paired check
    print("\n-- C1 legacy-OOF meta feature (reproduce), and whether the 8 constants "
          "still matter under it --")
    cv1, arr1 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
    r1 = B.run(lab, cv1, arr1, None, label="  C1 legacy-OOF")
    rep["C1"] = dict(EV=r1["EV"], dev=r1["dev"], safety=r1["safety"])
    fam = B.run(lab, cv1, arr1, None, transform=lambda l, a, p, c, t: (
        np.clip(a["fam"][:, :3], 0, 1), np.exp(np.clip(a["fam"][:, 3:6], -50, 50))),
        label="  family-mean under C1 stage")
    rep["C1_fam"] = dict(EV=fam["EV"], dev=fam["dev"])

    OUT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
