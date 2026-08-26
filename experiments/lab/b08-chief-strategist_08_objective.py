# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b08 - E[final] versus P(final >= 0.72): the objective decision, measured.

Direct simulation of the final score over (replicate fit/row-set) x (item resample),
with the three tiers coupled through the SAME batch, so the joint bust structure is
preserved.  No independence assumption.
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


def finals(lab, arrs, cfg, sfy, nboot=250, seeds=(7, 17), n=880):
    out = []
    for arr in arrs:
        idx = arr["idx"]; ts = lab.true_s[idx]; tc = lab.true_c[idx]; m = len(idx)
        comp = {t: lab.compose(arr, cfg, t) for t in TIERS}
        for s in seeds:
            smp = np.asarray(lab.samples_for(m, s, nboot, n))
            tot = np.zeros(len(smp))
            for t in TIERS:
                ps, pc = comp[t]
                PS, PC = ps[smp], pc[smp]
                TS, TC = ts[smp], tc[smp]
                pick = P.exact_allocate(PS, PC, MULTS[t], sfy[t])
                real = np.take_along_axis(TC, pick[:, :, None], axis=2)[:, :, 0].sum(axis=1)
                base = TC[:, :, 0].sum(axis=1)
                sc = np.take_along_axis(TS, pick[:, :, None], axis=2)[:, :, 0].mean(axis=1)
                ok = (real / base) <= MULTS[t] + 1e-15
                tot += W[t] * np.where(ok, sc, 0.0)
            out.append(tot)
    return np.concatenate(out)


def report(lab, arrs, cfg, sfy, label, n=880):
    f = finals(lab, arrs, cfg, sfy, n=n)
    print(f"{label:32s} s={'/'.join('%.3f' % sfy[t] for t in TIERS)}  "
          f"E={f.mean():.4f} sd={f.std():.4f} p5={np.quantile(f,0.05):.4f} "
          f"med={np.median(f):.4f} P>=.70={np.mean(f>=0.70):.3f} P>=.72={np.mean(f>=0.72):.3f} "
          f"P>=.74={np.mean(f>=0.74):.3f} P(anybust)={np.mean(f<0.60):.3f}", flush=True)
    return f


if __name__ == "__main__":
    t0 = time.perf_counter()
    lab = Lab()
    res = pickle.loads(Path("reports/lab/b08_rot_arr.pkl").read_bytes())
    cvB, arrB = B.stage(lab, None, tag="base")
    cvL, arrL = B.stage(lab, dict(legacy_oof_meta=True), tag="legoof")
    A = {"base": [res["base"][s] for s in range(R)] + [{k: arrB[k] for k in KEYS}],
         "C1": [res["C1"][s] for s in range(R)] + [{k: arrL[k] for k in KEYS}]}

    print("=== candidate triples, n=880 ===")
    cands = [
        ("C1", dict(fast=.980, balanced=.870, premium=.850), "E43 deployed constants"),
        ("C1", dict(fast=.980, balanced=.890, premium=.880), "the '0.7017' constants"),
        ("C1", dict(fast=.960, balanced=.825, premium=.840), "C1 bench2 argmax"),
        ("base", dict(fast=.960, balanced=.840, premium=.735), "base bench2 argmax"),
        ("C1", dict(fast=.920, balanced=.860, premium=.765), "C1 rotation optimum n=880"),
        ("C1", dict(fast=.935, balanced=.900, premium=.800), "C1 rotation optimum n=1760"),
        ("C1", dict(fast=.900, balanced=.820, premium=.700), "C1 conservative"),
        ("C1", dict(fast=1.00, balanced=.920, premium=.900), "variance-seeking"),
    ]
    for name, sfy, lbl in cands:
        report(lab, A[name], DEPLOYED_CFG, sfy, lbl)

    print("\n=== same at n=1760 ===")
    for name, sfy, lbl in cands:
        report(lab, A[name], DEPLOYED_CFG, sfy, lbl, n=1760)

    print("\n=== does ANY triple reach P(final>=0.72) worth having? coarse search, n=880 ===")
    bestE = bestP = None
    for f in (0.88, 0.92, 0.96, 1.00, 1.04):
        for b in (0.80, 0.86, 0.92, 0.98):
            for p in (0.70, 0.78, 0.86, 0.94):
                sfy = dict(fast=f, balanced=b, premium=p)
                fv = finals(lab, A["C1"], DEPLOYED_CFG, sfy, nboot=120, seeds=(7,))
                e = fv.mean(); pz = float(np.mean(fv >= 0.72))
                if bestE is None or e > bestE[0]:
                    bestE = (e, pz, sfy)
                if bestP is None or pz > bestP[1]:
                    bestP = (e, pz, sfy)
    print(f"  argmax E[final]      : E={bestE[0]:.4f} P>=.72={bestE[1]:.3f} s={bestE[2]}")
    print(f"  argmax P(final>=0.72): E={bestP[0]:.4f} P>=.72={bestP[1]:.3f} s={bestP[2]}")
    print(f"[b08] done ({time.perf_counter()-t0:.0f}s)")
