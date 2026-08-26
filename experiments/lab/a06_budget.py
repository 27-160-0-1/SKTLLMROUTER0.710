# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a06 step 10: three-way variance budget of each score column.

total var = between-family var + within-family LATENT var + binomial label noise.
Also the pooled corr ceiling implied by the noise term, for comparison with the
0.42/0.49/0.42 that the deployed heads reach.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a06_features import build  # noqa: E402
from a06_counterfactual import subfam  # noqa: E402

for nm in ("train", "dev"):
    d = build(nm)
    sp, fam = d["split"], d["fam"]
    gsub = subfam(fam, d["X"], d["names"])
    print(f"\n===== {nm} =====")
    for parts, tag in ((fam, "9 regex families"), (gsub, "11 sub-families")):
        fams = sorted(set(parts))
        print(f" partition = {tag}")
        print(f"   {'model':8s} {'var_tot':>8s} {'between':>8s} {'within_lat':>11s} "
              f"{'noise':>8s} {'ceil_corr':>9s} {'ceil|fam':>9s}")
        for j, mn in enumerate(("light", "mid", "k1")):
            y = sp.score[:, j]
            vt = y.var()
            mu = np.array([y[parts == f].mean() for f in fams])
            w = np.array([(parts == f).sum() for f in fams]) / len(y)
            vb = (w * (mu - y.mean()) ** 2).sum()
            noise = 0.0
            vlat = 0.0
            for f in fams:
                m = parts == f
                yy, g = y[m], sp.ngen[m, j].mean()
                mm, vv = yy.mean(), yy.var()
                nz = (mm - mm * mm) / g if g > 1 else 0.0
                vp = max((vv - nz) / (1 - 1 / g), 0.0) if g > 1 else vv
                nz = max(vv - vp, 0.0)
                noise += m.mean() * nz
                vlat += m.mean() * vp
            print(f"   {mn:8s} {vt:8.4f} {vb:8.4f} {vlat:11.4f} {noise:8.4f} "
                  f"{np.sqrt((vb+vlat)/vt):9.3f} {np.sqrt(vb/vt):9.3f}")
    # how much of the *achievable* correlation is pure family identity
    print("  (ceil_corr = best possible corr(pred, realised) for a perfect E[s] model;")
    print("   ceil|fam  = corr you get from the partition mean alone)")
