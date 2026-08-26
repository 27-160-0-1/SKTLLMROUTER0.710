# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Cost anatomy: input vs output share, per-model token behaviour, family split."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, RATES, TOKEN_UNIT
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

for name in ("train", "dev"):
    sp = load_split(name)
    print(f"===== {name} n={len(sp)}")
    for j, m in enumerate(MODEL_IDS):
        f, ir, orr = RATES[m]
        cin = sp.itok[:, j] * ir / TOKEN_UNIT
        cout = sp.otok[:, j] * orr / TOKEN_UNIT
        tot = cin.sum() + cout.sum()
        print(f"  {m:11s} input-share={cin.sum()/tot:.3f} output-share={cout.sum()/tot:.3f}"
              f"  itok/gen med={np.median(sp.itok[:,j]/sp.ngen[:,j]):.0f}"
              f"  otok/gen med={np.median(sp.otok[:,j]/sp.ngen[:,j]):.0f}"
              f"  otok/gen p95={np.percentile(sp.otok[:,j]/sp.ngen[:,j],95):.0f}"
              f"  max={np.max(sp.otok[:,j]/sp.ngen[:,j]):.0f}")
    # how well does light input token count predict the others?
    print("  itok correlation light vs mid vs k1 (per gen):",
          np.round(np.corrcoef(np.vstack([sp.itok[:,j]/sp.ngen[:,j] for j in range(3)]))[0], 4))
    fam = np.array([classify_family(t) for t in sp.texts])
    print("  families:", {k: int(v) for k, v in zip(*np.unique(fam, return_counts=True))})
    print("  per-family mean score / cost-ratio:")
    for f_ in sorted(set(fam)):
        msk = fam == f_
        cs = sp.cost[msk].sum(0) / sp.cost[msk, 0].sum()
        print(f"    {f_:10s} n={msk.sum():4d} score={np.round(sp.score[msk].mean(0),3)}"
              f" costratio={np.round(cs,2)}  ngen={np.unique(sp.ngen[msk,0])}")
