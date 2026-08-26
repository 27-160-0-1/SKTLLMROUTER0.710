# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""A01 Q4c: honest bust probability at n=880, and under the organiser's own
documented dev->private cost inflation (baselines/README.md: hash-regex premium
dev ratio 3.985 -> private ~4.2, i.e. x1.054)."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labdata as L

tr, dv = L.load_all()
z = np.load(ROOT/"reports/lab/dev_preds_e43.npz", allow_pickle=True)
S = {t: z[f"score_{t}"] for t in L.TIERS}
C = {t: z[f"cost_{t}"] for t in L.TIERS}

def run(idx, tier, safety, Sm, Cm, cost_mult=1.0):
    ps, pc = Sm[idx], Cm[idx]
    mult = L.TIER_MULT[tier]
    lt = pc[:, 0].sum(); cap = lt * max(1.0, mult * safety)
    ch = lambda pen: (ps - pen*pc/lt).argmax(axis=1)
    sel = ch(0.0); tot = pc[np.arange(len(sel)), sel].sum()
    if tot > cap:
        lo, hi = 0.0, 1.0
        sel = ch(hi); tot = pc[np.arange(len(sel)), sel].sum()
        while tot > cap and hi < 2**60:
            lo, hi = hi, hi*2.0; sel = ch(hi); tot = pc[np.arange(len(sel)), sel].sum()
        for _ in range(40):
            m = (lo+hi)/2.0; cd = ch(m); ct = pc[np.arange(len(cd)), cd].sum()
            if ct <= cap: hi, sel, tot = m, cd, ct
            else: lo = m
    if tot > cap: sel = np.zeros(len(ps), dtype=int)
    tc, ts = dv.cost[idx], dv.score[idx]
    num = tc[np.arange(len(sel)), sel].copy()
    num[sel > 0] *= cost_mult                 # inflate only the non-light spend
    ratio = num.sum()/tc[:, 0].sum()
    ok = ratio <= mult + 1e-15
    return float(ts[np.arange(len(sel)), sel].mean()), float(ratio), bool(ok)

rng = np.random.default_rng(7)
n = len(dv); NB = 2000
for mult, lbl in ((1.0, "nominal"), (1.054, "x1.054 (organiser's documented dev->private drift)")):
    print(f"\n== 880-item bootstrap from dev, honest train-only predictions, {lbl} ==")
    print("safety set              fast bust%  bal bust%  prem bust%   E[final]")
    for name, sa in (("deployed .98/.87/.85", dict(fast=.98, balanced=.87, premium=.85)),
                     ("holdout  .98/.89/.88", dict(fast=.98, balanced=.89, premium=.88)),
                     ("dev-opt  1.00/.985/.89", dict(fast=1.0, balanced=.985, premium=.89)),
                     ("robust   .98/.85/.82", dict(fast=.98, balanced=.85, premium=.82))):
        bust = {t: 0 for t in L.TIERS}; ev = 0.0
        for b in range(NB):
            idx = rng.integers(0, n, size=n)
            f = 0.0
            for t in L.TIERS:
                raw, ratio, ok = run(idx, t, sa[t], S[t], C[t], mult)
                if not ok: bust[t] += 1
                f += L.TIER_WEIGHT[t]*(raw if ok else 0.0)
            ev += f
        print(f"{name:22s}  {100*bust['fast']/NB:8.2f}  {100*bust['balanced']/NB:9.2f}  "
              f"{100*bust['premium']/NB:10.2f}   {ev/NB:.6f}")
    # point estimate on the actual 880
    print("  point estimate on the actual dev 880:")
    for name, sa in (("deployed .98/.87/.85", dict(fast=.98, balanced=.87, premium=.85)),
                     ("holdout  .98/.89/.88", dict(fast=.98, balanced=.89, premium=.88))):
        parts = []
        f = 0.0
        for t in L.TIERS:
            raw, ratio, ok = run(np.arange(n), t, sa[t], S[t], C[t], mult)
            parts.append(f"{t}: ratio={ratio:.3f}/{L.TIER_MULT[t]} pass={ok}")
            f += L.TIER_WEIGHT[t]*(raw if ok else 0.0)
        print(f"    {name}: final={f:.6f}  " + "  ".join(parts))
