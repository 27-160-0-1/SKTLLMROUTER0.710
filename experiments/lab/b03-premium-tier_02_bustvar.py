# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b03 step 2: decompose the variance of the PREMIUM budget ratio.

R = N/D = 1 + U/D,   U = sum_i (c_{i,pick_i} - c_{i,0}) >= 0,  D = sum_i c_{i,0}.

Three candidate sources:
  (a) how many k1 items are selected      -> law of total variance conditional on n_k1
  (b) the heavy tail of k1 output length  -> replace each item's true k1 cost by the
      family geometric-mean k1 cost (same expected level, tail removed)
  (c) light-baseline denominator noise    -> replace D by its pool expectation
All measured on the honest OOF pool (1,760 train rows), 880-item bootstrap.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
ci = cv["idx"]; ts = lab.true_s[ci]; tc = lab.true_c[ci]; m = len(ci)
fam = lab.fam_arr[ci]
TIER = "premium"; MULT = 4.0

ps, pc = lab.compose(cv, DEPLOYED_CFG, TIER)

# de-tailed cost matrix: per (family, model) replace c by exp(mean log c) of that cell
tc_flat = tc.copy()
for f in sorted(set(fam)):
    sel = fam == f
    for j in range(3):
        tc_flat[sel, j] = np.exp(np.log(tc[sel, j]).mean())

def analyse(safety, nboot=1200, seeds=(7, 17, 23), label=""):
    rows = []
    for s in seeds:
        smp = np.asarray(lab.samples_for(m, s, nboot, 880))
        picks = P.exact_allocate(ps[smp], pc[smp], MULT, safety)
        r = np.arange(len(smp))[:, None]
        Ct = np.take_along_axis(tc[smp], picks[:, :, None], axis=2)[:, :, 0]
        C0 = tc[smp][:, :, 0]
        Cf = np.take_along_axis(tc_flat[smp], picks[:, :, None], axis=2)[:, :, 0]
        C0f = tc_flat[smp][:, :, 0]
        U = (Ct - C0).sum(axis=1); D = C0.sum(axis=1)
        Uf = (Cf - C0f).sum(axis=1); Df = C0f.sum(axis=1)
        Sc = np.take_along_axis(ts[smp], picks[:, :, None], axis=2)[:, :, 0].mean(axis=1)
        nk1 = (picks == 2).sum(axis=1)
        rows.append(dict(U=U, D=D, Uf=Uf, Df=Df, nk1=nk1, sc=Sc))
    out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    U, D, Uf, Df, nk1, sc = out["U"], out["D"], out["Uf"], out["Df"], out["nk1"], out["sc"]
    R = 1 + U / D
    Dbar = D.mean()
    variants = {
        "full            R = 1+U/D": R,
        "(c) D fixed     1+U/Dbar": 1 + U / Dbar,
        "(b) k1 tail off 1+Uf/D  ": 1 + Uf / D,
        "(b+c) both off  1+Uf/Dbar": 1 + Uf / Dbar,
    }
    print(f"\n=== {label} safety={safety:.3f}  n={len(R)} ===")
    print(f"  mean R={R.mean():.4f} sd={R.std():.4f}  bust%={100*np.mean(R>MULT):.2f}"
          f"  score|pass={sc[R<=MULT].mean():.4f}  EV={np.where(R>MULT,0,sc).mean():.4f}")
    for k, v in variants.items():
        print(f"  {k}  mean={v.mean():.4f} sd={v.std():.4f}  var share={v.var()/R.var():6.3f}"
              f"  bust%={100*np.mean(v>MULT):5.2f}")
    # (a) law of total variance on n_k1
    qs = np.quantile(nk1, np.linspace(0, 1, 9))
    bins = np.clip(np.searchsorted(np.unique(qs), nk1, side="right") - 1, 0, None)
    within, betw = 0.0, []
    for b in np.unique(bins):
        s2 = R[bins == b]
        within += s2.var() * len(s2) / len(R)
        betw.append((s2.mean(), len(s2)))
    bv = np.average([x[0] for x in betw], weights=[x[1] for x in betw])
    between = np.average([(x[0] - bv) ** 2 for x in betw], weights=[x[1] for x in betw])
    print(f"  (a) n_k1: mean={nk1.mean():.1f} sd={nk1.std():.1f}  "
          f"var(R) between-n_k1={between/R.var():.3f}  within={within/R.var():.3f}")
    print(f"      corr(n_k1,R)={np.corrcoef(nk1,R)[0,1]:+.3f}  corr(D,R)={np.corrcoef(D,R)[0,1]:+.3f}"
          f"  corr(U,D)={np.corrcoef(U,D)[0,1]:+.3f}")
    print(f"      sd(log U)={np.log(U).std():.4f} sd(log D)={np.log(D).std():.4f}"
          f" sd(log R)={np.log(R).std():.4f}")
    return out

for sfty in (0.84, 0.90, 0.95):
    analyse(sfty, label="premium legoof")
