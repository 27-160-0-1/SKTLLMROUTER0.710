# SPDX-License-Identifier: Apache-2.0
"""a12 P10 -- stability of the conditional constants, and a proper truncation census."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from labdata import load_split  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

tr, dv = load_split("train"), load_split("dev")
ftr = np.array([classify_family(t) for t in tr.texts])
fdv = np.array([classify_family(t) for t in dv.texts])
fams = sorted(set(ftr) | set(fdv))


def hdr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


hdr("P10.1  stability of E[s_k1 | s_mid] = A + B*s_mid, train vs dev")
print(f"  {'family':16s} | {'A_tr':>7s} {'A_dv':>7s} {'dA':>7s} | {'B_tr':>7s} {'B_dv':>7s} {'dB':>7s} "
      f"| {'pred@s=0':>9s} {'true@s=0':>9s} | {'pred@s=1':>9s} {'true@s=1':>9s}")
for f in fams:
    mt, md = ftr == f, fdv == f
    wt = np.linalg.lstsq(np.column_stack([np.ones(mt.sum()), tr.score[mt, 1]]), tr.score[mt, 2], rcond=None)[0]
    wd = np.linalg.lstsq(np.column_stack([np.ones(md.sum()), dv.score[md, 1]]), dv.score[md, 2], rcond=None)[0]
    z0 = dv.score[md][dv.score[md, 1] == 0, 2]
    z1 = dv.score[md][dv.score[md, 1] == 1, 2]
    print(f"  {f:16s} | {wt[0]:+7.3f} {wd[0]:+7.3f} {wd[0]-wt[0]:+7.3f} | {wt[1]:+7.3f} {wd[1]:+7.3f} "
          f"{wd[1]-wt[1]:+7.3f} | {wt[0]:9.3f} {(z0.mean() if len(z0) else np.nan):9.3f} | "
          f"{wt[0]+wt[1]:9.3f} {(z1.mean() if len(z1) else np.nan):9.3f}")

hdr("P10.2  truncation census -- the 32,768-token in+out cap")
for sp, fm in ((tr, ftr), (dv, fdv)):
    ig = sp.itok[:, 2] / sp.ngen[:, 2]
    og = sp.otok[:, 2] / sp.ngen[:, 2]
    g = sp.ngen[:, 2]
    cap = 32768.0
    L = cap - ig                      # per-generation output budget
    full = og >= L - 2                # ALL generations truncated
    some = (sp.otok[:, 2] >= L - 2) & ~full   # at least one generation could be truncated
    print(f"\n  {sp.name}: n={len(sp)}")
    print(f"    items where EVERY generation hit the cap : {int(full.sum()):4d} "
          f"({full.mean()*100:.2f}%)  mean s_k1={sp.score[full,2].mean() if full.sum() else float('nan'):.3f}"
          f"  s_mid={sp.score[full,1].mean() if full.sum() else float('nan'):.3f}")
    print(f"    items where >=1 generation MAY have hit it: {int(some.sum()):4d} "
          f"({some.mean()*100:.2f}%)  mean s_k1={sp.score[some,2].mean() if some.sum() else float('nan'):.3f}")
    tot = sp.cost[:, 2].sum()
    print(f"    share of all k1 cost: full-truncation {sp.cost[full,2].sum()/tot*100:5.2f}% , "
          f"possible-truncation {sp.cost[some,2].sum()/tot*100:5.2f}%")
    print(f"    share of the LIGHT baseline they would consume: full {sp.cost[full,2].sum()/sp.cost[:,0].sum():.3f} "
          f"light-units = {sp.cost[full,2].sum()/sp.cost[:,0].sum()/4*100:.1f}% of the premium budget")
    # score of the near-cap band
    band = (og >= 0.5 * L) & ~full
    print(f"    'deep think' band (>=50% of the output budget, not capped): n={int(band.sum())} "
          f"s_k1={sp.score[band,2].mean() if band.sum() else float('nan'):.3f} "
          f"s_mid={sp.score[band,1].mean() if band.sum() else float('nan'):.3f} "
          f"cost={sp.cost[band,2].sum()/sp.cost[:,0].sum():.3f} light-units")

hdr("P10.3  the exact truncated items")
for sp, fm in ((tr, ftr), (dv, fdv)):
    ig = sp.itok[:, 2] / sp.ngen[:, 2]
    og = sp.otok[:, 2] / sp.ngen[:, 2]
    full = og >= (32768.0 - ig) - 2
    for i in np.where(full)[0]:
        print(f"  {sp.name:5s} {sp.episode_ids[i]:>12s} {fm[i]:15s} ngen={int(sp.ngen[i,2])} "
              f"in/gen={ig[i]:7.0f} out/gen={og[i]:8.0f} in+out={ig[i]+og[i]:8.0f} "
              f"s=[{sp.score[i,0]:.2f},{sp.score[i,1]:.2f},{sp.score[i,2]:.2f}] "
              f"cost_k1={sp.cost[i,2]/sp.cost[:,0].sum()*100:5.2f}% of light baseline")

hdr("P10.4  output-length ceiling by family: how close does k1 get to its budget?")
for sp, fm in ((dv, fdv),):
    ig = sp.itok[:, 2] / sp.ngen[:, 2]
    og = sp.otok[:, 2] / sp.ngen[:, 2]
    u = og / (32768.0 - ig)
    print(f"  {'family':16s} {'n':>4s} {'med in/gen':>11s} {'out budget':>11s} {'med use':>8s} "
          f"{'p90 use':>8s} {'p99 use':>8s} {'frac>0.9':>9s}")
    for f in fams:
        m = fm == f
        print(f"  {f:16s} {m.sum():4d} {np.median(ig[m]):11.0f} {np.median(32768-ig[m]):11.0f} "
              f"{np.median(u[m]):8.3f} {np.percentile(u[m],90):8.3f} {np.percentile(u[m],99):8.3f} "
              f"{np.mean(u[m]>0.9):9.3f}")
