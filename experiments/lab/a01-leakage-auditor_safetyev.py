# SPDX-License-Identifier: Apache-2.0
"""A01: re-price the per-tier safety ratio using HONEST (train-only holdout)
predictions.  E39 priced the same curve with CV / in-sample predictions.
880-item bootstrap; 'x1.054' = the organiser's own documented dev->private
cost drift (baselines/README.md: hash-regex premium 3.985 dev -> ~4.2 private)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labdata as L
SP = Path(r"C:/Users/PJ05/AppData/Local/Temp/claude/C--portable-skt-LLM1-LLM-ROUTE-0-7000/377d7fd8-9983-4bec-bb08-163cc405f7a3/scratchpad")
tr, dv = L.load_all()
NB = 400
rng = np.random.default_rng(101)
SAMP = [rng.integers(0, len(dv), size=len(dv)) for _ in range(NB)]

def run(idx, tier, safety, Sm, Cm, cm=1.0):
    ps, pc = Sm[idx], Cm[idx]; mult = L.TIER_MULT[tier]
    lt = pc[:,0].sum(); cap = lt*max(1.0, mult*safety)
    ch = lambda p: (ps - p*pc/lt).argmax(axis=1)
    sel = ch(0.0); tot = pc[np.arange(len(sel)), sel].sum()
    if tot > cap:
        lo, hi = 0.0, 1.0; sel = ch(hi); tot = pc[np.arange(len(sel)), sel].sum()
        while tot > cap and hi < 2**60:
            lo, hi = hi, hi*2.0; sel = ch(hi); tot = pc[np.arange(len(sel)), sel].sum()
        for _ in range(40):
            m=(lo+hi)/2.0; cd=ch(m); ct=pc[np.arange(len(cd)),cd].sum()
            if ct<=cap: hi,sel,tot=m,cd,ct
            else: lo=m
    if tot > cap: sel = np.zeros(len(ps), dtype=int)
    tc, ts = dv.cost[idx], dv.score[idx]
    num = tc[np.arange(len(sel)), sel].copy(); num[sel>0] *= cm
    ratio = num.sum()/tc[:,0].sum(); ok = ratio <= mult+1e-15
    return float(ts[np.arange(len(sel)),sel].mean()), ok

GRIDS = {"fast": np.arange(0.90,1.001,0.02),
         "balanced": np.arange(0.78,0.941,0.02),
         "premium": np.arange(0.72,0.921,0.02)}
for tag, path in (("HONEST train-only holdout (E43 chain)", ROOT/"reports/lab/dev_preds_e43.npz"),
                  ("DEPLOYED train+dev, in-sample on dev", SP/"a01_deployed_nolookup.npz")):
    z = np.load(path, allow_pickle=True)
    S={t:z[f"score_{t}"] for t in L.TIERS}; C={t:z[f"cost_{t}"] for t in L.TIERS}
    print(f"\n===== {tag}  ({NB} bootstraps of 880) =====", flush=True)
    for tier in L.TIERS:
        print(f"  {tier}:  safety   E[tier]  bust%  |  E[tier]x1.054  bust%", flush=True)
        best=bestd=None
        for s in GRIDS[tier]:
            r  = [run(x, tier, float(s), S[tier], C[tier]) for x in SAMP]
            r2 = [run(x, tier, float(s), S[tier], C[tier], 1.054) for x in SAMP]
            ev  = float(np.mean([a if ok else 0.0 for a,ok in r]));  bu  = 100*float(np.mean([not ok for _a,ok in r]))
            ev2 = float(np.mean([a if ok else 0.0 for a,ok in r2])); bu2 = 100*float(np.mean([not ok for _a,ok in r2]))
            if best is None or ev > best[0]: best=(ev,float(s))
            if bestd is None or ev2 > bestd[0]: bestd=(ev2,float(s))
            print(f"          {s:.2f}    {ev:.4f}  {bu:5.1f}  |  {ev2:.4f}       {bu2:5.1f}", flush=True)
        print(f"     -> nominal opt s={best[1]:.2f} EV {best[0]:.4f} | drift opt s={bestd[1]:.2f} EV {bestd[0]:.4f}", flush=True)
