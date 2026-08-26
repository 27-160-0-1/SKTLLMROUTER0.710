# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""A01: bootstrap distribution of the realised budget ratio (premium/balanced),
honest train-only predictions vs deployed in-sample predictions."""
import sys
import os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labdata as L
# 이 감사 스크립트가 읽는 중간 산출물의 위치. 원래는 개발 기계의 스크래치 경로가
# 하드코딩되어 있었고, 그 경로가 사용자 이름을 노출했다. 환경변수로 받고
# 저장소 안의 재생성 가능한 위치를 기본값으로 둔다.
SP = Path(os.environ.get("A01_SCRATCH", "experiments/lab"))
tr, dv = L.load_all()
sets = {
    "honest train-only holdout": np.load(ROOT/"reports/lab/dev_preds_e43.npz", allow_pickle=True),
    "DEPLOYED (train+dev, in-sample)": np.load(SP/"a01_deployed_nolookup.npz", allow_pickle=True),
}
def run(idx, tier, safety, Sm, Cm):
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
    tc = dv.cost[idx]
    return tc[np.arange(len(sel)), sel].sum()/tc[:,0].sum()
rng = np.random.default_rng(11); n=len(dv); NB=2000
SA = dict(fast=.98, balanced=.87, premium=.85)
for name, z in sets.items():
    S={t:z[f"score_{t}"] for t in L.TIERS}; C={t:z[f"cost_{t}"] for t in L.TIERS}
    print(f"\n== {name}, safety .98/.87/.85 ==")
    for t in L.TIERS:
        r = np.array([run(rng.integers(0,n,size=n), t, SA[t], S[t], C[t]) for _ in range(NB)])
        p = np.array([run(np.arange(n), t, SA[t], S[t], C[t])])
        print(f"  {t:9s} limit={L.TIER_MULT[t]}  point={p[0]:.4f}  boot mean={r.mean():.4f} sd={r.std():.4f} "
              f"p50={np.quantile(r,.5):.4f} p90={np.quantile(r,.9):.4f} p99={np.quantile(r,.99):.4f} "
              f"bust%={100*(r>L.TIER_MULT[t]).mean():.2f}")
# how heavy is the true k1 cost tail?
c = dv.cost[:,2]/dv.cost[:,0].mean()
print(f"\ndev true k1 cost / mean light cost: mean={c.mean():.1f} p50={np.median(c):.1f} "
      f"p95={np.quantile(c,.95):.1f} p99={np.quantile(c,.99):.1f} max={c.max():.1f}")
print(f"top-10 k1 items carry {100*np.sort(dv.cost[:,2])[-10:].sum()/dv.cost[:,2].sum():.1f}% of total k1 cost")
