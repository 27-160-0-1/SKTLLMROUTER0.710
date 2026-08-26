# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""A01: weighted E[final] of candidate safety sets under honest predictions."""
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
def run(idx, tier, s, Sm, Cm, cm=1.0):
    ps, pc = Sm[idx], Cm[idx]; mult = L.TIER_MULT[tier]
    lt = pc[:,0].sum(); cap = lt*max(1.0, mult*s)
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
    num = tc[np.arange(len(sel)), sel].copy(); num[sel>0]*=cm
    ratio = num.sum()/tc[:,0].sum()
    return float(ts[np.arange(len(sel)),sel].mean()), ratio <= mult+1e-15
CANDS = [("deployed E43   .98/.87/.85", dict(fast=.98,balanced=.87,premium=.85)),
         ("holdout script .98/.89/.88", dict(fast=.98,balanced=.89,premium=.88)),
         ("E39 min-regret .985/.875/.81", dict(fast=.985,balanced=.875,premium=.81)),
         ("A01 EV-optimal .96/.80/.74", dict(fast=.96,balanced=.80,premium=.74)),
         ("A01 conservative .95/.80/.72", dict(fast=.95,balanced=.80,premium=.72))]
z = np.load(ROOT/"reports/lab/dev_preds_e43.npz", allow_pickle=True)
S={t:z[f"score_{t}"] for t in L.TIERS}; C={t:z[f"cost_{t}"] for t in L.TIERS}
rng = np.random.default_rng(2026); n=len(dv); NB=2000
SAMP=[rng.integers(0,n,size=n) for _ in range(NB)]
print("honest train-only holdout predictions, 2000 bootstraps of 880")
print(f"{'safety set':30s} {'E[final]':>9s} {'E[final]x1.054':>15s}   {'point est. on dev 880':>22s}")
for nm, sa in CANDS:
    for cm, col in ((1.0,'a'),(1.054,'b')):
        tot=0.0
        for x in SAMP:
            f=0.0
            for t in L.TIERS:
                r,ok = run(x,t,sa[t],S[t],C[t],cm)
                f += L.TIER_WEIGHT[t]*(r if ok else 0.0)
            tot+=f
        if col=='a': ev=tot/NB
        else: ev2=tot/NB
    pt=0.0; det=[]
    for t in L.TIERS:
        r,ok = run(np.arange(n),t,sa[t],S[t],C[t]); pt += L.TIER_WEIGHT[t]*(r if ok else 0.0)
    print(f"{nm:30s} {ev:9.4f} {ev2:15.4f}   {pt:22.4f}")
