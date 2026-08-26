# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a05 step 6: robustness of the label-noise attribution + explicit
missed-upgrade / wasted-upgrade tables + bootstrap on the one rule that moved."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.similarity import classify_family

dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
Cc = np.load(ROOT / "experiments/lab/a05-selection-failure_cache.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
N = len(dv); IDX = np.arange(N); FAM = Cc["fam"]
L = dv.cost[:, 0].sum()


def eb(prior_scale):
    n = dv.ngen.astype(int); k = np.rint(dv.score * n).astype(int)
    A = np.ones_like(dv.score); B = np.ones_like(dv.score)
    for f_ in set(FAM.tolist()):
        msk = FAM == f_
        for j in range(3):
            x = dv.score[msk, j]; mu = float(x.mean()); var = float(x.var())
            nn = float(n[msk, j].mean())
            if not (0 < mu < 1):
                continue
            vp = (var - (mu - mu * mu) / nn) / (1 - 1.0 / nn) if nn > 1 else var
            vp = float(np.clip(vp, 1e-4, mu * (1 - mu) - 1e-4))
            c = (mu * (1 - mu) / vp - 1) * prior_scale
            A[msk, j] = max(c * mu, 0.05); B[msk, j] = max(c * (1 - mu), 0.05)
    return (A + k) / (A + B + n)


print("=== (1) sensitivity of the 'recoverable' share to the EB prior strength ===")
print(f"  {'prior_scale':11s} {'shrink sd':>10s} {'gap':>8s} {'recoverable':>12s} {'noise-only':>11s} {'rec %':>7s}")
for psc in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    PH = eb(psc)
    gap = rec = 0.0
    for t in TIERS:
        sd = Cc[f"sel_d_{t}"]; so = Cc[f"sel_o_{t}"]
        gap += TIER_WEIGHT[t] * (dv.score[IDX, so] - dv.score[IDX, sd]).mean()
        rec += TIER_WEIGHT[t] * (PH[IDX, so] - PH[IDX, sd]).mean()
    print(f"  {psc:11.2f} {PH.std(0).mean():10.3f} {gap:8.4f} {rec:12.4f} {gap-rec:11.4f} "
          f"{100*rec/gap:7.1f}%")

print("\n=== (2) simulation cross-check: draw fresh labels from the EB posterior mean ===")
PH = eb(1.0)
rng = np.random.default_rng(7)
inn, out = [], []
for _ in range(40):
    s1 = rng.binomial(dv.ngen.astype(int), PH) / dv.ngen
    a = b = 0.0
    for t in TIERS:
        r = tier_result(s1, dv.cost, dv, t, 1.0)      # oracle on the noisy draw
        a += TIER_WEIGHT[t] * (s1[IDX, r["sel"]].mean() if r["passed"] else 0.0)
        b += TIER_WEIGHT[t] * (PH[IDX, r["sel"]].mean() if r["passed"] else 0.0)
    inn.append(a); out.append(b)
print(f"  oracle scored on its own noisy draw  = {np.mean(inn):.4f} +- {np.std(inn):.4f}")
print(f"  same allocation scored on latent p   = {np.mean(out):.4f} +- {np.std(out):.4f}")
print(f"  => simulated oracle noise inflation  = {np.mean(inn)-np.mean(out):.4f}")
print(f"  (measured directly on the real data: 0.8034 realised vs "
      f"{sum(TIER_WEIGHT[t]*PH[IDX, Cc[f'sel_o_{t}']].mean() for t in TIERS):.4f} EB = "
      f"{0.8034 - sum(TIER_WEIGHT[t]*PH[IDX, Cc[f'sel_o_{t}']].mean() for t in TIERS):.4f})")

print("\n=== (3) missed upgrades and wasted upgrades, counted and costed ===")
sL, sM, sK = dv.score[:, 0], dv.score[:, 1], dv.score[:, 2]
cases = [
    ("MISSED  sL=0, sM=1, chose light", (sL <= 0) & (sM >= 1), 1),
    ("MISSED  sL=0, sK=1, chose light/mid", (sL <= 0) & (sK >= 1), 2),
    ("MISSED  sM>sL, chose light", sM > sL, 1),
    ("MISSED  sK>sM, chose light/mid", sK > sM, 2),
    ("WASTED  sL=1, upgraded anyway", sL >= 1, None),
    ("WASTED  sM<=sL, chose mid", sM <= sL, None),
    ("WASTED  sK<=sL, chose k1", sK <= sL, None),
    ("HOPELESS sL=sM=sK=0, upgraded", (sL <= 0) & (sM <= 0) & (sK <= 0), None),
]
for t in TIERS:
    sd = Cc[f"sel_d_{t}"]
    print(f"  --- tier {t} ---")
    for nm, msk, need in cases:
        if nm.startswith("MISSED"):
            m = msk & (sd < need)
            lost = (dv.score[m, need] - dv.score[m, sd[m]]).sum() / N
            price = (dv.cost[m, need] - dv.cost[m, sd[m]]).sum() / L
            ebl = (Cc["phat"][m, need] - Cc["phat"][m, sd[m]]).sum() / N
            print(f"    {nm:36s} n={m.sum():4d}  score forgone {lost:+.4f} (EB {ebl:+.4f})  "
                  f"would cost {price:6.3f}L  -> {lost/max(price,1e-9):7.4f} per L")
        else:
            if nm.startswith("WASTED  sM"):
                m = msk & (sd == 1)
            elif nm.startswith("WASTED  sK"):
                m = msk & (sd == 2)
            else:
                m = msk & (sd > 0)
            spent = (dv.cost[m, sd[m]] - dv.cost[m, 0]).sum() / L
            got = (dv.score[m, sd[m]] - dv.score[m, 0]).sum() / N
            ebg = (Cc["phat"][m, sd[m]] - Cc["phat"][m, 0]).sum() / N
            print(f"    {nm:36s} n={m.sum():4d}  score bought  {got:+.4f} (EB {ebg:+.4f})  "
                  f"spent      {spent:6.3f}L  -> {got/max(spent,1e-9):7.4f} per L")

print("\n=== (4) bootstrap of R3 (ban mid when pred gain < 0.02) vs baseline ===")
def alloc_final(mk_s, idxs):
    """re-score on a bootstrap resample of the dev items (allocation re-run on the resample)"""
    class Sub:  # minimal Split-like view
        pass
    sub = Sub()
    sub.score = dv.score[idxs]; sub.cost = dv.cost[idxs]
    tot = 0.0
    for t in TIERS:
        ps = mk_s(t)[idxs]; pcst = P[f"cost_{t}"][idxs]
        Lt = pcst[:, 0].sum(); cap = Lt * max(1.0, TIER_MULT[t] * SAFE[t])
        lo, hi = 0.0, 1.0
        def ch(pen):
            return (ps - pen * pcst / Lt).argmax(1)
        sel = ch(0.0); tt = pcst[np.arange(len(idxs)), sel].sum()
        if tt > cap:
            sel = ch(hi); tt = pcst[np.arange(len(idxs)), sel].sum()
            while tt > cap and hi < 2 ** 40:
                lo, hi = hi, hi * 2; sel = ch(hi); tt = pcst[np.arange(len(idxs)), sel].sum()
            for _ in range(40):
                mid = (lo + hi) / 2; cand = ch(mid)
                ct = pcst[np.arange(len(idxs)), cand].sum()
                if ct <= cap: hi, sel, tt = mid, cand, ct
                else: lo = mid
        rr = sub.cost[np.arange(len(idxs)), sel].sum() / sub.cost[:, 0].sum()
        ok = rr <= TIER_MULT[t] + 1e-15
        tot += TIER_WEIGHT[t] * (sub.score[np.arange(len(idxs)), sel].mean() if ok else 0.0)
    return tot

def base_s(t):
    return P[f"score_{t}"]
def r3_s(t):
    S = P[f"score_{t}"].copy()
    S[(S[:, 1] - S[:, 0]) < 0.02, 1] = -1e9
    return S

rng = np.random.default_rng(7)
d = []
for _ in range(400):
    idxs = rng.integers(0, N, N)
    d.append(alloc_final(r3_s, idxs) - alloc_final(base_s, idxs))
d = np.array(d)
print(f"  bootstrap 400x880: mean delta {d.mean():+.4f}  sd {d.std():.4f}  "
      f"P(delta>0)={np.mean(d>0):.3f}  2.5/97.5 pct {np.percentile(d,2.5):+.4f}/{np.percentile(d,97.5):+.4f}")
