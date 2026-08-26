# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a09: cost re-transform / calibration diagnostics on dev."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_MULT, TIER_WEIGHT, tier_result

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}   # E43 deployed
n = len(dv)

print("=== do the per-tier cost predictions differ? ===")
for a,b in (("fast","balanced"),("fast","premium")):
    d = np.abs(P[f"cost_{a}"]-P[f"cost_{b}"]).max()
    ds = np.abs(P[f"score_{a}"]-P[f"score_{b}"]).max()
    print(f"  max|cost_{a}-cost_{b}| = {d:.3e}   max|score diff| = {ds:.3e}")

print()
print("=== per-model log-residual structure (tier=fast costs) ===")
C = P["cost_fast"]; T = dv.cost
r = np.log(T) - np.log(C)          # true = pred * exp(r)
for j,m in enumerate(MODEL_IDS):
    rj = r[:,j]
    print(f"  {m:11s} sum_ratio(pred/true)={C[:,j].sum()/T[:,j].sum():.4f} "
          f"mean r={rj.mean():+.4f} sd={rj.std():.4f} "
          f"Duan mean(exp r)={np.exp(rj).mean():.4f} lognormal exp(m+s^2/2)={np.exp(rj.mean()+rj.var()/2):.4f} "
          f"skew={((rj-rj.mean())**3).mean()/rj.std()**3:+.2f}")

# family split
from ossp_router.similarity import classify_family
fam = np.array([classify_family(t) for t in dv.texts])
fams = sorted(set(fam))
print()
print("=== Duan factor mean(exp r) per family x model (dev, oracle) ===")
print(f"  {'family':16s} {'n':>4s} " + " ".join(f"{m:>12s}" for m in MODEL_IDS))
for f in fams:
    msk = fam==f
    vals = [np.exp(r[msk,j]).mean() for j in range(3)]
    print(f"  {f:16s} {msk.sum():4d} " + " ".join(f"{v:12.3f}" for v in vals))

print()
print("=== realised-vs-predicted totals at the deployed operating point ===")
for t in TIERS:
    res = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, SAFE[t])
    sel = res["sel"]; idx = np.arange(n)
    pc = P[f"cost_{t}"]
    pred_sel = pc[idx,sel].sum(); true_sel = dv.cost[idx,sel].sum()
    pred_lt  = pc[:,0].sum();     true_lt  = dv.cost[:,0].sum()
    R_sel = true_sel/pred_sel; R_lt = true_lt/pred_lt
    cnt = np.bincount(sel, minlength=3)
    print(f"  {t:9s} sel counts {cnt}  R_sel={R_sel:.4f} R_light={R_lt:.4f} "
          f"H=R_light/R_sel={R_lt/R_sel:.4f}  safety={SAFE[t]}  realised ratio={res['ratio']:.4f}/{TIER_MULT[t]}")

print()
print("=== score-side: monotonicity of predictions vs truth ===")
for t in TIERS:
    S = P[f"score_{t}"]
    v01 = (S[:,1]<S[:,0]).mean(); v12 = (S[:,2]<S[:,1]).mean()
    print(f"  pred {t:9s}: P(s_mid<s_light)={v01:.3f} P(s_k1<s_mid)={v12:.3f} "
          f"min={S.min():.3f} max={S.max():.3f}")
T_ = dv.score
print(f"  TRUE          : P(s_mid<s_light)={(T_[:,1]<T_[:,0]).mean():.3f} "
      f"P(s_k1<s_mid)={(T_[:,2]<T_[:,1]).mean():.3f}")
print(f"  TRUE means {np.round(T_.mean(0),4)}   pred(fast) means {np.round(P['score_fast'].mean(0),4)}")
print(f"  TRUE  sd   {np.round(T_.std(0),4)}   pred(fast) sd    {np.round(P['score_fast'].std(0),4)}")
