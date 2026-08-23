# SPDX-License-Identifier: Apache-2.0
"""a11 step 2 - what is the safety scalar actually paying for?

Decompose the realised ratio into
    R_true = (N_true/N_hat) / (D_true/D_hat) * R_hat
so the multiplicative bias ratio b = b_N / b_D is exactly the factor the
safety scalar has to absorb.  Then measure the residual moments per model.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, MODEL_IDS, tier_result

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
n = len(dv); idx = np.arange(n)

print("=== A. log-residual moments per model (dev, held-out predictions) ===")
print(f"{'tier':9s} {'model':12s} {'mu':>7s} {'sd':>6s} {'k=E[c]/chat':>12s} "
      f"{'sum_true/sum_pred':>18s} {'corr(e_j,e_0)':>14s}")
for t in TIERS:
    C = P[f"cost_{t}"]
    E = np.log(dv.cost) - np.log(C)
    for j, m in enumerate(MODEL_IDS):
        k_mom = np.exp(E[:, j].mean() + 0.5 * E[:, j].var())
        print(f"{t:9s} {m:12s} {E[:,j].mean():+7.3f} {E[:,j].std():6.3f} {k_mom:12.3f} "
              f"{dv.cost[:,j].sum()/C[:,j].sum():18.3f} "
              f"{np.corrcoef(E[:,j], E[:,0])[0,1]:14.3f}")

print("\n=== B. bias decomposition of the deployed selection ===")
print(f"{'tier':9s} {'R_hat':>7s} {'b_N':>6s} {'b_D':>6s} {'b_N/b_D':>8s} "
      f"{'R_true':>7s} {'mult':>5s} {'s_dep':>6s} {'s_needed':>9s} {'margin':>7s}")
for t in TIERS:
    C = P[f"cost_{t}"]
    r = tier_result(P[f"score_{t}"], C, dv, t, SAFE[t])
    sel = r["sel"]
    Nh = C[idx, sel].sum(); Dh = C[:, 0].sum()
    Nt = dv.cost[idx, sel].sum(); Dt = dv.cost[:, 0].sum()
    Rh, Rt = Nh / Dh, Nt / Dt
    bN, bD = Nt / Nh, Dt / Dh
    s_need = TIER_MULT[t] / (Rh * bN / bD) * (Rh / TIER_MULT[t])  # s at which R_true == mult
    s_need = SAFE[t] * TIER_MULT[t] / Rt
    print(f"{t:9s} {Rh:7.4f} {bN:6.3f} {bD:6.3f} {bN/bD:8.4f} {Rt:7.4f} "
          f"{TIER_MULT[t]:5.2f} {SAFE[t]:6.3f} {s_need:9.3f} {s_need/SAFE[t]-1:+6.1%}")
print("  s_needed = the safety scalar at which the dev realised ratio would exactly")
print("  hit the cap (linearised); margin = how much insurance the deployed value holds.")

print("\n=== C. per-model calibration factors k_j = sum_true/sum_pred (per tier) ===")
K = {}
for t in TIERS:
    C = P[f"cost_{t}"]
    K[t] = dv.cost.sum(0) / C.sum(0)
    print(f"  {t:9s} k = {np.round(K[t], 4)}   k_j/k_0 = {np.round(K[t]/K[t][0], 4)}")
print("  k_j/k_0 > 1 means the allocator under-prices model j RELATIVE to light,")
print("  i.e. the part of the bias that the safety scalar must absorb.")

print("\n=== D. what does the safety scalar become after per-model calibration? ===")
for t in TIERS:
    C = P[f"cost_{t}"] * K[t][None, :]
    for s in (1.0, 0.98, 0.95, 0.92, 0.90, 0.87, 0.85):
        r = tier_result(P[f"score_{t}"], C, dv, t, s)
        flag = "" if r["passed"] else "  BUST"
        print(f"  {t:9s} calib s={s:.2f}  score={r['score']:.4f} ratio={r['ratio']:.4f}{flag}")
    print()
