# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a02-critic #11.  The 16,414-dim ridge has strictly BETTER score and cost
correlations than the deployed stack, and a strictly WORSE final score.  Why?

Also: is the project's noise floor a data limit or a compute budget?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, TIER_MULT, tier_result

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(r"C:\Users\PJ05\AppData\Local\Temp\claude\C--portable-skt-LLM1-LLM-ROUTE-0-7000"
             r"\377d7fd8-9983-4bec-bb08-163cc405f7a3\scratchpad")
dv = load_split("dev")
N = len(dv)
ar = np.arange(N)

SETS = {
    "legacy 256-bin only": "a02_ablate_legacy_256-bin_hash-regex_ONLY.npz",
    "ridge 16,414-dim only": "a02_ablate_ridge_16414-dim_ONLY.npz",
    "linear ensemble .9/.1": "a02_ablate_linear_ensemble_legacy_9_p_ridge_1.npz",
    "FULL deployed stack": "a02_ablate_FULL_deployed_E43_stack.npz",
}
SG = np.arange(0.50, 1.601, 0.005)

print("=" * 104)
print("A.  Selection-induced cost bias (the E42 mechanism), measured per prediction set")
print("=" * 104)
print(f"{'prediction set':24s} {'tier':9s} {'best':>7s} {'sf':>6s} {'n_up':>5s} "
      f"{'true/pred cost of the UPGRADED items':>36s} {'... of ALL items':>18s}")
for name, fn in SETS.items():
    P = np.load(CACHE / fn)
    for t in TIERS:
        best = None
        for sf in SG:
            r = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, float(sf))
            if r["passed"] and (best is None or r["score"] > best[0]):
                best = (r["score"], float(sf), r)
        r = best[2]
        sel = r["sel"]
        up = sel > 0
        tp_up = dv.cost[ar[up], sel[up]].sum() / P[f"cost_{t}"][ar[up], sel[up]].sum()
        tp_all = dv.cost.sum() / P[f"cost_{t}"].sum()
        print(f"{name:24s} {t:9s} {best[0]:7.4f} {best[1]:6.3f} {int(up.sum()):5d} "
              f"{tp_up:36.3f} {tp_all:18.3f}")

print("\nSame, restricted to the k1 arm (the expensive one):")
print(f"{'prediction set':24s} {'tier':9s} {'n_k1':>5s} {'true/pred on chosen k1':>23s} "
      f"{'true/pred on ALL k1':>21s} {'selection penalty':>18s}")
for name, fn in SETS.items():
    P = np.load(CACHE / fn)
    for t in ("premium",):
        best = None
        for sf in SG:
            r = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, float(sf))
            if r["passed"] and (best is None or r["score"] > best[0]):
                best = (r["score"], float(sf), r)
        sel = best[2]["sel"]
        m = sel == 2
        if m.sum() == 0:
            print(f"{name:24s} {t:9s} {0:5d}  (never picks k1)")
            continue
        a_ = dv.cost[m, 2].sum() / P[f"cost_{t}"][m, 2].sum()
        b_ = dv.cost[:, 2].sum() / P[f"cost_{t}"][:, 2].sum()
        print(f"{name:24s} {t:9s} {int(m.sum()):5d} {a_:23.3f} {b_:21.3f} {a_/b_:18.3f}")

print("\nB.  Score-prediction spread: does the ridge simply predict a wider (over-confident)")
print("    distribution, so the allocator chases extremes?")
print(f"{'prediction set':24s} {'sd(pred s) l/m/k':>26s} {'sd(pred gain) m-l / k-m':>26s} "
      f"{'corr_s l/m/k':>20s}")
for name, fn in SETS.items():
    P = np.load(CACHE / fn)
    S = P["score_premium"]
    sd = S.std(0)
    g1, g2 = (S[:, 1] - S[:, 0]).std(), (S[:, 2] - S[:, 1]).std()
    cs = [np.corrcoef(P["score_fast"][:, j], dv.score[:, j])[0, 1] for j in range(3)]
    print(f"{name:24s} {'/'.join(f'{x:.3f}' for x in sd):>26s} "
          f"{g1:.3f} / {g2:.3f}{'':>13s} {'/'.join(f'{c:.2f}' for c in cs):>20s}")
print(f"{'TRUE labels':24s} {'/'.join(f'{x:.3f}' for x in dv.score.std(0)):>26s} "
      f"{(dv.score[:,1]-dv.score[:,0]).std():.3f} / {(dv.score[:,2]-dv.score[:,1]).std():.3f}")

print()
print("=" * 104)
print("C.  Is the noise floor a data limit or a compute budget?  sd of a PAIRED EV")
print("    difference vs the number of bootstrap resamples B")
print("=" * 104)
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz")
SAFE = {"fast": .98, "balanced": .89, "premium": .88}


def alloc_batch(S, C, cap):
    B, m, _ = S.shape
    L = C[:, :, 0].sum(1)

    def choose(pen):
        return (S - pen[:, None, None] * C / L[:, None, None]).argmax(2)
    sel = choose(np.zeros(B))
    tot = np.take_along_axis(C, sel[:, :, None], 2)[:, :, 0].sum(1)
    over = tot > cap
    if not over.any():
        return sel
    low, high = np.zeros(B), np.ones(B)
    for _ in range(24):
        s2 = choose(high)
        t2 = np.take_along_axis(C, s2[:, :, None], 2)[:, :, 0].sum(1)
        bad = (t2 > cap) & over
        if not bad.any():
            break
        low = np.where(bad, high, low); high = np.where(bad, high * 2.0, high)
    for _ in range(30):
        mid = (low + high) / 2.0
        cand = choose(mid)
        ct = np.take_along_axis(C, cand[:, :, None], 2)[:, :, 0].sum(1)
        ok = ct <= cap
        high = np.where(ok, mid, high); low = np.where(ok, low, mid)
    sfn = choose(high)
    ct = np.take_along_axis(C, sfn[:, :, None], 2)[:, :, 0].sum(1)
    sel = np.where(over[:, None], sfn, sel); tot = np.where(over, ct, tot)
    give = tot > cap
    if give.any():
        sel = np.where(give[:, None], 0, sel)
    return sel


def ev(idx, cmul):
    tot = 0.0
    for t in TIERS:
        S = P[f"score_{t}"][idx]
        C = (P[f"cost_{t}"] * np.array([1.0, 1.0, cmul])[None, :])[idx]
        ts, tc = dv.score[idx], dv.cost[idx]
        cap = C[:, :, 0].sum(1) * max(1.0, TIER_MULT[t] * SAFE[t])
        sel = alloc_batch(S, C, cap)
        got = np.take_along_axis(ts, sel[:, :, None], 2)[:, :, 0].mean(1)
        cst = np.take_along_axis(tc, sel[:, :, None], 2)[:, :, 0].sum(1)
        ok = (cst / tc[:, :, 0].sum(1)) <= TIER_MULT[t] + 1e-15
        tot += TIER_WEIGHT[t] * float(np.mean(got * ok))
    return tot


print(f"  {'B':>7s} {'sd(paired delta) over 8 seeds':>31s} {'1/sqrt(B) prediction':>21s} {'wall':>7s}")
import time
ref = None
for B in (100, 400, 1600, 6400):
    t0 = time.time()
    ds = []
    for sd in (7, 17, 23, 31, 43, 57, 71, 83):
        idx = np.random.default_rng(sd).integers(0, N, size=(B, N))
        ds.append(ev(idx, 1.01) - ev(idx, 1.00))
    s = float(np.std(ds, ddof=1))
    if ref is None:
        ref = (B, s)
    print(f"  {B:7d} {s:31.5f} {ref[1]*np.sqrt(ref[0]/B):21.5f} {time.time()-t0:6.0f}s")
