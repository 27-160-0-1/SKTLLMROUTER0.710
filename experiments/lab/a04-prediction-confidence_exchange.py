# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a04 / step 5 -- what an HONEST confidence/accuracy improvement is worth.

(a) honest exchange rate: ensemble two independent honest score heads
    (deployed E43 + my rebuilt ordinal stack) and read Delta(final) per Delta(corr).
    Compare with the label-blend exchange rate quoted in the BRIEF.
(b) shrink the DEPLOYED predictions toward the train family means (uncertainty
    shrinkage with no new information) -- does it buy anything?
(c) cost side: decompose the budget-ratio error into "relative cost ratio"
    error and "light-cost weight" error.
(d) the score-noise -> bust cliff: safety factor needed as a function of how
    much the score head is perturbed.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import similarity

tr, dv = load_split("train"), load_split("dev")
n = len(dv); IDX = np.arange(n)
DEP = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
Z = np.load(Path(__file__).resolve().parents[0] / "a04_ordinal_cache.npz")
Edev = Z["Edev"]
fam = np.array([similarity.classify_family(t) for t in dv.texts])
famtr = np.array([similarity.classify_family(t) for t in tr.texts])
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}


def final_of(mkS, safe=SAFE, C=None):
    tot = 0.0; parts = []
    for t in TIERS:
        cc = DEP[f"cost_{t}"] if C is None else C
        r = tier_result(mkS(t), cc, dv, t, safe[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else '!BUST'}")
    return tot, "  ".join(parts)


def best_safety_final(mkS, C=None):
    tot = 0.0; det = []
    for t in TIERS:
        cc = DEP[f"cost_{t}"] if C is None else C
        best = None
        for sf in np.arange(0.60, 1.201, 0.005):
            r = tier_result(mkS(t), cc, dv, t, float(sf))
            if r["passed"] and (best is None or r["score"] > best[0]):
                best = (r["score"], float(sf), r["ratio"])
        tot += TIER_WEIGHT[t] * best[0]
        det.append(f"{t[:4]}={best[0]:.4f}@{best[1]:.3f}")
    return tot, " ".join(det)


print("=" * 100)
print("STEP 5a  HONEST exchange rate: ensemble of two independent honest score heads")
print("=" * 100)
print(f"{'w(mine)':>8s} {'corr L/M/K':>20s} {'mean corr':>10s} {'final@dep-safety':>17s} {'final@best-safety':>18s}")
rows = []
for w in (0.0, 0.15, 0.3, 0.5, 0.7, 1.0):
    mk = lambda t, w=w: (1 - w) * DEP[f"score_{t}"] + w * Edev
    S = mk("fast")
    cor = [np.corrcoef(S[:, j], dv.score[:, j])[0, 1] for j in range(3)]
    f1, _ = final_of(mk)
    f2, det = best_safety_final(mk)
    rows.append((w, np.mean(cor), f1, f2))
    print(f"{w:8.2f} {'/'.join(f'{c:.3f}' for c in cor):>20s} {np.mean(cor):10.4f} "
          f"{f1:17.4f} {f2:18.4f}   {det}")
c0, f0 = rows[0][1], rows[0][3]
best = max(rows, key=lambda r: r[3])
if abs(best[1] - c0) > 1e-6:
    print(f"\n  honest slope: d(final@best-safety)/d(mean corr) = "
          f"{(best[3]-f0)/(best[1]-c0):+.3f}  (from w=0 -> w={best[0]})")
print("  BRIEF label-blend slope for comparison: lam .00->.10 gives mean corr .443->.590,")
print("  final 0.7063->0.7436  =>  +0.254 final per +1.0 corr.")

print()
print("=" * 100)
print("STEP 5b  pure shrinkage of the DEPLOYED predictions toward the train family means")
print("  (no new information -- only variance reduction)")
print("=" * 100)
fmean = np.zeros((n, 3))
for f in set(fam):
    mm = fam == f
    fmean[mm] = tr.score[famtr == f].mean(0) if (famtr == f).sum() >= 8 else tr.score.mean(0)
print(f"{'k':>6s} {'corr L/M/K':>20s} {'rmse mean':>10s} {'final@dep':>10s} {'final@best-safety':>18s}")
for k in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
    mk = lambda t, k=k: np.clip((1 - k) * DEP[f"score_{t}"] + k * fmean, 0, 1)
    S = mk("fast")
    cor = [np.corrcoef(S[:, j], dv.score[:, j])[0, 1] for j in range(3)]
    rm = np.mean([np.sqrt(((S[:, j] - dv.score[:, j]) ** 2).mean()) for j in range(3)])
    f1, p1 = final_of(mk); f2, det = best_safety_final(mk)
    print(f"{k:6.2f} {'/'.join(f'{c:.3f}' for c in cor):>20s} {rm:10.4f} {f1:10.4f} {f2:18.4f}   {det}")

print()
print("=" * 100)
print("STEP 5c  budget-ratio error decomposition")
print("  ratio = sum_i c[i,sel_i] / sum_i c[i,0] = sum_i w_i * r_i,   w_i = c[i,0]/sum c[.,0],")
print("  r_i = c[i,sel_i]/c[i,0].   Which factor does the router get wrong?")
print("=" * 100)
for t in TIERS:
    pc = DEP[f"cost_{t}"]
    r = tier_result(DEP[f"score_{t}"], pc, dv, t, SAFE[t])
    sel = r["sel"]
    w_t = dv.cost[:, 0] / dv.cost[:, 0].sum()
    w_p = pc[:, 0] / pc[:, 0].sum()
    r_t = dv.cost[IDX, sel] / dv.cost[:, 0]
    r_p = pc[IDX, sel] / pc[:, 0]
    true_ratio = (w_t * r_t).sum()
    pred_ratio = (w_p * r_p).sum()
    mixed_wt = (w_t * r_p).sum()      # true weights, predicted relative ratios
    mixed_rt = (w_p * r_t).sum()      # predicted weights, true relative ratios
    print(f"  {t:9s} pred_ratio={pred_ratio:.4f} true_ratio={true_ratio:.4f} gap={true_ratio-pred_ratio:+.4f}")
    print(f"            true-w  x pred-r = {mixed_wt:.4f} (isolates relative-ratio error {mixed_wt-pred_ratio:+.4f})")
    print(f"            pred-w  x true-r = {mixed_rt:.4f} (isolates weight error        {mixed_rt-pred_ratio:+.4f})")
    for j in (1, 2):
        m = sel == j
        if m.sum():
            print(f"            selected as model {j}: n={m.sum():4d} mean pred r={r_p[m].mean():7.3f} "
                  f"true r={r_t[m].mean():7.3f}  cost-weighted pred={np.average(r_p[m],weights=w_p[m]):7.3f} "
                  f"true={np.average(r_t[m],weights=w_t[m]):7.3f}")
    # selection-induced bias: relative-ratio error on selected vs all items
    for j in (1, 2):
        allr_p = pc[:, j] / pc[:, 0]; allr_t = dv.cost[:, j] / dv.cost[:, 0]
        le = np.log(allr_p) - np.log(allr_t)
        m = sel == j
        if m.sum() >= 5:
            print(f"            log(pred r / true r) for model {j}: all items mean={le.mean():+.3f} "
                  f"| selected items mean={le[m].mean():+.3f} (selection bias {le[m].mean()-le.mean():+.3f})")

print()
print("=" * 100)
print("STEP 5d  score-noise -> bust cliff: safety needed to survive a perturbed score head")
print("=" * 100)
rng = np.random.default_rng(1)
sig = np.array([np.sqrt(((DEP['score_fast'][:, j] - dv.score[:, j]) ** 2).mean()) for j in range(3)])
print(f"{'tier':9s} {'noise x':>8s} {'bust@dep-safety':>16s} {'mean score':>11s} {'safety for 0 bust in 200':>26s}")
for t in TIERS:
    for scale in (0.0, 0.1, 0.25, 0.5, 1.0):
        S = [np.clip(DEP[f"score_{t}"] + rng.normal(0, 1, (n, 3)) * sig * scale, 0, 1) for _ in range(200)]
        busts = 0; scs = []
        for s_ in S:
            r = tier_result(s_, DEP[f"cost_{t}"], dv, t, SAFE[t])
            busts += (not r["passed"]); scs.append(r["score"] if r["passed"] else 0.0)
        need = None
        for sf in np.arange(SAFE[t], 0.5, -0.01):
            b = sum(not tier_result(s_, DEP[f"cost_{t}"], dv, t, float(sf))["passed"] for s_ in S)
            if b == 0:
                need = float(sf); break
        print(f"{t:9s} {scale:8.2f} {busts:12d}/200 {np.mean(scs):11.4f} "
              f"{('%.2f' % need) if need else '  n/a':>26s}")
