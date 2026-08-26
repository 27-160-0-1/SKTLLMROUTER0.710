# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a04 / step 6 -- the cost-side counterpart of the confidence question.

(a) exact additive decomposition of the budget-ratio error
(b) selection-induced cost bias per model, and what a per-model multiplier on
    the predicted cost buys (this is NOT the global safety scalar: it changes
    the relative attractiveness of the models, safety only moves the cap)
(c) fast-tier fragility: true-ratio distribution under score jitter
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
ROOT = Path(__file__).resolve().parents[2]

dv = load_split("dev")
n = len(dv); IDX = np.arange(n)
DEP = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}

print("=" * 100)
print("STEP 6a  exact decomposition  ratio_true - ratio_pred = SUM (w_t-w_p) r_p + SUM w_t (r_t-r_p)")
print("=" * 100)
for t in TIERS:
    pc = DEP[f"cost_{t}"]
    sel = tier_result(DEP[f"score_{t}"], pc, dv, t, SAFE[t])["sel"]
    w_t = dv.cost[:, 0] / dv.cost[:, 0].sum(); w_p = pc[:, 0] / pc[:, 0].sum()
    r_t = dv.cost[IDX, sel] / dv.cost[:, 0]; r_p = pc[IDX, sel] / pc[:, 0]
    rp, rt = (w_p * r_p).sum(), (w_t * r_t).sum()
    a = ((w_t - w_p) * r_p).sum()
    b = (w_t * (r_t - r_p)).sum()
    print(f"  {t:9s} pred={rp:.4f} true={rt:.4f}  gap={rt-rp:+.4f} = weight-term {a:+.4f} + ratio-term {b:+.4f}"
          f"   |weight|+|ratio| = {abs(a)+abs(b):.4f}  cap={TIER_MULT[t]:.2f} headroom={TIER_MULT[t]-rt:+.4f}")

print()
print("=" * 100)
print("STEP 6b  selection-induced cost bias, and per-model predicted-cost multipliers")
print("=" * 100)
for t in TIERS:
    pc = DEP[f"cost_{t}"]
    sel = tier_result(DEP[f"score_{t}"], pc, dv, t, SAFE[t])["sel"]
    print(f"  -- {t}")
    for j in range(3):
        le = np.log(pc[:, j]) - np.log(dv.cost[:, j])
        m = sel == j
        if m.sum() >= 5:
            print(f"     model {j} ({MODEL_IDS[j]:11s}) log(pred/true): all={le.mean():+.4f} sd={le.std():.3f} | "
                  f"selected(n={m.sum():3d})={le[m].mean():+.4f} | selection bias={le[m].mean()-le.mean():+.4f}"
                  f" -> implied multiplier {np.exp(-(le[m].mean()-le.mean())):.3f}")

print()
print("  effect of a per-model multiplier on the predicted cost (dev, deployed safety and best safety)")
print(f"  {'mult(mid,k1)':>16s} {'final@dep':>10s} {'final@best':>11s}   detail@best")
def run(mult, safe=None):
    tot = 0.0; det = []
    for t in TIERS:
        pc = DEP[f"cost_{t}"] * np.array([1.0, mult[0], mult[1]])[None, :]
        if safe is not None:
            r = tier_result(DEP[f"score_{t}"], pc, dv, t, safe[t])
            tot += TIER_WEIGHT[t] * r["tier_score"]
            det.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else '!'}")
        else:
            best = None
            for sf in np.arange(0.60, 1.201, 0.005):
                r = tier_result(DEP[f"score_{t}"], pc, dv, t, float(sf))
                if r["passed"] and (best is None or r["score"] > best[0]):
                    best = (r["score"], float(sf))
            tot += TIER_WEIGHT[t] * best[0]
            det.append(f"{t[:4]}={best[0]:.4f}@{best[1]:.3f}")
    return tot, " ".join(det)

for mult in [(1.0, 1.0), (1.0, 1.10), (1.0, 1.20), (1.0, 1.35), (1.05, 1.10), (1.05, 1.20),
             (1.10, 1.20), (1.0, 0.9), (0.95, 1.0)]:
    f1, _ = run(mult, SAFE); f2, det = run(mult, None)
    print(f"  {str(mult):>16s} {f1:10.4f} {f2:11.4f}   {det}")

print()
print("=" * 100)
print("STEP 6c  fast-tier fragility: true-ratio distribution under score jitter")
print("=" * 100)
rng = np.random.default_rng(3)
sig = np.array([np.sqrt(((DEP['score_fast'][:, j] - dv.score[:, j]) ** 2).mean()) for j in range(3)])
for t in TIERS:
    print(f"  -- {t} (cap {TIER_MULT[t]}, deployed safety {SAFE[t]}, "
          f"base true ratio {tier_result(DEP[f'score_{t}'], DEP[f'cost_{t}'], dv, t, SAFE[t])['ratio']:.4f})")
    for scale in (0.02, 0.05, 0.10, 0.25, 0.5, 1.0):
        ratios, scs, nup = [], [], []
        for _ in range(200):
            S = np.clip(DEP[f"score_{t}"] + rng.normal(0, 1, (n, 3)) * sig * scale, 0, 1)
            r = tier_result(S, DEP[f"cost_{t}"], dv, t, SAFE[t])
            ratios.append(r["ratio"]); scs.append(r["score"]); nup.append((r["sel"] > 0).sum())
        ratios = np.array(ratios)
        print(f"     noise x{scale:<5.2f} true ratio mean={ratios.mean():.4f} sd={ratios.std():.4f} "
              f"max={ratios.max():.4f} bust={np.mean(ratios>TIER_MULT[t])*100:5.1f}%  "
              f"score(if pass)={np.mean(scs):.4f}  n_upgraded={np.mean(nup):.0f}")
