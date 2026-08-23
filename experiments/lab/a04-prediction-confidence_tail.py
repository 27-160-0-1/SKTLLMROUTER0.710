# SPDX-License-Identifier: Apache-2.0
"""a04 / step 7 -- the cost tail that the fragility lives in.

Which single items can move the budget ratio by more than the headroom, how
badly are they cost-predicted, and does excluding/flooring them change the
fragility?  (E08 rejected per-item cost caps in the 0.684 era; this re-tests
the premise on the E43 predictions.)
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

dv = load_split("dev")
n = len(dv); IDX = np.arange(n)
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
fam = np.array([similarity.classify_family(t) for t in dv.texts])
LT = dv.cost[:, 0].sum()

print("=" * 100)
print("STEP 7a  single-item budget footprint (true upgrade cost / true light total)")
print("=" * 100)
for t in TIERS:
    sel = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, SAFE[t])["sel"]
    ratio = dv.cost[IDX, sel].sum() / LT
    head = TIER_MULT[t] - ratio
    d_up = np.where(sel == 0, (dv.cost[:, 1] - dv.cost[:, 0]) / LT,
                    np.where(sel == 1, (dv.cost[:, 2] - dv.cost[:, 1]) / LT, 0.0))
    danger = d_up > head
    print(f"  {t:9s} ratio={ratio:.4f} headroom={head:.4f}  items whose ONE upgrade busts the tier: "
          f"{danger.sum():3d}  (top5 footprints " + " ".join(f"{v:.3f}" for v in np.sort(d_up)[::-1][:5]) + ")")
    idx = np.argsort(-d_up)[:6]
    for i in idx:
        j_from = sel[i]; j_to = min(j_from + 1, 2)
        pd = (P[f"cost_{t}"][i, j_to] - P[f"cost_{t}"][i, j_from]) / P[f"cost_{t}"][:, 0].sum()
        print(f"      item {i:4d} fam={fam[i]:15s} sel={j_from} footprint={d_up[i]:.4f} "
              f"predicted={pd:.4f} under-pred x{d_up[i]/max(pd,1e-9):6.1f} "
              f"len={len(dv.texts[i]):6d} otok(k1)={dv.otok[i,2]:.0f}")

print()
print("=" * 100)
print("STEP 7b  is the fast-tier bust under jitter caused by a handful of items?")
print("=" * 100)
rng = np.random.default_rng(11)
sig = np.array([np.sqrt(((P['score_fast'][:, j] - dv.score[:, j]) ** 2).mean()) for j in range(3)])
t = "fast"
sel0 = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, SAFE[t])["sel"]
base_ratio = dv.cost[IDX, sel0].sum() / LT
d_up = (dv.cost[:, 1] - dv.cost[:, 0]) / LT
worst = np.argsort(-np.where(sel0 == 0, d_up, 0.0))[:5]
print(f"  base ratio {base_ratio:.4f}, cap {TIER_MULT[t]}, watching items {worst.tolist()} "
      f"(footprints {np.round(d_up[worst],4).tolist()})")
for scale in (0.05, 0.10, 0.25):
    busts = 0; bust_with = 0; flips = np.zeros(len(worst))
    for _ in range(400):
        S = np.clip(P[f"score_{t}"] + rng.normal(0, 1, (n, 3)) * sig * scale, 0, 1)
        r = tier_result(S, P[f"cost_{t}"], dv, t, SAFE[t])
        f = r["sel"][worst] > 0
        flips += f
        if not r["passed"]:
            busts += 1
            bust_with += bool(f.any())
    print(f"    noise x{scale:<5.2f} bust {busts:3d}/400   of those {bust_with:3d} had >=1 watched item upgraded"
          f"   per-item flip rate {np.round(flips/400,3).tolist()}")

print()
print("=" * 100)
print("STEP 7c  a cheap guard: never upgrade an item whose PREDICTED footprint is tiny but")
print("  whose text is long / k1-output-prone.  Proxy guard = clip predicted upgrade cost")
print("  from below by a family+length quantile of the TRAIN relation (here: use a floor")
print("  on predicted cost = q-quantile of predicted cost within (family, length decile)).")
print("=" * 100)
L = np.array([len(x) for x in dv.texts], dtype=float)
lb = np.digitize(L, np.quantile(L, np.linspace(0, 1, 11))[1:-1])
for t in TIERS:
    pc = P[f"cost_{t}"]
    base = tier_result(P[f"score_{t}"], pc, dv, t, SAFE[t])
    out = [f"  {t:9s} base score={base['score']:.4f} ratio={base['ratio']:.4f}"]
    for q in (0.5, 0.75, 0.9):
        pc2 = pc.copy()
        for j in range(3):
            for f in set(fam):
                for b in range(10):
                    m = (fam == f) & (lb == b)
                    if m.sum() >= 5:
                        pc2[m, j] = np.maximum(pc2[m, j], np.quantile(pc[m, j], q))
        pc2[:, 1] = np.maximum(pc2[:, 1], pc2[:, 0] * 1.0000001)
        pc2[:, 2] = np.maximum(pc2[:, 2], pc2[:, 1] * 1.0000001)
        r = tier_result(P[f"score_{t}"], pc2, dv, t, SAFE[t])
        # best safety too
        best = None
        for sf in np.arange(0.6, 1.201, 0.005):
            rr = tier_result(P[f"score_{t}"], pc2, dv, t, float(sf))
            if rr["passed"] and (best is None or rr["score"] > best[0]):
                best = (rr["score"], float(sf))
        out.append(f"      floor q={q:.2f}: score={r['score']:.4f} ratio={r['ratio']:.4f} "
                   f"pass={r['passed']} | best-safety {best[0]:.4f}@{best[1]:.3f}")
    print("\n".join(out))
