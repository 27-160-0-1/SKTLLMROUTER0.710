# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a02-critic #12.  Test proposal P3 directly: price the k1 arm in the BUDGET
ACCOUNTING at max(pred, family quantile estimated on TRAIN), then re-tune safety.

Rationale from #10/#5: the catastrophic episodes are invisible to the cost head
(pred/true = 0.03..0.35), so a predicted-cost-quantile veto is a no-op.  But the
catastrophes are concentrated in two families, so a family-level floor is
estimable from train and does not need to identify the item.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, TIER_MULT, tier_result
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
tr, dv = load_split("train"), load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
famd = np.array([classify_family(t) for t in dv.texts])
famt = np.array([classify_family(t) for t in tr.texts])
fams = sorted(set(famt) | set(famd))
N = len(dv)
SG = np.arange(0.60, 1.401, 0.005)

print("per-family TRAIN quantiles of the true k1 cost (credits), vs dev truth:")
print(f"  {'family':11s} {'n_tr':>5s} {'q50':>8s} {'q75':>8s} {'q90':>8s} {'q95':>8s} "
      f"{'dev mean pred':>13s} {'dev mean true':>13s}")
Q = {}
for f in fams:
    mt = famt == f
    md = famd == f
    Q[f] = {q: float(np.quantile(tr.cost[mt, 2], q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)}
    print(f"  {f:11s} {int(mt.sum()):5d} {Q[f][0.5]:8.4f} {Q[f][0.75]:8.4f} {Q[f][0.9]:8.4f} "
          f"{Q[f][0.95]:8.4f} {P['cost_premium'][md,2].mean():13.4f} {dv.cost[md,2].mean():13.4f}")


def run(mode, q=None, label=""):
    tot = 0.0
    parts = []
    for t in TIERS:
        C = P[f"cost_{t}"].copy()
        if mode == "floor":
            fl = np.array([Q[f][q] for f in famd])
            C[:, 2] = np.maximum(C[:, 2], fl)
            C[:, 1] = np.maximum(C[:, 1], C[:, 0] * (1 + 1e-12))
            C[:, 2] = np.maximum(C[:, 2], C[:, 1] * (1 + 1e-12))
        best = None
        for sf in SG:
            r = tier_result(P[f"score_{t}"], C, dv, t, float(sf))
            if r["passed"] and (best is None or r["score"] > best[0]):
                best = (r["score"], float(sf), r["ratio"], int((r["sel"] == 2).sum()))
        rr = np.array([tier_result(P[f"score_{t}"], C, dv, t, float(sf))["ratio"] for sf in SG])
        tot += TIER_WEIGHT[t] * best[0]
        parts.append(f"{t[:4]}={best[0]:.4f}@{best[1]:.3f} k1={best[3]:3d} jump={np.abs(np.diff(rr)).max():.3f}")
    print(f"  {label:38s} weighted={tot:.4f}   " + "  ".join(parts))
    return tot


print("\noracle-tuned safety per configuration (the comparison across rows is the point):")
base = run("none", label="deployed cost head")
for q in (0.5, 0.75, 0.9, 0.95, 0.99):
    run("floor", q, f"k1 priced at max(pred, train q{int(q*100)})")

print("\nsame, but at the DEPLOYED safety constants (.98/.89/.88) with no re-tuning:")
SAFE = {"fast": .98, "balanced": .89, "premium": .88}
for q in (None, 0.75, 0.9, 0.95):
    tot = 0.0
    parts = []
    for t in TIERS:
        C = P[f"cost_{t}"].copy()
        if q is not None:
            fl = np.array([Q[f][q] for f in famd])
            C[:, 2] = np.maximum(C[:, 2], fl)
        r = tier_result(P[f"score_{t}"], C, dv, t, SAFE[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else '!'}")
    print(f"  {'deployed' if q is None else f'floor q{int(q*100)}':38s} weighted={tot:.4f}   " + "  ".join(parts))

print("\nbootstrap bust probability (880 resamples x 1500) at the oracle-tuned safety of each row:")


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


idx = np.random.default_rng(7).integers(0, N, size=(1500, N))
for q, safeties in ((None, {"fast": .98, "balanced": .89, "premium": .88}),
                    (0.9, None), (0.95, None)):
    row = []
    tot = 0.0
    for t in TIERS:
        C = P[f"cost_{t}"].copy()
        if q is not None:
            fl = np.array([Q[f][q] for f in famd])
            C[:, 2] = np.maximum(C[:, 2], fl)
        if safeties is None:
            sf = max(SG, key=lambda s: (lambda r: r["score"] if r["passed"] else -1)(
                tier_result(P[f"score_{t}"], C, dv, t, float(s))))
        else:
            sf = safeties[t]
        Sb, Cb = P[f"score_{t}"][idx], C[idx]
        ts, tc = dv.score[idx], dv.cost[idx]
        cap = Cb[:, :, 0].sum(1) * max(1.0, TIER_MULT[t] * sf)
        sel = alloc_batch(Sb, Cb, cap)
        got = np.take_along_axis(ts, sel[:, :, None], 2)[:, :, 0].mean(1)
        cst = np.take_along_axis(tc, sel[:, :, None], 2)[:, :, 0].sum(1)
        ok = (cst / tc[:, :, 0].sum(1)) <= TIER_MULT[t] + 1e-15
        tot += TIER_WEIGHT[t] * float(np.mean(got * ok))
        row.append(f"{t[:4]} sf={sf:.3f} bust={100*(1-ok.mean()):.1f}%")
    print(f"  {'deployed' if q is None else f'floor q{int(q*100)}':16s} bootstrap EV={tot:.4f}   " + "  ".join(row))
