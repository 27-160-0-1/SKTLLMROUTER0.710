# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a08 probe 2: cost anatomy -- how much of log-cost variance is input tokens?

If the input-token part of the cost were known EXACTLY (a tokenizer gives it),
how much of the remaining cost uncertainty is left?  Bounds the value of an
exact-token-count feature.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_all, MODEL_IDS, RATES, TOKEN_UNIT  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

tr, dv = load_all()
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)

for sp in (tr, dv):
    print(f"=== {sp.name} n={len(sp)}")
    for j, m in enumerate(MODEL_IDS):
        _f, ri, ro = RATES[m]
        cin = sp.itok[:, j] * ri / TOKEN_UNIT
        cout = sp.otok[:, j] * ro / TOKEN_UNIT
        tot = cin + cout
        print(f"  {m:11s} in-share mean={cin.sum()/tot.sum():.3f} "
              f"median={np.median(cin/tot):.3f}  "
              f"itok med={np.median(sp.itok[:,j]):.0f} otok med={np.median(sp.otok[:,j]):.0f} "
              f"otok p95={np.percentile(sp.otok[:,j],95):.0f}")
    print()

print("=== how much log-cost variance survives if itok is known exactly? ===")
print("   (compare sd of log(cost) around family mean, vs sd of log(cost) given exact cin)")
sp = dv
fams = np.array([classify_family(t) for t in sp.texts])
for j, m in enumerate(MODEL_IDS):
    _f, ri, ro = RATES[m]
    cin = sp.itok[:, j] * ri / TOKEN_UNIT
    cout = sp.otok[:, j] * ro / TOKEN_UNIT
    lc = np.log(cin + cout)
    # residual after removing family means (a strong cheap predictor)
    res_f = lc.copy()
    for f in np.unique(fams):
        msk = fams == f
        res_f[msk] -= lc[msk].mean()
    # residual if we knew cin exactly and predicted cout by family median
    pred = cin.copy()
    for f in np.unique(fams):
        msk = fams == f
        pred[msk] = cin[msk] + np.median(cout[msk])
    res_h = lc - np.log(pred)
    res_h -= res_h.mean()
    print(f"  {m:11s} sd[log c | family] = {res_f.std():.3f}   "
          f"sd[log c | exact cin + family-median cout] = {res_h.std():.3f}  "
          f"deployed dev log-err sd = "
          f"{(np.log(P['cost_premium'][:,j]) - np.log(sp.cost[:,j])).std():.3f}")

print()
print("=== ceiling probe: replace the INPUT part of the deployed cost prediction with truth ===")
from labdata import TIERS, TIER_WEIGHT, tier_result  # noqa: E402
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}


def run(name, mkc, safety=SAFE):
    tot = 0.0
    parts = []
    for t in TIERS:
        r = tier_result(P[f"score_{t}"], mkc(t), sp, t, safety[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else '!BUST'}")
    print(f"  {name:52s} {tot:.4f}  " + " ".join(parts))
    return tot


def best_safety(mkc):
    tot = 0.0
    parts = []
    for t in TIERS:
        best = None
        for s in np.arange(0.60, 1.301, 0.005):
            r = tier_result(P[f"score_{t}"], mkc(t), sp, t, float(s))
            if r["passed"] and (best is None or r["score"] > best[0]):
                best = (r["score"], float(s))
        tot += TIER_WEIGHT[t] * best[0]
        parts.append(f"{t[:4]}={best[0]:.4f}@{best[1]:.3f}")
    print(f"  {'   ^ same, best safety':52s} {tot:.4f}  " + " ".join(parts))
    return tot


IN_RATE = np.array([RATES[m][1] for m in MODEL_IDS]) / TOKEN_UNIT
OUT_RATE = np.array([RATES[m][2] for m in MODEL_IDS]) / TOKEN_UNIT
cin_true = sp.itok * IN_RATE[None, :]
cout_true = sp.otok * OUT_RATE[None, :]


def hybrid(t):
    """keep the deployed prediction's implied *total*, but swap in the exact input part,
    scaling the predicted output part so the model keeps its own output estimate."""
    pc = P[f"cost_{t}"]
    # the deployed head predicts total cost.  implied output part = pc - cin_true (floored)
    cout_hat = np.maximum(pc - cin_true, 1e-9)
    return cin_true + cout_hat


run("deployed pred cost", lambda t: P[f"cost_{t}"])
best_safety(lambda t: P[f"cost_{t}"])
run("exact input part + implied output part", hybrid)
best_safety(hybrid)
run("true cost", lambda t: sp.cost)
best_safety(lambda t: sp.cost)
