# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a14 step 3 -- ASSUMPTION-FREE bracket on the honest ceiling (no noise model).

The generations inside an episode are i.i.d. Bernoulli(p) given p, so a random
disjoint split of the n generations gives two conditionally independent views.
    allocate on  view A (m generations)
    evaluate on  view B (n-m generations)     ->  unbiased for  E[ p_sel(A) ]
Because the p-oracle maximises E[p_sel] over all selections, this is an
unbiased estimate of a quantity that is <= the honest ceiling: a LOWER bound.

Conversely, for every item  E[ max_j s_ij ] >= max_j p_ij, so the realised-score
oracle (0.8034 on dev) is an UPPER bound on the honest ceiling in expectation.

We also compute, for the *deployed* predictions, the honest (split-evaluated)
final score -- which needs no split at all, because the deployed predictions do
not see the labels; it is reported as a consistency anchor.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_WEIGHT, TIER_MULT, tier_result   # noqa: E402

dv = load_split("dev")
N = len(dv)
IDX = np.arange(N)


def eval_alloc(pred_score, pred_cost, eval_score, safety):
    """allocate with pred_*, score the chosen arm with eval_score."""
    tot = 0.0
    parts = []
    for t in TIERS:
        r = tier_result(pred_score, pred_cost, dv, t, safety[t])
        v = float(eval_score[IDX, r["sel"]].mean())
        tot += TIER_WEIGHT[t] * (v if r["passed"] else 0.0)
        parts.append((t, v, r["ratio"], r["passed"]))
    return tot, parts


SAFE1 = {t: 1.0 for t in TIERS}


def split_views(rng, m):
    """random disjoint split: view A has m generations, view B has n-m."""
    n = dv.ngen.astype(int)
    k = np.rint(dv.score * dv.ngen).astype(int)
    # hypergeometric: successes landing in view A
    kA = rng.hypergeometric(k, n - k, m)
    sA = kA / m
    sB = (k - kA) / (n - m)
    return sA, sB


def main():
    print("=" * 92)
    print("ASSUMPTION-FREE BRACKET ON THE HONEST CEILING (dev, true costs, safety 1.0)")
    print("=" * 92)
    tot_true, _ = eval_alloc(dv.score, dv.cost, dv.score, SAFE1)
    print(f"realised-score oracle, evaluated on the same realised score : {tot_true:.4f}"
          "   (UPPER bound, inflated)")

    rng = np.random.default_rng(20260819)
    R = 200
    for m in (1,):
        a_own, a_cross = [], []
        for _ in range(R):
            sA, sB = split_views(rng, m)
            v1, _ = eval_alloc(sA, dv.cost, sA, SAFE1)   # inflated, m-gen oracle
            v2, _ = eval_alloc(sA, dv.cost, sB, SAFE1)   # honest, m-gen oracle
            a_own.append(v1); a_cross.append(v2)
        a_own = np.array(a_own); a_cross = np.array(a_cross)
        print(f"m={m}-generation oracle, self-evaluated                       : "
              f"{a_own.mean():.4f} +- {a_own.std()/np.sqrt(R):.4f}")
        print(f"m={m}-generation oracle, evaluated on the held-out generations: "
              f"{a_cross.mean():.4f} +- {a_cross.std()/np.sqrt(R):.4f}"
              "   (LOWER bound, honest)")
        print(f"  -> self-evaluation inflation at m={m}: "
              f"{a_own.mean()-a_cross.mean():+.4f}")

    # the same split trick applied to a *fixed* selection: sanity check that
    # cross-evaluation is unbiased when the selection does not look at labels
    print("\nsanity: fixed (label-free) selections, self vs cross evaluation")
    for name, sel_score in (("all-light", np.tile([1., 0., 0.], (N, 1))),
                            ("all-k1", np.tile([0., 0., 1.], (N, 1)))):
        vs, vc = [], []
        for _ in range(50):
            sA, sB = split_views(rng, 1)
            vs.append(eval_alloc(sel_score, dv.cost, sA, SAFE1)[0])
            vc.append(eval_alloc(sel_score, dv.cost, sB, SAFE1)[0])
        print(f"  {name:10s} viewA={np.mean(vs):.4f}  viewB={np.mean(vc):.4f}  "
              f"full={eval_alloc(sel_score, dv.cost, dv.score, SAFE1)[0]:.4f}")

    # ------------------------------------------------------------------ the same,
    # but under the DEPLOYED cost model + deployed safety, so the numbers are on the
    # same footing as the 0.7017 held-out score
    P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
    SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
    print("\n" + "=" * 92)
    print("SAME BRACKET UNDER THE DEPLOYED COST MODEL + E43 SAFETY (.98/.87/.85)")
    print("=" * 92)

    def eval_alloc_costpred(pred_score_fn, eval_score, safety):
        tot = 0.0
        for t in TIERS:
            r = tier_result(pred_score_fn(t), P[f"cost_{t}"], dv, t, safety[t])
            v = float(eval_score[IDX, r["sel"]].mean())
            tot += TIER_WEIGHT[t] * (v if r["passed"] else 0.0)
        return tot

    print(f"deployed predictions, realised-score eval : "
          f"{eval_alloc_costpred(lambda t: P[f'score_{t}'], dv.score, SAFE):.4f}")
    vs, vc = [], []
    for _ in range(R):
        sA, sB = split_views(rng, 1)
        vs.append(eval_alloc_costpred(lambda t: sA, sA, SAFE))
        vc.append(eval_alloc_costpred(lambda t: sA, sB, SAFE))
    print(f"1-gen oracle self-eval                    : {np.mean(vs):.4f}")
    print(f"1-gen oracle honest eval                  : {np.mean(vc):.4f}")
    print(f"realised-score oracle self-eval           : "
          f"{eval_alloc_costpred(lambda t: dv.score, dv.score, SAFE):.4f}")


if __name__ == "__main__":
    main()
