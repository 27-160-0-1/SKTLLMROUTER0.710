# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a14 step 10 -- independent check of the deployed bust probability.

Does NOT use the fast Alloc class: resamples 880 dev episodes with replacement,
rebuilds the arrays, and calls labdata.allocate / labdata.tier_result directly,
exactly as the deployed router would run on that batch.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from labdata import load_split, allocate, TIERS, TIER_MULT, TIER_WEIGHT, Split  # noqa: E402

dv = load_split("dev")
P43 = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE43 = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
N = len(dv)


def run(B=1000, seed=99, size=None, safeties=None):
    size = size or N
    safeties = safeties or SAFE43
    rng = np.random.default_rng(seed)
    res = {t: dict(ratio=[], score=[], bust=0) for t in TIERS}
    for b in range(B):
        idx = rng.integers(0, N, size)
        tc = dv.cost[idx]
        ts = dv.score[idx]
        for t in TIERS:
            ps = P43[f"score_{t}"][idx]
            pc = P43[f"cost_{t}"][idx]
            sel = allocate(ps, pc, tc, TIER_MULT[t], safeties[t])
            ar = np.arange(size)
            ratio = tc[ar, sel].sum() / tc[:, 0].sum()
            ok = ratio <= TIER_MULT[t] + 1e-15
            res[t]["ratio"].append(ratio)
            res[t]["score"].append(ts[ar, sel].mean() if ok else 0.0)
            res[t]["bust"] += 0 if ok else 1
    return res


def main():
    for size in (880, 1760):
        print("=" * 88)
        print(f"direct bootstrap, batch size {size}, deployed predictions and safety "
              f"{SAFE43['fast']}/{SAFE43['balanced']}/{SAFE43['premium']}")
        print("=" * 88)
        res = run(B=1000, size=size)
        tot = 0.0
        for t in TIERS:
            r = np.array(res[t]["ratio"]); s = np.array(res[t]["score"])
            tot += TIER_WEIGHT[t] * s.mean()
            print(f"  {t:9s} ratio mean={r.mean():.4f} sd={r.std():.4f} "
                  f"p95={np.percentile(r,95):.4f} cap={TIER_MULT[t]:.2f} "
                  f"bust={100*res[t]['bust']/len(r):.1f}%  EV(score)={s.mean():.4f}")
        print(f"  weighted EV = {tot:.4f}   (dev point score = 0.7005 with these preds)")

    print("\n" + "=" * 88)
    print("subsample WITHOUT replacement (440 of 880) -- removes bootstrap duplication")
    print("=" * 88)
    rng = np.random.default_rng(5)
    for t in TIERS:
        rs, busts = [], 0
        for b in range(1000):
            idx = rng.permutation(N)[:440]
            tc = dv.cost[idx]
            sel = allocate(P43[f"score_{t}"][idx], P43[f"cost_{t}"][idx], tc,
                           TIER_MULT[t], SAFE43[t])
            ratio = tc[np.arange(440), sel].sum() / tc[:, 0].sum()
            rs.append(ratio); busts += ratio > TIER_MULT[t]
        rs = np.array(rs)
        print(f"  {t:9s} ratio mean={rs.mean():.4f} sd={rs.std():.4f} "
              f"bust={100*busts/len(rs):.1f}%")

    print("\n" + "=" * 88)
    print("what safety keeps the 880-bootstrap bust below 1% / 5%?")
    print("=" * 88)
    for t in TIERS:
        row = []
        for s in np.arange(0.60, 1.101, 0.01):
            res = run(B=300, seed=7, safeties={**SAFE43, t: float(s)})
            row.append((float(s), res[t]["bust"] / 300.0))
        s1 = max([s for s, b in row if b <= 0.01], default=None)
        s5 = max([s for s, b in row if b <= 0.05], default=None)
        print(f"  {t:9s} bust<=1% at safety <= {s1}   bust<=5% at safety <= {s5}   "
              f"deployed {SAFE43[t]}")


if __name__ == "__main__":
    main()
