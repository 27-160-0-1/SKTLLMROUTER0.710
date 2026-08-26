# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Why does the safety ratio have to be so low, and what would raising it be worth?

The repriced triple leaves premium at 2.33 of an allowed 4.0 -- 40 % of the budget held back as
insurance.  Whether that insurance can be reduced depends on what it is insuring against:

  A. cost *prediction* error -- the allocator commits to items whose realised cost differs from
     the prediction.  Fixable by better cost modelling.
  B. item-mix variance -- a different draw of items has a different achievable ratio, whatever
     the predictions.  Not fixable; the margin is irreducible.

Allocating on the true costs isolates B: whatever safety ratio is still unsafe under oracle
costs is variance, and the gap up to it is the prize for better prediction.

It also tests one targeted intervention: refusing any upgrade whose predicted cost exceeds a
share of the batch's own budget.  b01 found a single dev episode carrying 6.3 % of the light
baseline, and E59b found think emitting ~9,765 tokens on an AIME item; if the tail is what
forces the ratio down, capping per-item exposure should buy back safety ratio cheaply.

Usage:
    PYTHONPATH=src python tools/diag_safety_headroom.py --artifact A.json [--boot 400]
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

import numpy as np

from ossp_router import learned_router
from ossp_router.protocol import MODEL_IDS, TIERS, load_bundled_policy, load_input, load_outcomes

WEIGHTS = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}


def allocate(ps, pc, mult, safety, item_cap=None):
    """Lagrangian allocation; item_cap forbids picks whose predicted cost exceeds that share
    of the batch budget (None = the shipped behaviour)."""
    light_total = pc[:, 0].sum()
    cap = light_total * max(1.0, mult * safety)
    tie = np.array([2e-12, 1e-12, 0.0])
    banned = None
    if item_cap is not None:
        banned = pc > item_cap * cap
        banned[:, 0] = False                    # the light model is always allowed

    def choose(penalty):
        u = ps - penalty * pc / light_total + tie
        if banned is not None:
            u = np.where(banned, -np.inf, u)
        pick = np.argmax(u, axis=1)
        return pick, pc[np.arange(len(pick)), pick].sum()

    pick, total = choose(0.0)
    if total > cap:
        low, high = 0.0, 1.0
        pick, total = choose(high)
        while total > cap and high < 2 ** 60:
            low, high = high, high * 2.0
            pick, total = choose(high)
        for _ in range(40):
            mid = (low + high) / 2.0
            cand, cand_total = choose(mid)
            if cand_total <= cap:
                high, pick, total = mid, cand, cand_total
            else:
                low = mid
    return pick


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--input", type=Path, default=Path("data/materialized/dev/inputs.json"))
    ap.add_argument("--outcomes", type=Path, default=Path("data/dev/outcomes.json"))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    policy = load_bundled_policy()
    raw = json.loads(a.artifact.read_text(encoding="utf-8"))
    raw.pop("public_lookup", None)
    artifact = learned_router.parse_artifact(raw, base_path=a.artifact.parent)
    episodes = list(load_input(a.input).episodes)
    idx = {(o.episode_id, o.model_id): o for o in load_outcomes(a.outcomes).outcomes}
    n = len(episodes)
    unit = Decimal(policy.token_unit)

    def true_cost(eid, mid):
        o = idx[(eid, mid)]
        r = policy.models[mid]
        return float(r.fixed_cost + Decimal(o.input_tokens) * r.input_token_rate / unit
                     + Decimal(o.output_tokens) * r.output_token_rate / unit)

    C = np.array([[true_cost(e.episode_id, m) for m in MODEL_IDS] for e in episodes])
    S = np.array([[float(idx[(e.episode_id, m)].score) for m in MODEL_IDS] for e in episodes])

    rng = np.random.default_rng(a.seed)
    samples = [rng.integers(0, n, size=n) for _ in range(a.boot)]

    def bust_and_score(ps, pc, mult, safety, item_cap=None):
        bust, sc = 0, 0.0
        for s in samples:
            q = allocate(ps[s], pc[s], mult, safety, item_cap)
            r = np.arange(len(s))
            if C[s][r, q].sum() / C[s][:, 0].sum() > mult:
                bust += 1
            else:
                sc += S[s][r, q].mean()
        return bust / len(samples), sc / len(samples)

    for tier in TIERS:
        mult = float(policy.tiers[tier].budget_multiplier)
        preds = [learned_router.predict_episode_augmented(e, artifact, tier) for e in episodes]
        ps = np.array([[p[0][m] for m in MODEL_IDS] for p in preds])
        pc = np.array([[p[1][m] for m in MODEL_IDS] for p in preds])

        # how wrong are the cost predictions on the items the allocator actually upgrades?
        err = np.log(np.maximum(pc, 1e-12)) - np.log(np.maximum(C, 1e-12))
        print(f"\n=== {tier} (multiplier {mult}) ===")
        print("  로그 비용 예측오차(중앙값 / p90 절대):  "
              + "  ".join(f"{m} {np.median(err[:, j]):+.2f}/{np.quantile(np.abs(err[:, j]), 0.9):.2f}"
                          for j, m in enumerate(MODEL_IDS)))
        print(f"{'safety':>8}{'예측비용 초과':>14}{'참비용 초과':>13}"
              f"{'점수':>9}{'상한(참비용)':>14}")
        for value in (0.56, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.94):
            if mult * value < 1.0:
                continue
            b_pred, s_pred = bust_and_score(ps, pc, mult, value)
            b_true, s_true = bust_and_score(ps, C, mult, value)
            print(f"{value:>8.2f}{b_pred:>13.1%}{b_true:>13.1%}{s_pred:>9.4f}{s_true:>13.4f}")

        print("  -- 문항별 비용 상한 (예산 대비 비율) --")
        print(f"{'cap':>8}{'safety':>8}{'초과':>9}{'점수':>9}")
        for item_cap in (0.02, 0.01, 0.005, 0.002):
            for value in (0.56, 0.65, 0.73, 0.85):
                if mult * value < 1.0:
                    continue
                b, sc = bust_and_score(ps, pc, mult, value, item_cap)
                if b == 0.0:
                    print(f"{item_cap:>8.3f}{value:>8.2f}{b:>8.1%}{sc:>9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
