# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""E65 -- buy back safety ratio with per-model conservative cost offsets.

E64 established that the whole safety margin insures against cost prediction error (allocating
on true costs never busts, at any ratio) and that the error sits in the think head: log-cost
RMSE light 0.556 / mid 0.458 / think 0.677.

A uniform cost inflation is a no-op -- the cap is `light_total * mult * safety`, so scaling every
model's cost scales the cap with it.  A *per-model* offset does not cancel: adding `d` to a
model's predicted log-cost makes the allocator treat that model as `exp(d)` times more expensive
than it really predicts, in both the efficiency ranking and the budget accounting.  If the
think head is the one that surprises us, charging think a premium should let the *safety ratio*
rise by more than the offset costs in lost upgrades.

The search is over (offset_mid, offset_think) x safety, scored by "highest tier score among
settings that bust in no resample of any scenario".

Runtime impact: none.  An offset is a constant added to a log-cost head's baseline, so it ships
as a coefficient change, not a code change.

Usage:
    PYTHONPATH=src python tools/e65_cost_offsets.py --artifact A.json [--boot 200]
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


def allocate(ps, pc, mult, safety):
    light_total = pc[:, 0].sum()
    cap = light_total * max(1.0, mult * safety)
    tie = np.array([2e-12, 1e-12, 0.0])

    def choose(penalty):
        pick = np.argmax(ps - penalty * pc / light_total + tie, axis=1)
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
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mid", default="0.0,0.15,0.3")
    ap.add_argument("--think", default="0.0,0.2,0.4,0.6,0.8")
    a = ap.parse_args()

    policy = load_bundled_policy()
    raw = json.loads(a.artifact.read_text(encoding="utf-8"))
    raw.pop("public_lookup", None)
    artifact = learned_router.parse_artifact(raw, base_path=a.artifact.parent)
    shipped = raw.get("tier_safety_ratios") or {}
    episodes = list(load_input(a.input).episodes)
    idx = {(o.episode_id, o.model_id): o for o in load_outcomes(a.outcomes).outcomes}
    n = len(episodes)
    unit = Decimal(policy.token_unit)

    def true_cost(eid, mid_):
        o = idx[(eid, mid_)]
        r = policy.models[mid_]
        return float(r.fixed_cost + Decimal(o.input_tokens) * r.input_token_rate / unit
                     + Decimal(o.output_tokens) * r.output_token_rate / unit)

    C = np.array([[true_cost(e.episode_id, m) for m in MODEL_IDS] for e in episodes])
    S = np.array([[float(idx[(e.episode_id, m)].score) for m in MODEL_IDS] for e in episodes])

    rng = np.random.default_rng(a.seed)
    full = [rng.integers(0, n, size=n) for _ in range(a.boot)]
    small = [rng.integers(0, n, size=n // 2) for _ in range(a.boot)]
    hit_at = [rng.integers(0, n) for _ in range(a.boot)]

    def safe(ps, pc, mult, safety):
        """True when no resample of any scenario busts."""
        for name in ("plain", "runaway", "inflation", "small"):
            samples = small if name == "small" else full
            for k, s in enumerate(samples):
                q = allocate(ps[s], pc[s], mult, safety)
                r = np.arange(len(s))
                cost = C[s][r, q].copy()
                light = C[s][:, 0].sum()
                if name == "runaway":
                    cost[hit_at[k] % len(s)] += 0.065 * light
                elif name == "inflation":
                    cost = cost * np.where(q == 2, 1.25, np.where(q == 1, 1.10, 1.0))
                if cost.sum() / light > mult:
                    return False
        return True

    mids = [float(v) for v in a.mid.split(",")]
    thinks = [float(v) for v in a.think.split(",")]
    grand_best, grand_total = {}, 0.0

    for tier in TIERS:
        mult = float(policy.tiers[tier].budget_multiplier)
        preds = [learned_router.predict_episode_augmented(e, artifact, tier) for e in episodes]
        ps = np.array([[p[0][m] for m in MODEL_IDS] for p in preds])
        pc0 = np.array([[p[1][m] for m in MODEL_IDS] for p in preds])
        # safety grid must start above 1/mult, below which the cap collapses to the light total
        grid = [round(v, 2) for v in np.arange(max(1.0 / mult, 0.5) + 0.02, 1.0, 0.02)]

        print(f"\n=== {tier} (multiplier {mult}, 현행 안전계수 {shipped.get(tier)}) ===")
        print(f"{'mid 오프셋':>10}{'think 오프셋':>13}{'최대 안전계수':>14}{'사용률':>9}{'점수':>9}")
        best = None
        for dm in mids:
            for dt in thinks:
                pc = pc0 * np.array([1.0, np.exp(dm), np.exp(dt)])
                top = None
                for value in reversed(grid):            # descend: first safe value is the largest
                    if safe(ps, pc, mult, value):
                        top = value
                        break
                if top is None:
                    continue
                pick = allocate(ps, pc, mult, top)
                rows = np.arange(n)
                score = S[rows, pick].mean()
                ratio = C[rows, pick].sum() / C[:, 0].sum()
                mark = ""
                if best is None or score > best[0]:
                    best = (score, dm, dt, top, ratio)
                    mark = "  <- 최고"
                print(f"{dm:>10.2f}{dt:>13.2f}{top:>14.2f}{ratio:>9.3f}{score:>9.4f}{mark}")
        if best is None:
            print("  안전한 설정 없음")
            continue
        score, dm, dt, top, ratio = best
        grand_best[tier] = dict(mid_offset=dm, think_offset=dt, safety=top, score=score, ratio=ratio)
        grand_total += WEIGHTS[tier] * score
        print(f"  -> {tier}: mid +{dm:.2f} / think +{dt:.2f}, 안전계수 {top:.2f}, 점수 {score:.4f}")

    print("\n=== E65 요약 ===")
    for tier, v in grand_best.items():
        print(f"  {tier:<10} mid +{v['mid_offset']:.2f}  think +{v['think_offset']:.2f}  "
              f"safety {v['safety']:.2f}  사용률 {v['ratio']:.3f}  점수 {v['score']:.4f}")
    print(f"  가중 held-out {grand_total:.6f}   (E63b 기준 0.701903)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
