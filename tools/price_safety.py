# SPDX-License-Identifier: Apache-2.0
"""Re-price the per-tier safety ratios against a no-bust requirement.

A tier that exceeds its budget multiplier scores 0, so when the requirement is "must not bust"
the safety ratio is not an EV argmax -- it is the largest value that survives every stress
scenario.  Choosing it that way also keeps selection bias small: we are satisfying a constraint,
not fishing the evaluation set for score.

Scenarios (E55's, plus a size stress):
  plain      bootstrap resamples of the evaluation set
  runaway    plus one injected pathological upgrade worth 6.5 % of the light baseline
             (b01: 82 % of the fast tier's ratio variance on dev came from a single episode)
  inflation  plus a systematic cost surprise, 1.25x on axk1-think and 1.10x on ax31
  small      half-size batches, where the same variance has twice the relative weight

The allocator is re-run inside every resample: `select_models` sizes its cap from the batch it
is handed, so the router re-balances rather than reusing the picks it made for this set.

Usage:
    PYTHONPATH=src python tools/price_safety.py --artifact A.json [--boot 800]
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
SCENARIOS = ("plain", "runaway", "inflation", "small")


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
    ap.add_argument("--boot", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grid", default=None, help="comma-separated safety values to test")
    ap.add_argument("--write", type=Path, default=None,
                    help="write the chosen triple into this artifact's tier_safety_ratios "
                         "(safety enters allocation only, so no refit is needed)")
    a = ap.parse_args()

    policy = load_bundled_policy()
    raw = json.loads(a.artifact.read_text(encoding="utf-8"))
    raw.pop("public_lookup", None)
    artifact = learned_router.parse_artifact(raw, base_path=a.artifact.parent)
    shipped = raw.get("tier_safety_ratios") or {}
    episodes = list(load_input(a.input).episodes)
    index = {(o.episode_id, o.model_id): o for o in load_outcomes(a.outcomes).outcomes}
    n = len(episodes)
    unit = Decimal(policy.token_unit)

    def true_cost(eid, mid):
        o = index[(eid, mid)]
        r = policy.models[mid]
        return float(r.fixed_cost + Decimal(o.input_tokens) * r.input_token_rate / unit
                     + Decimal(o.output_tokens) * r.output_token_rate / unit)

    C = np.array([[true_cost(e.episode_id, m) for m in MODEL_IDS] for e in episodes])
    S = np.array([[float(index[(e.episode_id, m)].score) for m in MODEL_IDS] for e in episodes])

    rng = np.random.default_rng(a.seed)
    full = [rng.integers(0, n, size=n) for _ in range(a.boot)]
    small = [rng.integers(0, n, size=n // 2) for _ in range(a.boot)]
    hit_at = [rng.integers(0, n) for _ in range(a.boot)]

    chosen, rows_out = {}, []
    print(f"[price] {a.artifact.parent.name}: {n} episodes, {a.boot} resamples per scenario")
    for tier in TIERS:
        mult = float(policy.tiers[tier].budget_multiplier)
        preds = [learned_router.predict_episode_augmented(e, artifact, tier) for e in episodes]
        ps = np.array([[p[0][m] for m in MODEL_IDS] for p in preds])
        pc = np.array([[p[1][m] for m in MODEL_IDS] for p in preds])
        grid = ([round(x, 3) for x in np.arange(0.50, 1.001, 0.01)]
                if not a.grid else [float(v) for v in a.grid.split(",")])
        print(f"\n=== {tier} (multiplier {mult}, shipped safety {shipped.get(tier)}) ===")
        print(f"{'safety':>8}{'ratio':>8}" + "".join(f"{s:>11}" for s in SCENARIOS)
              + f"{'score':>9}{'E[score]':>10}")
        safe_value = None
        for value in grid:
            pf = allocate(ps, pc, mult, value)
            ratio = C[np.arange(n), pf].sum() / C[:, 0].sum()
            busts, ev_plain = {}, 0.0
            for name in SCENARIOS:
                samples = small if name == "small" else full
                bust = 0
                for k, sample in enumerate(samples):
                    q = allocate(ps[sample], pc[sample], mult, value)
                    r = np.arange(len(sample))
                    cost = C[sample][r, q].copy()
                    light = C[sample][:, 0].sum()
                    if name == "runaway":
                        cost[hit_at[k] % len(sample)] += 0.065 * light
                    elif name == "inflation":
                        cost = cost * np.where(q == 2, 1.25, np.where(q == 1, 1.10, 1.0))
                    if cost.sum() / light > mult:
                        bust += 1
                    elif name == "plain":
                        ev_plain += S[sample][r, q].mean()
                busts[name] = bust / len(samples)
            score = S[np.arange(n), pf].mean()
            ev = ev_plain / len(full)
            worst = max(busts.values())
            if worst == 0.0:
                safe_value = (value, ratio, score, ev)
            if value in (0.55, 0.60, 0.65, 0.70, 0.73, 0.80, 0.85, 0.90, 0.94, 0.98) or worst == 0.0:
                mark = "  <- 전 시나리오 초과 0" if worst == 0.0 else ""
                print(f"{value:>8.2f}{ratio:>8.3f}"
                      + "".join(f"{busts[s]:>10.1%}" + " " for s in SCENARIOS)
                      + f"{score:>9.4f}{ev:>10.4f}{mark}")
        if safe_value is None:
            raise SystemExit(f"{tier}: no safety ratio on the grid avoids busting")
        chosen[tier] = safe_value[0]
        rows_out.append((tier, *safe_value))
        print(f"  -> {tier}: 최대 안전값 {safe_value[0]:.2f} "
              f"(사용률 {safe_value[1]:.3f}/{mult}, 점수 {safe_value[2]:.4f})")

    print("\n=== 선택된 트리플 ===")
    total_score = sum(WEIGHTS[tier] * score for tier, _v, _r, score, _e in rows_out)
    total_ev = sum(WEIGHTS[tier] * ev for tier, _v, _r, _s, ev in rows_out)
    for tier, value, ratio, score, ev in rows_out:
        print(f"  {tier:<10} {value:.2f}   사용률 {ratio:.3f}   점수 {score:.4f}   E[점수] {ev:.4f}")
    print(f"  가중 held-out {total_score:.6f}   가중 기대점수 {total_ev:.6f}")
    print(f"  (현행 {json.dumps(shipped)})")

    if a.write:
        target = json.loads(a.write.read_text(encoding="utf-8"))
        target["tier_safety_ratios"] = {t: float(v) for t, v in chosen.items()}
        a.write.write_text(json.dumps(target), encoding="utf-8")
        print(f"\n[price] wrote tier_safety_ratios={json.dumps(chosen)} into {a.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
