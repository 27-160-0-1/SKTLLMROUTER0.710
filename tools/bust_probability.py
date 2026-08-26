# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Per-tier bust probability and expected score for a built artifact.

A tier scores 0 outright if its realised cost ratio exceeds the budget multiplier, so the safety
ratio is a bet: too high and the tier busts, too low and score is left unspent.  The reported
held-out number assumes the bet is won; this estimates how often it is.

The allocator is re-run inside every resample.  That matters: `select_models` sizes its cap from
the batch it is given, so a router facing a different item mix re-balances rather than keeping
the picks it made for the held-out set.  Holding the picks fixed and resampling them -- the
obvious way to write this -- overstates the risk badly.

Scenarios, following E55:
  plain     -- bootstrap resamples of the held-out set
  runaway   -- plus one injected pathological upgrade worth 6.5 % of the light baseline
               (b01: 82 % of the fast tier's ratio variance on dev came from one episode)
  inflation -- plus a systematic cost surprise, 1.25x on axk1-think and 1.10x on ax31

Usage:
    PYTHONPATH=src python tools/bust_probability.py --artifact A.json [--boot 1500]
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
    """Vectorised twin of learned_router.select_models: Lagrangian bisection on predictions."""
    light_total = pc[:, 0].sum()
    cap = light_total * max(1.0, mult * safety)
    tie = np.array([2e-12, 1e-12, 0.0])          # matches the -MODEL_IDS.index tie-break

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
    ap.add_argument("--boot", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sweep", default=None,
                    help="tier:v1,v2,... -- re-price one tier's safety ratio without rebuilding "
                         "(safety only enters allocation, never the fitted model)")
    a = ap.parse_args()

    policy = load_bundled_policy()
    raw = json.loads(a.artifact.read_text(encoding="utf-8"))
    raw.pop("public_lookup", None)
    artifact = learned_router.parse_artifact(raw, base_path=a.artifact.parent)
    safety = raw.get("tier_safety_ratios") or {}
    episodes = list(load_input(a.input).episodes)
    index = {(o.episode_id, o.model_id): o for o in load_outcomes(a.outcomes).outcomes}
    n = len(episodes)
    unit = Decimal(policy.token_unit)

    def true_cost(eid, mid):
        o = index[(eid, mid)]
        r = policy.models[mid]
        return float(r.fixed_cost + Decimal(o.input_tokens) * r.input_token_rate / unit
                     + Decimal(o.output_tokens) * r.output_token_rate / unit)

    true_c = np.array([[true_cost(e.episode_id, m) for m in MODEL_IDS] for e in episodes])
    true_s = np.array([[float(index[(e.episode_id, m)].score) for m in MODEL_IDS] for e in episodes])

    print(f"[bust] {a.artifact.parent.name}: {n} episodes, safety={safety}, {a.boot} resamples")
    print(f"{'tier':<10}{'mult':>6}{'safety':>8}{'ratio':>9}"
          f"{'pass plain':>12}{'runaway':>10}{'inflation':>11}{'score':>9}{'E[score]':>11}")

    rng = np.random.default_rng(a.seed)
    samples = [rng.integers(0, n, size=n) for _ in range(a.boot)]
    runaway_at = [rng.integers(0, n) for _ in range(a.boot)]
    totals = {"reported": 0.0, "plain": 0.0, "runaway": 0.0, "inflation": 0.0}

    sweep_tier, sweep_values = None, []
    if a.sweep:
        sweep_tier, raw_values = a.sweep.split(":")
        sweep_values = [float(v) for v in raw_values.split(",")]

    for tier in TIERS:
        mult = float(policy.tiers[tier].budget_multiplier)
        s_ratio = float(safety.get(tier, 1.0))
        preds = [learned_router.predict_episode_augmented(e, artifact, tier) for e in episodes]
        ps = np.array([[p[0][m] for m in MODEL_IDS] for p in preds])
        pc = np.array([[p[1][m] for m in MODEL_IDS] for p in preds])

        pick_full = allocate(ps, pc, mult, s_ratio)
        rows = np.arange(n)
        full_ratio = true_c[rows, pick_full].sum() / true_c[:, 0].sum()
        reported = true_s[rows, pick_full].mean()

        res, evs = {}, {}
        for name in ("plain", "runaway", "inflation"):
            bust, ev = 0, 0.0
            for k, sample in enumerate(samples):
                p = allocate(ps[sample], pc[sample], mult, s_ratio)   # the router re-balances
                r = np.arange(len(sample))
                chosen = true_c[sample][r, p].copy()
                light = true_c[sample][:, 0].sum()
                if name == "runaway":
                    chosen[runaway_at[k] % len(sample)] += 0.065 * light
                elif name == "inflation":
                    chosen = chosen * np.where(p == 2, 1.25, np.where(p == 1, 1.10, 1.0))
                if chosen.sum() / light > mult:
                    bust += 1
                else:
                    ev += true_s[sample][r, p].mean()
            res[name] = bust / len(samples)
            evs[name] = ev / len(samples)

        if tier == sweep_tier:
            for value in sweep_values:
                pf = allocate(ps, pc, mult, value)
                fr = true_c[rows, pf].sum() / true_c[:, 0].sum()
                rep = true_s[rows, pf].mean()
                sb, sev = 0, 0.0
                for sample in samples:
                    q = allocate(ps[sample], pc[sample], mult, value)
                    rr = np.arange(len(sample))
                    if true_c[sample][rr, q].sum() / true_c[sample][:, 0].sum() > mult:
                        sb += 1
                    else:
                        sev += true_s[sample][rr, q].mean()
                print(f"  {tier}@{value:<5.3f}{'':>3}{mult:>6.2f}{value:>8.3f}{fr:>9.3f}"
                      f"{1-sb/len(samples):>11.1%}{'':>10}{'':>11}{rep:>9.4f}"
                      f"{sev/len(samples):>11.4f}   가중기여 {WEIGHTS[tier]*sev/len(samples):.4f}")
        print(f"{tier:<10}{mult:>6.2f}{s_ratio:>8.3f}{full_ratio:>9.3f}"
              f"{1-res['plain']:>11.1%}{1-res['runaway']:>10.1%}{1-res['inflation']:>11.1%}"
              f"{reported:>9.4f}{evs['plain']:>11.4f}")
        totals["reported"] += WEIGHTS[tier] * reported
        for name in ("plain", "runaway", "inflation"):
            totals[name] += WEIGHTS[tier] * evs[name]

    print(f"\n  held-out dev (초과 없다고 가정)  {totals['reported']:.6f}")
    print(f"  기대점수  plain {totals['plain']:.6f}  runaway {totals['runaway']:.6f}"
          f"  inflation {totals['inflation']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
