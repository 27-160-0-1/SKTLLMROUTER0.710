# SPDX-License-Identifier: Apache-2.0
"""Paired comparison of two artifacts on Dev under the no-bust triple.

A single Dev number hides whether a +0.0005 is one item flipping.  This resamples Dev and
scores BOTH artifacts on the same resample, so the difference distribution is paired and the
item-mix noise cancels.  Resampling is done by `stem_id` group (analysis §3.3: 430 groups share
a passage or function; resampling items independently understates the variance).

Reports: mean paired delta, its bootstrap 90 % interval, and P(delta > 0).

Usage:
    PYTHONPATH=src python tools/e67_paired.py --a A.json --b B.json [--boot 1500]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

from ossp_router import learned_router
from ossp_router.protocol import MODEL_IDS, TIERS, load_bundled_policy, load_input, load_outcomes

ROOT = Path(__file__).resolve().parent.parent
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


def predictions(path, episodes):
    raw = json.loads(path.read_text(encoding="utf-8")); raw.pop("public_lookup", None)
    art = learned_router.parse_artifact(raw, base_path=path.parent)
    out = {}
    for tier in TIERS:
        pr = [learned_router.predict_episode_augmented(e, art, tier) for e in episodes]
        out[tier] = (np.array([[p[0][m] for m in MODEL_IDS] for p in pr]),
                     np.array([[p[1][m] for m in MODEL_IDS] for p in pr]))
    return out, raw.get("tier_safety_ratios") or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--safety", default="0.90,0.70,0.56", help="fast,balanced,premium applied to both")
    ap.add_argument("--boot", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    saf = dict(zip(TIERS, (float(x) for x in a.safety.split(","))))

    policy = load_bundled_policy(); unit = Decimal(policy.token_unit)
    episodes = list(load_input(ROOT / "data/materialized/dev/inputs.json").episodes)
    idx = {(o.episode_id, o.model_id): o for o in load_outcomes(ROOT / "data/dev/outcomes.json").outcomes}
    n = len(episodes)

    def tc(eid, mid):
        o = idx[(eid, mid)]; r = policy.models[mid]
        return float(r.fixed_cost + Decimal(o.input_tokens) * r.input_token_rate / unit
                     + Decimal(o.output_tokens) * r.output_token_rate / unit)

    C = np.array([[tc(e.episode_id, m) for m in MODEL_IDS] for e in episodes])
    S = np.array([[float(idx[(e.episode_id, m)].score) for m in MODEL_IDS] for e in episodes])

    # stem groups from the analysis table; singletons are their own group
    stem = {}
    with (ROOT / "analysis/episodes.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem[row["episode_id"]] = row.get("stem_id") or row["episode_id"]
    groups = defaultdict(list)
    for i, e in enumerate(episodes):
        groups[stem.get(e.episode_id, e.episode_id)].append(i)
    glist = list(groups.values())
    print(f"[paired] dev {n} items in {len(glist)} stem groups")

    PA, _ = predictions(a.a, episodes)
    PB, _ = predictions(a.b, episodes)

    rng = np.random.default_rng(a.seed)
    rows = np.arange(n)
    full = {}
    for label, P in (("A", PA), ("B", PB)):
        tot = 0.0
        for tier, mult in (("fast", 1.25), ("balanced", 2.0), ("premium", 4.0)):
            pk = allocate(P[tier][0], P[tier][1], mult, saf[tier])
            tot += WEIGHTS[tier] * S[rows, pk].mean()
        full[label] = tot
    print(f"[paired] full dev @ {a.safety}:  A {full['A']:.6f}   B {full['B']:.6f}   delta {full['B']-full['A']:+.6f}")

    deltas, busts = [], {"A": 0, "B": 0}
    for _ in range(a.boot):
        pick_groups = rng.integers(0, len(glist), size=len(glist))
        s = np.concatenate([glist[g] for g in pick_groups])
        r = np.arange(len(s))
        val = {}
        for label, P in (("A", PA), ("B", PB)):
            tot = 0.0
            for tier, mult in (("fast", 1.25), ("balanced", 2.0), ("premium", 4.0)):
                pk = allocate(P[tier][0][s], P[tier][1][s], mult, saf[tier])
                if C[s][r, pk].sum() / C[s][:, 0].sum() > mult:
                    busts[label] += 1
                else:
                    tot += WEIGHTS[tier] * S[s][r, pk].mean()
            val[label] = tot
        deltas.append(val["B"] - val["A"])
    d = np.array(deltas)
    lo, hi = np.quantile(d, [0.05, 0.95])
    print(f"[paired] B - A over {a.boot} stem-grouped resamples:")
    print(f"   mean {d.mean():+.6f}   90% CI [{lo:+.6f}, {hi:+.6f}]   P(B>A) {np.mean(d > 0):.3f}")
    print(f"   busts: A {busts['A']}  B {busts['B']}  (tier-resamples out of {3*a.boot})")
    verdict = ("ADOPT" if lo > 0 else "noise" if np.mean(d > 0) < 0.8 else "lean-B, not significant")
    print(f"   -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
