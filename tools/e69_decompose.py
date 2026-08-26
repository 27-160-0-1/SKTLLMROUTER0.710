# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""E69 step 1 -- decompose the remaining oracle gap into score-error vs cost-error, and
measure the score heads' calibration (the shrinkage E67 exposed, quantified).

E64 already showed: allocate on TRUE costs -> never busts at any safety, worth ~+0.022.
Nobody has measured the mirror: allocate on TRUE scores with predicted costs.  Together the
two isolate what better calibration of the score side could buy at the no-bust triple.

Also draws the reliability curve of the stacked score predictions (dev = honest out-of-sample,
train = in-sample for reference).  E67 saw light predicted 0.30-0.37 where truth is 0.03: if
that shrinkage is a stable monotone distortion, a fold-fitted monotone recalibration restores
the efficiency ranking without touching any model.

Usage: PYTHONPATH=src python tools/e69_decompose.py [--artifact A.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from bust_probability import allocate  # noqa: E402
from ossp_router import learned_router  # noqa: E402
from ossp_router.protocol import MODEL_IDS, TIERS, load_bundled_policy, load_input, load_outcomes  # noqa: E402

WEIGHTS = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
SAFETY = {"fast": 0.90, "balanced": 0.70, "premium": 0.56}


def load(split):
    policy = load_bundled_policy(); unit = Decimal(policy.token_unit)
    eps = list(load_input(ROOT / f"data/materialized/{split}/inputs.json").episodes)
    idx = {(o.episode_id, o.model_id): o for o in load_outcomes(ROOT / f"data/{split}/outcomes.json").outcomes}

    def tc(eid, mid):
        o = idx[(eid, mid)]; r = policy.models[mid]
        return float(r.fixed_cost + Decimal(o.input_tokens) * r.input_token_rate / unit
                     + Decimal(o.output_tokens) * r.output_token_rate / unit)

    C = np.array([[tc(e.episode_id, m) for m in MODEL_IDS] for e in eps])
    S = np.array([[float(idx[(e.episode_id, m)].score) for m in MODEL_IDS] for e in eps])
    return eps, S, C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=ROOT / "reports/e67_append/learned-router.v1.json")
    a = ap.parse_args()
    raw = json.loads(a.artifact.read_text(encoding="utf-8")); raw.pop("public_lookup", None)
    art = learned_router.parse_artifact(raw, base_path=a.artifact.parent)

    eps, S, C = load("dev")
    n = len(eps)
    P = {}
    for tier in TIERS:
        pr = [learned_router.predict_episode_augmented(e, art, tier) for e in eps]
        P[tier] = (np.array([[p[0][m] for m in MODEL_IDS] for p in pr]),
                   np.array([[p[1][m] for m in MODEL_IDS] for p in pr]))

    print("=== dev: EV at the no-bust triple, by which side is oracle ===")
    print(f"{'variant':<34}{'fast':>9}{'bal':>9}{'prem':>9}{'weighted':>10}")
    rows = np.arange(n)

    def ev(ps_by_tier, pc_by_tier, safety):
        out, tot = {}, 0.0
        for tier in TIERS:
            pk = allocate(ps_by_tier(tier), pc_by_tier(tier), MULTS[tier], safety[tier])
            got = S[rows, pk].mean()
            realized = C[rows, pk].sum() / C[:, 0].sum()
            got = 0.0 if realized > MULTS[tier] else got
            out[tier] = got; tot += WEIGHTS[tier] * got
        return out, tot

    variants = [
        ("predicted / predicted (deployed)", lambda t: P[t][0], lambda t: P[t][1], SAFETY),
        ("TRUE scores / predicted costs", lambda t: S, lambda t: P[t][1], SAFETY),
        ("predicted scores / TRUE costs", lambda t: P[t][0], lambda t: C, SAFETY),
        ("TRUE / TRUE (oracle @ triple)", lambda t: S, lambda t: C, SAFETY),
        ("TRUE / TRUE @ safety 1.0", lambda t: S, lambda t: C, {t: 1.0 for t in TIERS}),
    ]
    for name, fs, fc, saf in variants:
        out, tot = ev(fs, fc, saf)
        print(f"{name:<34}{out['fast']:>9.4f}{out['balanced']:>9.4f}{out['premium']:>9.4f}{tot:>10.4f}")

    # ---- reliability of the stacked scores (premium head; the tiers share the stack shape) ----
    print("\n=== reliability: predicted score bin -> realized mean (dev, premium stack) ===")
    ps = P["premium"][0]
    bins = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001])
    print(f"{'bin':<12}" + "".join(f"{m:>16}" for m in MODEL_IDS))
    for lo, hi in zip(bins[:-1], bins[1:]):
        row = f"{lo:.1f}-{hi:.1f}    "
        for j in range(3):
            sel = (ps[:, j] >= lo) & (ps[:, j] < hi)
            row += (f"{S[sel, j].mean():>9.3f} n={sel.sum():<4}" if sel.sum() else f"{'—':>9}      ")
        print(row)

    # train side, same artifact (in-sample -- only to see whether the distortion shape matches)
    eps_t, S_t, _ = load("train")
    pr_t = [learned_router.predict_episode_augmented(e, art, "premium") for e in eps_t]
    ps_t = np.array([[p[0][m] for m in MODEL_IDS] for p in pr_t])
    print("\n=== same curve on train (in-sample; shape comparison only) ===")
    for lo, hi in zip(bins[:-1], bins[1:]):
        row = f"{lo:.1f}-{hi:.1f}    "
        for j in range(3):
            sel = (ps_t[:, j] >= lo) & (ps_t[:, j] < hi)
            row += (f"{S_t[sel, j].mean():>9.3f} n={sel.sum():<4}" if sel.sum() else f"{'—':>9}      ")
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
