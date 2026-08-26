# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""E69 final gate -- paired comparison of two PACKAGES (artifact + its own safety triple).

`e67_paired.py` applies one triple to both artifacts; here each side carries its own, because
the blend changes the certified triple (.90/.70/.56 without it, .90/.72/.52 with it) and the
honest question is whether the whole shipped package improves.  Stem-grouped resamples, both
packages scored on the same resample; a busted tier contributes 0.

Usage:
    PYTHONPATH=src python tools/e69_package_paired.py \
        --a OLD.json --sa 0.90,0.70,0.56 --b NEW.json --sb 0.90,0.72,0.52 [--boot 1500]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from bust_probability import allocate  # noqa: E402
from ossp_router import learned_router  # noqa: E402
from ossp_router.protocol import MODEL_IDS, TIERS, load_bundled_policy, load_input, load_outcomes  # noqa: E402

WEIGHTS = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}


def package(path, episodes):
    raw = json.loads(path.read_text(encoding="utf-8")); raw.pop("public_lookup", None)
    art = learned_router.parse_artifact(raw, base_path=path.parent)
    out = {}
    for tier in TIERS:
        pr = [learned_router.predict_episode_augmented(e, art, tier) for e in episodes]
        out[tier] = (np.array([[p[0][m] for m in MODEL_IDS] for p in pr]),
                     np.array([[p[1][m] for m in MODEL_IDS] for p in pr]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--sa", required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--sb", required=True)
    ap.add_argument("--boot", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    SA = dict(zip(TIERS, (float(x) for x in a.sa.split(","))))
    SB = dict(zip(TIERS, (float(x) for x in a.sb.split(","))))

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

    stem = {}
    with (ROOT / "analysis/episodes.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem[row["episode_id"]] = row.get("stem_id") or row["episode_id"]
    groups = defaultdict(list)
    for i, e in enumerate(episodes):
        groups[stem.get(e.episode_id, e.episode_id)].append(i)
    glist = list(groups.values())

    PA = package(a.a, episodes); PB = package(a.b, episodes)
    rows = np.arange(n)

    def full(P, saf):
        tot = 0.0
        for tier in TIERS:
            pk = allocate(P[tier][0], P[tier][1], MULTS[tier], saf[tier])
            got = S[rows, pk].mean()
            if C[rows, pk].sum() / C[:, 0].sum() > MULTS[tier]:
                got = 0.0
            tot += WEIGHTS[tier] * got
        return tot

    fa, fb = full(PA, SA), full(PB, SB)
    print(f"[pkg] full dev:  A({a.sa}) {fa:.6f}   B({a.sb}) {fb:.6f}   delta {fb-fa:+.6f}")

    rng = np.random.default_rng(a.seed)
    deltas = []; busts = {"A": 0, "B": 0}
    for _ in range(a.boot):
        pick = rng.integers(0, len(glist), size=len(glist))
        s = np.concatenate([glist[g] for g in pick]); r = np.arange(len(s))
        vals = {}
        for label, P, saf in (("A", PA, SA), ("B", PB, SB)):
            tot = 0.0
            for tier in TIERS:
                pk = allocate(P[tier][0][s], P[tier][1][s], MULTS[tier], saf[tier])
                if C[s][r, pk].sum() / C[s][:, 0].sum() > MULTS[tier]:
                    busts[label] += 1
                else:
                    tot += WEIGHTS[tier] * S[s][r, pk].mean()
            vals[label] = tot
        deltas.append(vals["B"] - vals["A"])
    d = np.array(deltas); lo, hi = np.quantile(d, [0.05, 0.95])
    print(f"[pkg] B - A over {a.boot} stem-grouped paired resamples:")
    print(f"   mean {d.mean():+.6f}   90% CI [{lo:+.6f}, {hi:+.6f}]   P(B>A) {np.mean(d > 0):.3f}")
    print(f"   busts: A {busts['A']}  B {busts['B']}  / {3*a.boot} tier-resamples")
    print("   -> " + ("ADOPT" if lo > 0 else "noise" if np.mean(d > 0) < 0.8 else "lean-B, not significant"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
