# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Per-family (and per-tier) breakdown of a held-out Dev run, for two artifacts side by side.

The aggregate Dev number hides where a change acted.  AIME is 12 of 880 dev episodes, so a large
improvement there is invisible in the total -- but it is exactly the family the E59b work
targeted, and the private set will contain AIME items the public split never had.

Usage:
    PYTHONPATH=src python tools/holdout_by_family.py --artifact A.json B.json [--labels e59 e59b]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ossp_router import learned_router, similarity
from ossp_router.heuristic import episode_text
from ossp_router.protocol import TIERS, load_bundled_policy, load_input, load_outcomes


def picks_for(artifact_path: Path, inputs, policy):
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    raw.pop("public_lookup", None)
    artifact = learned_router.parse_artifact(raw, base_path=artifact_path.parent)
    safety = raw.get("tier_safety_ratios") or {}
    out = {}
    for tier in TIERS:
        preds = [learned_router.predict_episode_augmented(e, artifact, tier) for e in inputs.episodes]
        selected, ratio = learned_router.select_models(
            [p[0] for p in preds], [p[1] for p in preds],
            budget_multiplier=float(policy.tiers[tier].budget_multiplier),
            safety_ratio=float(safety.get(tier, 1.0)))
        out[tier] = (list(selected), ratio)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--input", type=Path, default=Path("data/materialized/dev/inputs.json"))
    ap.add_argument("--outcomes", type=Path, default=Path("data/dev/outcomes.json"))
    ap.add_argument("--aime-selection", type=Path, default=Path("data/dev/aime-selection.json"))
    a = ap.parse_args()
    labels = a.labels or [p.parent.name for p in a.artifact]

    policy = load_bundled_policy()
    inputs = load_input(a.input)
    outcomes = load_outcomes(a.outcomes)
    index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    episodes = list(inputs.episodes)
    fam = [similarity.classify_family(episode_text(e)) for e in episodes]
    aime_ids = set()
    if a.aime_selection.exists():
        aime_ids = {e["episode_id"] for e in json.loads(a.aime_selection.read_text(encoding="utf-8"))["episodes"]}
    groups = {f: [i for i, x in enumerate(fam) if x == f] for f in sorted(set(fam))}
    if aime_ids:
        groups["** organiser AIME **"] = [i for i, e in enumerate(episodes) if e.episode_id in aime_ids]

    runs = {label: picks_for(path, inputs, policy) for label, path in zip(labels, a.artifact)}

    for tier in TIERS:
        print(f"\n=== tier {tier} ===")
        head = f"{'family':<22}{'n':>5}"
        for label in labels:
            head += f"{label:>12}"
        print(head + f"{'delta':>9}")
        for name, idx in groups.items():
            if not idx:
                continue
            row = f"{name:<22}{len(idx):>5}"
            means = []
            for label in labels:
                sel = runs[label][tier][0]
                m = sum(float(index[(episodes[i].episode_id, sel[i])].score) for i in idx) / len(idx)
                means.append(m)
                row += f"{m:>12.4f}"
            row += f"{means[-1] - means[0]:>+9.4f}" if len(means) > 1 else ""
            print(row)
        for label in labels:
            print(f"   [{label}] budget ratio {runs[label][tier][1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
