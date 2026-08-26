# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Report budget usage for a single tier submission, without reporting its accuracy.

`ossp_router.scoring.score_submissions` insists on one submission per tier, so this calls the
same per-tier routine it calls internally -- the cost arithmetic is the official one.

It exists for the submission artifact, which is fitted on Train+Dev: its Dev accuracy is
in-sample and must not be quoted, but whether it stays inside the cost budget is still a
meaningful thing to check.

Usage:
    PYTHONPATH=src python tools/budget_check.py \
        --inputs data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json \
        --submission out/submission-premium.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ossp_router.protocol import (
    load_bundled_policy,
    load_input,
    load_outcomes,
    load_submission,
    policy_sha256,
)
from ossp_router.scoring import _outcome_index, _score_tier


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--outcomes", type=Path, required=True)
    ap.add_argument("--submission", type=Path, required=True)
    a = ap.parse_args()

    policy = load_bundled_policy()
    inputs = load_input(a.inputs)
    outcomes = load_outcomes(a.outcomes)
    submission = load_submission(a.submission)

    report = _score_tier(
        inputs=inputs,
        submission=submission,
        outcome_by_key=_outcome_index(inputs, outcomes, policy),
        policy=policy,
        policy_digest=policy_sha256(policy),
    )
    print(
        f"  tier={submission.tier}  budget_ratio={report['budget_ratio']}"
        f"  limit={report['budget_multiplier']}  passed={report['budget_passed']}"
    )
    print("  (accuracy deliberately not reported -- see the tool's docstring)")
    return 0 if report["budget_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
