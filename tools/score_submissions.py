# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Score three tier submission files with the official scorer.

The container writes one submission per tier; this feeds all three to
`ossp_router.scoring.score_submissions` -- the same function the challenge uses -- and prints
the per-tier report plus the weighted final score.

Usage:
    PYTHONPATH=src python tools/score_submissions.py \
        --inputs data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json \
        --submissions out/submission-fast.json out/submission-balanced.json out/submission-premium.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ossp_router.protocol import TIERS, load_bundled_policy, load_input, load_outcomes, load_submission
from ossp_router.scoring import score_submissions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--outcomes", type=Path, required=True)
    ap.add_argument("--submissions", type=Path, nargs=3, required=True,
                    help="three submission files, any order (tier is read from each file)")
    a = ap.parse_args()

    policy = load_bundled_policy()
    inputs = load_input(a.inputs)
    outcomes = load_outcomes(a.outcomes)
    submissions = [load_submission(p) for p in a.submissions]

    report = score_submissions(inputs, outcomes, submissions, policy)
    print(f"split={report['split']}  episodes={len(inputs.episodes)}")
    for tier in TIERS:
        t = report["tiers"][tier]
        print(f"  {tier:<9} score={t['tier_score']}  budget_ratio={t['budget_ratio']}  "
              f"passed={t['budget_passed']}")
    print(f"FINAL SCORE = {report['final_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
