# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Convert `run_labels.py` output rows into the row shape `tools/build_prior_lookup.py` consumes.

`run_labels.py` writes {id, family, score, n, out_per_gen, gens:[{text, tokens, correct}]}.
`build_prior_lookup.py` reads {id, family, score, n, out_tokens_total, sc}, where

  * `score` is None (-> -1.0 in the table) when the item has no public gold answer, so that
    items we cannot judge still contribute length and consistency, and
  * `sc` is self-consistency: the share of the n generations whose extracted final answer
    equals the modal answer.  It needs no gold answer, which is what lifted E53's Dev
    coverage from 0.69 to 0.91.

Usage:
    python to_prior_labels.py IN.jsonl OUT.jsonl [--model TAG]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from judge import final_answer  # noqa: E402


def self_consistency(gens) -> float | None:
    answers = [final_answer(g.get("text", "")) for g in gens]
    answers = [a for a in answers if a not in ("", None)]
    if len(answers) < 2:
        return None
    modal = Counter(answers).most_common(1)[0][1]
    return modal / len(gens)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--model", default=None, help="tag written to the `model` field (column name)")
    ap.add_argument("--length-only", action="store_true",
                    help="drop the score, keeping length and self-consistency.  For a column "
                         "whose value is cost prediction: the reasoning model agrees with the "
                         "organiser's grading only at corr 0.363 (ruletaker 0.09), and E42's "
                         "lesson is that a weak score head can cost EV even while it looks "
                         "informative.  Its output length is the part we are buying.")
    a = ap.parse_args()

    out, n_scored, n_sc = [], 0, 0
    for line in a.src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gens = r.get("gens") or []
        judged = any(g.get("correct") is not None for g in gens)
        total = sum(int(g.get("tokens", 0)) for g in gens)
        sc = self_consistency(gens)
        n_scored += bool(judged)
        n_sc += sc is not None
        out.append({
            "id": r["id"],
            "family": r["family"],
            "source": r.get("source"),
            "model": a.model or r.get("model", "unknown"),
            "n": r.get("n", len(gens)),
            "score": (None if a.length_only else
                      (float(r["score"]) if judged and r.get("score") is not None else None)),
            "out_tokens_total": total,
            "sc": sc,
        })
    a.dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    print(f"[to_prior_labels] {len(out)} rows -> {a.dst}  (with score {n_scored}, with consistency {n_sc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
