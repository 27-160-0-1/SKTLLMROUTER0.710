# SPDX-License-Identifier: Apache-2.0

"""Embed the offline difficulty prior as a prompt-hash lookup block.

The prior is produced by running a public open-weight model
(`skt/A.X-3.1-Light`, Apache-2.0) over the public benchmark sources with
`local-llm/run_llama.py`, judging each generation against the public gold
answers, and recording the per-item success rate and generation length.
CHALLENGE_RULES permits lookup tables and search indexes built from public data,
exact-prompt / prompt-hash lookup against public data, and offline use of models
whose weights are public.  The router performs no inference at evaluation time;
it reads this table and falls back to its usual path on a miss.

Run BEFORE build_meta_gbm.py (the meta heads consume the seven prior features).

    PYTHONPATH=src python tools/build_prior_lookup.py \
        --artifact src/ossp_router/resources/learned-router.v1.json \
        --labels local-llm/labels_axlight.jsonl \
        --items colab-label/bundle/all.jsonl colab-label/bundle/ext.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from ossp_router.learned_router import PRIOR_LOOKUP_HASH


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_column(paths, prompts, rounding):
    entries = {}
    by_family = defaultdict(list)
    missing = 0
    tag = None
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tag = tag or row.get("model", "unknown")
            text = prompts.get(row["id"])
            if text is None:
                missing += 1
                continue
            total = row.get("out_tokens_total")
            if total is None:
                total = sum(row.get("out_tokens", []) or [0])
            length = math.log1p(total / max(row.get("n", 4), 1))
            score = row.get("score")
            consistency = row.get("sc")
            value = (round(float(score), rounding) if score is not None else -1.0,
                     round(length, rounding),
                     round(float(consistency), rounding) if consistency is not None else -1.0)
            entries[_sha(text)] = list(value)
            by_family[row["family"]].append(value)

    def mean(rows, index):
        values = [row[index] for row in rows if row[index] >= 0.0]
        return round(sum(values) / len(values), rounding) if values else 0.0

    family_means = {name: [mean(rows, 0), mean(rows, 1), mean(rows, 2)]
                    for name, rows in by_family.items() if len(rows) >= 8}
    flat = [row for rows in by_family.values() for row in rows]
    global_mean = ([mean(flat, 0), mean(flat, 1), mean(flat, 2)] if flat else [0.0, 0.0, 0.0])
    return {"tag": tag or "unknown", "entries": entries,
            "family_means": family_means, "global_mean": global_mean}, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--column", action="append", nargs="+", required=True, metavar="LABELS",
                        help="one --column per model pass; repeat the flag, list its label files")
    parser.add_argument("--items", type=Path, nargs="+", required=True)
    parser.add_argument("--round", type=int, default=4)
    args = parser.parse_args()

    prompts = {}
    for path in args.items:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                prompts.setdefault(row["id"], row["prompt"])

    columns, dropped = [], 0
    for paths in args.column:
        column, missing = _load_column(paths, prompts, args.round)
        dropped += missing
        columns.append(column)
        print(f"  column '{column['tag']}': {len(column['entries'])} entries, "
              f"{len(column['family_means'])} family means")

    raw = json.loads(args.artifact.read_text(encoding="utf-8"))
    raw["prior_lookup"] = {
        "hash_algorithm": PRIOR_LOOKUP_HASH,
        "columns": columns,
        "provenance": {
            "models": [c["tag"] for c in columns],
            "sources": "public benchmark items rendered by colab-label/build_pool*.py",
            "judged_against": "data/gold/gold-answers.v1.json and the public source answers",
            "item_counts": [len(c["entries"]) for c in columns],
        },
    }
    args.artifact.write_text(json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
                             encoding="utf-8")
    print(f"OK: prior lookup with {len(columns)} column(s), "
          f"{dropped} labels without a prompt -> {args.artifact} "
          f"({args.artifact.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
