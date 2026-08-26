# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Add (or substitute) a prior column in an artifact, reusing the already-compiled columns.

`build_prior_lookup.py` compiles every column from raw label files, but the raw labels for
columns A and B are not in the repository -- only the compiled tables inside the shipped
artifact.  This script compiles just the new column with the same code path
(`build_prior_lookup._load_column`, so the value/rounding/family-mean conventions are
identical) and splices it into the existing column list.

Column order is semantic: the feature block appends the deltas between *consecutive* columns,
which stand in for the upgrade gain the allocator ranks.  Keep them ordered weak -> strong.

    --mode append   [A, B] -> [A, B, C]     (deltas B-A and C-B)
    --mode replace  [A, B] -> [A, C]        (delta C-A; drops the weak Qwen proxy)

Usage:
    PYTHONPATH=src python tools/splice_prior_column.py \
        --artifact build/learned-router.v1.json --reference src/.../learned-router.v1.json \
        --labels colab-label/out/labels_mid_pool.jsonl colab-label/out/labels_gate.jsonl \
        --items colab-label/bundle/all.jsonl colab-label/bundle/ext.jsonl \
        --mode append
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from build_prior_lookup import _load_column  # noqa: E402
from ossp_router.learned_router import PRIOR_LOOKUP_HASH  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True, help="artifact to modify in place")
    ap.add_argument("--reference", type=Path, required=True, help="artifact holding the compiled columns A/B")
    ap.add_argument("--labels", type=Path, nargs="+", default=None)
    ap.add_argument("--items", type=Path, nargs="+", default=None)
    ap.add_argument("--column-json", type=Path, default=None,
                    help="a column already compiled by --dump-column (skips labels/items entirely)")
    ap.add_argument("--dump-column", type=Path, default=None,
                    help="write the compiled column here and exit; lets a machine without the "
                         "item pools (100 MB of jsonl) still splice it")
    ap.add_argument("--mode", choices=["append", "replace", "only"], default="append",
                    help="append: keep every existing column; replace: drop the last one; only: this column alone")
    ap.add_argument("--round", type=int, default=4)
    a = ap.parse_args()

    if a.column_json:
        column = json.loads(a.column_json.read_text(encoding="utf-8"))
        print(f"[splice] loaded compiled column '{column['tag']}': {len(column['entries'])} entries")
        return _splice(a, column)

    prompts = {}
    for path in a.items:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                prompts[row["id"]] = row["prompt"]
    print(f"[splice] item pool {len(prompts)} prompts")

    column, missing = _load_column([str(p) for p in a.labels], prompts, a.round)
    print(f"[splice] new column '{column['tag']}': {len(column['entries'])} entries "
          f"({missing} label rows had no matching prompt)")

    if a.dump_column:
        a.dump_column.write_text(json.dumps(column), encoding="utf-8")
        print(f"[splice] wrote compiled column -> {a.dump_column} "
              f"({a.dump_column.stat().st_size/1e6:.1f} MB)")
        return 0
    return _splice(a, column)


def _splice(a, column) -> int:
    ref = json.loads(a.reference.read_text(encoding="utf-8"))
    existing = list((ref.get("prior_lookup") or {}).get("columns", []))
    if a.mode == "only":
        columns = [column]
    elif a.mode == "replace":
        columns = existing[:-1] + [column]
    else:
        columns = existing + [column]

    art = json.loads(a.artifact.read_text(encoding="utf-8"))
    provenance = dict((ref.get("prior_lookup") or {}).get("provenance", {}))
    provenance["models"] = [c.get("tag") for c in columns]
    provenance["item_counts"] = [len(c["entries"]) for c in columns]
    art["prior_lookup"] = {"hash_algorithm": PRIOR_LOOKUP_HASH, "columns": columns, "provenance": provenance}
    a.artifact.write_text(json.dumps(art), encoding="utf-8")
    print(f"[splice] mode={a.mode} -> columns {[c.get('tag') for c in columns]} "
          f"sizes {[len(c['entries']) for c in columns]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
