# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Copy the compiled prior_lookup block from a reference artifact into a freshly built one.

The raw prior labels (local-llm/*.jsonl) are not in the repository, only the compiled table
inside the shipped artifact.  The table is derived from public benchmark items scored by
open-weight models against public gold answers -- it contains no organiser label and no Dev
outcome -- so copying it into a Train-only rebuild does not leak the held-out set.

Usage: python tools/inject_prior.py --artifact <new> --reference <shipped> [--drop]
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--artifact", type=Path, required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--drop", action="store_true", help="remove the prior instead of copying it (control arm)")
a = p.parse_args()

art = json.loads(a.artifact.read_text(encoding="utf-8"))
if a.drop:
    art.pop("prior_lookup", None)
    print("[inject_prior] prior_lookup removed (no-prior control)")
else:
    ref = json.loads(a.reference.read_text(encoding="utf-8"))
    block = ref.get("prior_lookup")
    if block is None:
        raise SystemExit("reference artifact has no prior_lookup")
    art["prior_lookup"] = block
    counts = [len(c.get("entries", {})) for c in block["columns"]]
    print(f"[inject_prior] copied prior_lookup: {len(block['columns'])} columns, entries {counts}")
a.artifact.write_text(json.dumps(art), encoding="utf-8")
