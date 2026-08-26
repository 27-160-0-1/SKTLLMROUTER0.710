# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Pack the E66b Colab bundle: fill the reasoning column's coverage gap.

Carries `reason_covered_digests.txt` so the 1,951 prompts E66 already labelled are skipped and
only the remaining 689 public prompts are run.  The source-rendered pool is not shipped -- the
optional cell 6 regenerates it, and that pass is for private-set coverage, not the dev measurement.

Usage: python colab-label/make_e66b_zip.py
"""
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "colab-label/e66b_colab_bundle.zip"

wanted = [
    *[p for p in (REPO / "src/ossp_router").glob("*.py")],
    REPO / "src/ossp_router/resources/__init__.py",
    REPO / "src/ossp_router/resources/hash-regex-public.v1.json",
    REPO / "src/ossp_router/resources/routing-policy.v1.json",
    REPO / "data/materialized/train/inputs.json",
    REPO / "data/materialized/dev/inputs.json",
    REPO / "data/train/outcomes.json",
    REPO / "data/dev/outcomes.json",
    REPO / "data/gold/gold-answers.v1.json",
    REPO / "colab-label/judge.py",
    REPO / "colab-label/run_labels.py",
    REPO / "colab-label/build_public_all.py",
    REPO / "colab-label/build_pool.py",
    REPO / "colab-label/build_pool_ext.py",
    REPO / "colab-label/think_cost_gate.py",
    REPO / "colab-label/reason_covered_digests.txt",
]
missing = [p for p in wanted if not p.exists()]
if missing:
    raise SystemExit("missing:\n  " + "\n  ".join(str(p) for p in missing))
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in wanted:
        z.write(p, f"official-router/{p.relative_to(REPO).as_posix()}")
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {len(wanted)} files)")
