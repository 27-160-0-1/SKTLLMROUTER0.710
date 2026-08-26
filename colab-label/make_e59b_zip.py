# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Pack the E59b Colab bundle (coverage fill-in for the 34B mid column).

Same contents as the E59 bundle plus `build_public_all.py`.  The item pool is not shipped --
the notebook restores it from `e59_out.zip` (which already contains bundle/*.jsonl), so E59b
never re-renders the sources.

Usage: python colab-label/make_e59b_zip.py
"""
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "colab-label/e59b_colab_bundle.zip"

wanted = [
    *[p for p in (REPO / "src/ossp_router").glob("*.py")],
    REPO / "src/ossp_router/resources/__init__.py",
    REPO / "src/ossp_router/resources/hash-regex-public.v1.json",
    REPO / "src/ossp_router/resources/learned-router.v1.json",
    REPO / "src/ossp_router/resources/routing-policy.v1.json",
    REPO / "data/materialized/train/inputs.json",
    REPO / "data/materialized/dev/inputs.json",
    REPO / "data/train/outcomes.json",
    REPO / "data/dev/outcomes.json",
    REPO / "data/gold/gold-answers.v1.json",
    REPO / "colab-label/judge.py",
    REPO / "colab-label/run_labels.py",
    REPO / "colab-label/build_public_all.py",
    REPO / "colab-label/build_pool_aime.py",
    REPO / "colab-label/covered_digests.txt",
    REPO / "colab-label/bundle/aime.jsonl",
    REPO / "data/train/aime-selection.json",
    REPO / "data/dev/aime-selection.json",
    REPO / "colab-label/to_prior_labels.py",
    REPO / "colab-label/prior_column_report.py",
]
missing = [p for p in wanted if not p.exists()]
if missing:
    raise SystemExit("missing:\n  " + "\n  ".join(str(p) for p in missing))
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in wanted:
        z.write(p, f"official-router/{p.relative_to(REPO).as_posix()}")
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {len(wanted)} files)")
