# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Pack the E60 Colab bundle: the repository's own GPU build chain, no CPU substitution.

Everything needed to rebuild Train-only and score Dev once, plus the 34B prior column already
compiled (`prior_column_c.json`, 3.5 MB) so the 100 MB of item pools never has to be uploaded.

Usage: python colab-label/make_e60_zip.py
"""
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "colab-label/e60_colab_bundle.zip"

wanted = [
    *[p for p in (REPO / "src/ossp_router").glob("*.py")],
    REPO / "src/ossp_router/resources/__init__.py",
    REPO / "src/ossp_router/resources/hash-regex-public.v1.json",
    REPO / "src/ossp_router/resources/learned-router.v1.json",   # holds the shipped columns A and B
    REPO / "src/ossp_router/resources/routing-policy.v1.json",
    REPO / "data/materialized/train/inputs.json",
    REPO / "data/materialized/dev/inputs.json",
    REPO / "data/train/outcomes.json",
    REPO / "data/dev/outcomes.json",
    REPO / "data/train/aime-selection.json",
    REPO / "data/dev/aime-selection.json",
    REPO / "tools/train_learned_router_gpu.py",
    REPO / "tools/build_router_augmentation.py",
    REPO / "tools/build_meta_gbm.py",
    REPO / "tools/build_prior_lookup.py",
    REPO / "tools/splice_prior_column.py",
    REPO / "tools/inject_prior.py",
    REPO / "tools/holdout_eval.py",
    REPO / "tools/holdout_by_family.py",
    REPO / "colab-label/prior_column_c.json",
    REPO / "run_repo_chain.sh",
]
missing = [p for p in wanted if not p.exists()]
if missing:
    raise SystemExit("missing:\n  " + "\n  ".join(str(p) for p in missing))
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in wanted:
        z.write(p, f"official-router/{p.relative_to(REPO).as_posix()}")
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {len(wanted)} files)")
