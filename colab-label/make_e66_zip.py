# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Pack the E66 Colab bundle: reasoning-model prior for think's cost.

Only the gate needs to run first, so the bundle carries `bundle/pilot.jsonl` (the public 2,640)
and the pool builders -- not the 92 MB pool, which is only regenerated if the gate passes.

Usage: python colab-label/make_e66_zip.py
"""
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "colab-label/e66_colab_bundle.zip"

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
    REPO / "colab-label/build_pool.py",
    REPO / "colab-label/build_pool_ext.py",
    REPO / "colab-label/think_cost_gate.py",
    REPO / "colab-label/bundle/pilot.jsonl",
]
missing = [p for p in wanted if not p.exists()]
if missing:
    raise SystemExit("missing:\n  " + "\n  ".join(str(p) for p in missing))
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in wanted:
        z.write(p, f"official-router/{p.relative_to(REPO).as_posix()}")
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {len(wanted)} files)")
