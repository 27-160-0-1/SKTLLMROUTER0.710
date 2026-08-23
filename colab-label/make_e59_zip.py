# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Pack the E59 Colab GPU bundle (real skt/A.X-3.1 mid prior column).

Contents: the runtime package + shipped artifact (needed for the digest self-check and the
fidelity gate), the public Train/Dev inputs and outcomes (the gate's ground truth), and the
pool builders / labeller / converter / report scripts.  The item pool itself is NOT shipped --
the notebook regenerates it from the public sources so the prompt hashes line up.

Usage: python colab-label/make_e59_zip.py [--repo PATH]
    --repo points at the 0705-line checkout that has build_pool_ext.py, run_labels.py and the
    prior-bearing artifact; defaults to this repository.
"""
import argparse
import zipfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "e59_colab_bundle.zip")
a = ap.parse_args()
REPO = a.repo.resolve()
HERE = Path(__file__).resolve().parent

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
    REPO / "colab-label/build_pool.py",
    REPO / "colab-label/build_pool_ext.py",
    REPO / "colab-label/judge.py",
    REPO / "colab-label/run_labels.py",
    HERE / "to_prior_labels.py",
    HERE / "prior_column_report.py",
    REPO / "colab-label/build_public_all.py",
]

missing = [p for p in wanted if not p.exists()]
if missing:
    raise SystemExit("missing inputs (point --repo at the 0705-line checkout):\n  "
                     + "\n  ".join(str(p) for p in missing))

with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in wanted:
        base = REPO if str(p).startswith(str(REPO)) else HERE.parent
        z.write(p, f"official-router/{p.relative_to(base).as_posix()}")
print(f"wrote {a.out} ({a.out.stat().st_size/1e6:.1f} MB, {len(wanted)} files)")
