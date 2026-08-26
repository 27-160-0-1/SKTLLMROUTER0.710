# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Render the AIME source pool.  `build_pool.py` has no AIME renderer at all, so the prior
table covers no AIME item beyond the ones the public split already contains -- and AIME is the
family where routing up to the reasoning model matters most.

Provenance is pinned by the organiser's own files:

    data/{train,dev}/aime-selection.json -> source_id aime24-public / aime25-public,
                                            source_key {"source_id": <upstream row id>, "year": ...}

The challenge prompt is the upstream `problem` field verbatim -- no template around it.  Which
upstream, per year, was settled by hashing: measured against the 36 AIME episodes of the public
Train+Dev split,

    HuggingFaceH4/aime_2024        30 rows   18/36   (all of 2024; its ids 60-89 match source_id)
    math-ai/aime25                 30 rows   16/36   (2025)
    allenai/aime-2022-2025        120 rows   31/36
    AI-MO/aimo-validation-aime     90 rows   18/36
    union of the four                        35/36   (only train-0845 / 2025 #19 differs)

Rows are read through the HF datasets-server API (these sets are small) and fall back to
`datasets.load_dataset`.  Duplicate problems across sources are merged by prompt hash.

Usage:
    PYTHONPATH=src python colab-label/build_pool_aime.py --verify
    PYTHONPATH=src python colab-label/build_pool_aime.py --out colab-label/bundle/aime.jsonl \
        --have colab-label/bundle/all.jsonl colab-label/bundle/ext.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import load_input  # noqa: E402

# (dataset, config, split, tag) -- verified above; order only affects which id a duplicate keeps.
SOURCES = [
    ("HuggingFaceH4/aime_2024", "default", "train", "aime24-public"),
    ("math-ai/aime25", "default", "test", "aime25-public"),
    ("allenai/aime-2022-2025", "default", "train", "aime-2022-2025"),
    ("AI-MO/aimo-validation-aime", "default", "train", "aime-2022-2024"),
]
API = "https://datasets-server.huggingface.co/rows"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fields(row):
    problem = row.get("problem") or row.get("Problem") or row.get("question")
    answer = row.get("answer") or row.get("Answer")
    return (None if problem is None else str(problem),
            None if answer is None else str(answer).strip(),
            row.get("id"))


def fetch(dataset, config, split, cap=400):
    """datasets-server first (fast, no download), then datasets.load_dataset."""
    out, offset = [], 0
    try:
        while offset < cap:
            url = (f"{API}?dataset={urllib.parse.quote(dataset, safe='')}"
                   f"&config={config}&split={split}&offset={offset}&length=100")
            with urllib.request.urlopen(url, timeout=60) as fh:
                rows = json.load(fh)["rows"]
            if not rows:
                break
            out.extend(_fields(r["row"]) for r in rows)
            offset += len(rows)
            if len(rows) < 100:
                break
        if out:
            return out
    except Exception as exc:
        print(f"[aime] {dataset}: datasets-server unavailable ({type(exc).__name__}), falling back", flush=True)
    from datasets import load_dataset
    return [_fields(r) for r in load_dataset(dataset, split=split)]


def public_aime():
    """The AIME episodes of the public Train+Dev split, keyed by prompt digest."""
    out = {}
    for split in ("train", "dev"):
        sel = json.loads((ROOT / f"data/{split}/aime-selection.json").read_text(encoding="utf-8"))
        ids = {e["episode_id"]: e for e in sel["episodes"]}
        for ep in load_input(ROOT / f"data/materialized/{split}/inputs.json").episodes:
            if ep.episode_id in ids:
                out[sha(episode_text(ep))] = (ep.episode_id, ids[ep.episode_id]["source_key"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "colab-label/bundle/aime.jsonl")
    ap.add_argument("--have", type=Path, nargs="*", default=[])
    ap.add_argument("--have-digests", type=Path, default=None,
                    help="newline-separated prompt digests already covered (avoids shipping the big pools)")
    ap.add_argument("--verify", action="store_true", help="report template match only, write nothing")
    a = ap.parse_args()

    target = public_aime()
    print(f"[aime] organiser AIME episodes in the public split: {len(target)}")

    have = set()
    for path in a.have:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    have.add(sha(json.loads(line)["prompt"]))
    if a.have_digests and a.have_digests.exists():
        have |= {l.strip() for l in a.have_digests.read_text(encoding="utf-8").splitlines() if l.strip()}
    if have:
        print(f"[aime] prompts already in the pool: {len(have)}")

    seen, rows_out, matched = set(), [], set()
    for dataset, config, split, tag in SOURCES:
        try:
            rows = fetch(dataset, config, split)
        except Exception as exc:                      # a renamed source must not kill the run
            print(f"[aime] {dataset:<30} SKIPPED ({type(exc).__name__}: {exc})")
            continue
        hit = 0
        for problem, answer, rid in rows:
            if not problem:
                continue
            digest = sha(problem)
            if digest in target:
                hit += 1
                matched.add(digest)
            if digest in seen or digest in have:
                continue
            seen.add(digest)
            rows_out.append({
                "id": f"aime-{tag}-{rid if rid is not None else len(rows_out)}",
                "family": "aime",
                "source": tag,
                "prompt": problem,
                "gold": {"answer": answer} if answer else None,
            })
        print(f"[aime] {dataset:<30} rows {len(rows):>4}  reproduces {hit:>2}/{len(target)}")

    print(f"[aime] template verification: {len(matched)}/{len(target)} organiser AIME prompts reproduced exactly")
    missing = [v for k, v in target.items() if k not in matched]
    if missing:
        print(f"[aime] not reproduced ({len(missing)}): {missing}")
    if len(matched) < 0.8 * len(target):
        print("[aime] WARNING: under 80 % reproduced -- the upstream text has drifted, stop and report this")

    if a.verify:
        return 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out) + "\n", encoding="utf-8")
    with_gold = sum(1 for r in rows_out if r["gold"])
    print(f"[aime] wrote {len(rows_out)} items -> {a.out} ({with_gold} with a gold answer) "
          f"{dict(Counter(r['source'] for r in rows_out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
