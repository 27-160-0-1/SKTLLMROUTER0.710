# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Write bundle/public_all.jsonl: every public Train+Dev prompt, verbatim.

`build_pool.py` renders items *from the public sources*, which reaches only ~74 % of the
challenge's own prompts (DeepMind-Mathematics and some AIME/GSM8K items do not round-trip).
`tools/deploy_v2.ps1` therefore also feeds `bundle/public_all.jsonl` to the prior builder --
the public prompts themselves.  This script produces that file.

Gold answers come from `data/gold/gold-answers.v1.json` where they exist; items without one
get `gold: null`, so the labeller records length and self-consistency but no score (exactly the
degradation E53 relied on to lift coverage from 0.69 to 0.91).

Note on what this coverage buys: entries keyed to the public prompts help the held-out Dev
measurement and any evaluation item drawn from the same public rows, but an unseen private
prompt is only covered by the source-rendered pool (`pool.jsonl` / `ext.jsonl`).  Include it to
compare like with like against the shipped column A, which is built the same way.

Usage: PYTHONPATH=src python colab-label/build_public_all.py [--out colab-label/bundle/public_all.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ossp_router import similarity  # noqa: E402
from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import load_input  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+",
                    default=[ROOT / "data/materialized/train/inputs.json",
                             ROOT / "data/materialized/dev/inputs.json"])
    ap.add_argument("--gold", type=Path, default=ROOT / "data/gold/gold-answers.v1.json")
    ap.add_argument("--out", type=Path, default=ROOT / "colab-label/bundle/public_all.jsonl")
    ap.add_argument("--exclude-digests", type=Path, default=None,
                    help="newline-separated prompt digests to leave out, so a second labelling pass "
                         "only covers what the first one missed")
    a = ap.parse_args()

    gold = json.loads(a.gold.read_text(encoding="utf-8")) if a.gold.exists() else {}
    skip = set()
    if a.exclude_digests and a.exclude_digests.exists():
        skip = {l.strip() for l in a.exclude_digests.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"[public_all] excluding {len(skip)} already-covered prompts")
    rows, fams, with_gold = [], Counter(), 0
    for path in a.inputs:
        for ep in load_input(path).episodes:
            text = episode_text(ep)
            if hashlib.sha256(text.encode("utf-8")).hexdigest() in skip:
                continue
            g = gold.get(ep.episode_id)
            family = (g or {}).get("family") or similarity.classify_family(text)
            fams[family] += 1
            with_gold += g is not None
            rows.append({
                "id": f"public-{ep.episode_id}",
                "family": family,
                "source": (g or {}).get("source", "public"),
                "prompt": text,
                "gold": {"answer": g["answer"]} if g else None,
            })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"[public_all] {len(rows)} prompts -> {a.out} ({with_gold} with a gold answer)")
    print(f"[public_all] families: {dict(fams)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
