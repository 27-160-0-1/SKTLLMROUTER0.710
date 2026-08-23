# SPDX-License-Identifier: Apache-2.0
"""Extend the public prompt pool for maximum evaluation-time lookup coverage.

`build_pool.py` samples its pool (one random wrong answer per TruthfulQA
question, a shuffled slice of the RuleTaker *train* split, 500 BABILong items).
For a lookup table we want the opposite: every renderable item of the sources the
organiser actually drew from, so that an unseen evaluation prompt is in the
table.  `meta.json` from --verify shows which source each family came from:
GSM8K test, Belebele test, CRUXEval, RuleTaker **test**, TruthfulQA MC1 pairs,
BABILong, HRMCR.

Writes only the items that are not already in bundle/all.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import build_pool as BP  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "bundle/ext.jsonl")
    ap.add_argument("--have", type=Path, default=HERE / "bundle/all.jsonl")
    ap.add_argument("--ruletaker-max", type=int, default=24000)
    ap.add_argument("--babilong", action="store_true")
    a = ap.parse_args()
    from datasets import load_dataset

    have = set()
    for line in a.have.read_text(encoding="utf-8").splitlines():
        if line.strip():
            have.add(BP.norm(json.loads(line)["prompt"]))
    print(f"[ext] already have {len(have)} prompts", flush=True)

    rows, seen = [], set()

    def add(family, prompt, answer, source, check=None):
        key = BP.norm(prompt)
        if key in have or key in seen:
            return
        seen.add(key)
        g = {"answer": answer, "source": source}
        if check:
            g["check"] = check
        rows.append({"id": f"ext-{len(rows):06d}", "family": family, "source": source,
                     "prompt": prompt, "gold": g})

    # TruthfulQA: every (correct, wrong) pair in both orders
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    for r in ds:
        ch, lb = r["mc1_targets"]["choices"], r["mc1_targets"]["labels"]
        correct = [c for c, l in zip(ch, lb) if l == 1]
        wrong = [c for c, l in zip(ch, lb) if l == 0]
        if not correct or not wrong:
            continue
        c = correct[0]
        for w in wrong:
            add("truthfulqa", BP.t_truthfulqa(r["question"], c, w), "A", "truthfulqa-ext")
            add("truthfulqa", BP.t_truthfulqa(r["question"], w, c), "B", "truthfulqa-ext")
    print(f"[ext] after truthfulqa: {len(rows)}", flush=True)

    # RuleTaker: the TEST split is what the organiser drew from
    ds = load_dataset("tasksource/ruletaker", split="test")
    n = min(len(ds), a.ruletaker_max)
    for i in range(n):
        r = ds[i]
        add("ruletaker", BP.t_ruletaker(r), BP.ruletaker_label(r["label"]),
            f"ruletaker-test-{r.get('config','')}")
    print(f"[ext] after ruletaker (test, {n} of {len(ds)}): {len(rows)}", flush=True)

    if a.babilong:
        for length in ("4k", "16k"):
            for task in [f"qa{i}" for i in range(1, 11)]:
                try:
                    ds = load_dataset("RMT-team/babilong", length, split=task)
                except Exception:
                    continue
                for r in ds:
                    add("longdoc", BP.t_babilong(r), str(r["target"]),
                        f"babilong-{length}-{task}")
        print(f"[ext] after babilong: {len(rows)}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                     encoding="utf-8")
    print(f"[ext] wrote {len(rows)} new items -> {a.out}: "
          f"{dict(Counter(r['family'] for r in rows))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
