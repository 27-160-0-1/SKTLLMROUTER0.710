# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Smoke test for to_prior_labels.py + prior_column_report.py, using the shipped column A.

Builds a fake item pool from the public Dev/Train prompts and a fake run_labels.py output whose
scores are column A's own stored values, then checks that the report recovers column A's known
agreement with ax31-light (~0.70-0.75).  Catches format/plumbing errors before any GPU time.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import load_input  # noqa: E402

TMP = ROOT / "colab-label/_smoke"
TMP.mkdir(exist_ok=True)
art = json.loads((ROOT / "src/ossp_router/resources/learned-router.v1.json").read_text(encoding="utf-8"))
colA = art["prior_lookup"]["columns"][0]["entries"]

items, labels = [], []
for split in ("train", "dev"):
    for ep in load_input(ROOT / f"data/materialized/{split}/inputs.json").episodes:
        text = episode_text(ep)
        row = colA.get(hashlib.sha256(text.encode()).hexdigest())
        if row is None:
            continue
        iid = f"smoke-{ep.episode_id}"
        items.append({"id": iid, "family": "gsm8k_or_other", "source": "smoke", "prompt": text})
        score, _length, sc = row
        # emit in run_labels.py's shape so to_prior_labels.py has something real to convert
        n = 4
        ncorrect = int(round(float(score) * n)) if score >= 0 else 0
        gens = [{"text": f"Answer: {i < ncorrect}", "tokens": 40, **({"correct": i < ncorrect} if score >= 0 else {})}
                for i in range(n)]
        labels.append({"id": iid, "family": "gsm8k_or_other", "source": "smoke", "n": n,
                       "score": float(score) if score >= 0 else None, "out_per_gen": 40.0, "gens": gens})

(TMP / "items.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in items) + "\n", encoding="utf-8")
(TMP / "raw.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in labels) + "\n", encoding="utf-8")
print(f"[smoke] built {len(items)} items / {len(labels)} label rows from column A")

py = sys.executable
subprocess.run([py, "-X", "utf8", str(ROOT / "colab-label/to_prior_labels.py"),
                str(TMP / "raw.jsonl"), str(TMP / "labels.jsonl"), "--model", "smoke-colA"], check=True)
subprocess.run([py, "-X", "utf8", str(ROOT / "colab-label/prior_column_report.py"),
                "--labels", str(TMP / "labels.jsonl"), "--items", str(TMP / "items.jsonl")], check=True)
print("[smoke] expected: corr(ax31-light) ~ 0.70-0.75 (this replays column A), digest overlap ~1.0 for column 0")
