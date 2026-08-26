# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the gold=None path that crashed E59b.

Drives `run_labels.run_stage` with the mock engine over real `public_all.jsonl` rows -- both
kinds, with and without a gold answer -- then runs the converter.  Asserts that gold-less items
survive with score=None and a self-consistency value, which is exactly what the prior table
needs for dmmath/longdoc coverage.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "colab-label"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import run_labels as RL

items = [json.loads(l) for l in (HERE / "bundle/public_all.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
with_gold = [i for i in items if i.get("gold")][:6]
no_gold = [i for i in items if not i.get("gold")][:6]
sample = with_gold + no_gold
print(f"[smoke] {len(with_gold)} with gold, {len(no_gold)} without "
      f"(families {sorted({i['family'] for i in no_gold})})")

TMP = HERE / "_smoke_nogold"
TMP.mkdir(exist_ok=True)
out = TMP / "labels.jsonl"
out.unlink(missing_ok=True)


class Engine:
    """Deterministic stand-in: 4 generations, two distinct answers, so consistency is well defined."""

    def generate(self, prompts, n, temperature, top_p, max_tokens):
        res = []
        for _p in prompts:
            gens = [{"text": f"Answer: {42 if k < 3 else 7}", "tokens": 12 + k} for k in range(n)]
            res.append((100, gens))
        return res


RL.run_stage(Engine(), sample, out, 0.7, 4, 0.95, 384, batch=8, instruct="v1")
rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
assert len(rows) == len(sample), f"expected {len(sample)} rows, got {len(rows)}"
scored = [r for r in rows if r["score"] is not None]
unscored = [r for r in rows if r["score"] is None]
print(f"[smoke] run_stage OK: {len(scored)} scored, {len(unscored)} unscored (gold-less survived)")
assert len(unscored) == len(no_gold), "gold-less items were dropped or scored"

conv = TMP / "prior.jsonl"
subprocess.run([sys.executable, "-X", "utf8", str(HERE / "to_prior_labels.py"), str(out), str(conv),
                "--model", "smoke"], check=True)
prior = [json.loads(l) for l in conv.read_text(encoding="utf-8").splitlines() if l.strip()]
assert len(prior) == len(sample)
for r in prior:
    assert r["sc"] is not None, "self-consistency missing"
    assert r["out_tokens_total"] > 0, "output length missing"
nulls = sum(r["score"] is None for r in prior)
print(f"[smoke] converter OK: {nulls} rows carry score=None with sc={prior[-1]['sc']} "
      f"and out_tokens_total={prior[-1]['out_tokens_total']}")
print("[smoke] PASS — gold-less items now reach the prior table with length + consistency")
