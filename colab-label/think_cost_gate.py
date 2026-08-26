# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Gate for E66: does a reasoning model's output length predict `axk1-think`'s?

E64 located the remaining headroom precisely.  The whole safety margin insures against cost
prediction error (allocating on true costs never busts, at any ratio), the error sits in the
think head (log-cost RMSE 0.677 against 0.458 for mid), and nothing in the artifact carries the
signal: the 34B column's output length correlates **0.182** with think's log output length, and
even the real `ax31`'s measured output length only reaches **0.319**.

think's cost is ~87 % output tokens, so predicting its *length* is the whole problem.  A
reasoning model is the natural proxy -- it is the only thing that emits a long chain the way
think does.  This measures whether that is true, on the public items where think's real output
length is known.

The bar: beat **0.319**, the correlation the real 34B's own output length achieves.  Below that
the reasoning column adds nothing the artifact could not already have.

Usage:
    PYTHONPATH=src python colab-label/think_cost_gate.py \
        --labels colab-label/out/labels_pilot_T0.7_reason.jsonl \
        --items colab-label/bundle/pilot.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ossp_router import similarity  # noqa: E402
from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import load_input, load_outcomes  # noqa: E402


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--items", type=Path, nargs="+", required=True)
    ap.add_argument("--inputs", type=Path, nargs="+",
                    default=[ROOT / "data/materialized/train/inputs.json",
                             ROOT / "data/materialized/dev/inputs.json"])
    ap.add_argument("--outcomes", type=Path, nargs="+",
                    default=[ROOT / "data/train/outcomes.json", ROOT / "data/dev/outcomes.json"])
    a = ap.parse_args()

    prompts = {}
    for path in a.items:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                prompts[row["id"]] = row["prompt"]

    by_digest = {}
    for path in a.labels:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = prompts.get(row["id"])
            if text is None:
                continue
            gens = row.get("gens") or []
            length = (sum(g.get("tokens", 0) for g in gens) / len(gens)) if gens else row.get("out_per_gen")
            if length:
                by_digest[sha(text)] = float(length)
    print(f"[gate] reasoning-model lengths joined to prompts: {len(by_digest)}")

    episodes, index = [], {}
    for ip, op in zip(a.inputs, a.outcomes):
        episodes.extend(load_input(ip).episodes)
        for o in load_outcomes(op).outcomes:
            index[(o.episode_id, o.model_id)] = o

    ours, think, mid, light, fams = [], [], [], [], []
    for ep in episodes:
        value = by_digest.get(sha(episode_text(ep)))
        if value is None:
            continue
        ours.append(np.log1p(value))
        fams.append(similarity.classify_family(episode_text(ep)))
        for store, model in ((think, "axk1-think"), (mid, "ax31"), (light, "ax31-light")):
            o = index[(ep.episode_id, model)]
            store.append(np.log1p(o.output_tokens / max(o.num_generations, 1)))
    ours = np.array(ours); think = np.array(think); mid = np.array(mid); light = np.array(light)
    fams = np.array(fams)
    print(f"[gate] public items covered: {len(ours)}/{len(episodes)} = {len(ours)/len(episodes):.3f}")
    if len(ours) < 100:
        print("[gate] too few items to judge")
        return 0

    corr = float(np.corrcoef(ours, think)[0, 1])
    print(f"\n{'대상':<28}{'corr':>8}")
    print(f"{'추론모델 길이 -> think 길이':<28}{corr:>8.3f}   <-- 게이트")
    print(f"{'추론모델 길이 -> ax31 길이':<28}{float(np.corrcoef(ours, mid)[0, 1]):>8.3f}")
    print(f"{'추론모델 길이 -> light 길이':<28}{float(np.corrcoef(ours, light)[0, 1]):>8.3f}")
    print(f"{'(기존 최고) ax31 길이 -> think':<28}{float(np.corrcoef(mid, think)[0, 1]):>8.3f}")
    both = np.column_stack([ours, mid])
    beta, *_ = np.linalg.lstsq(np.column_stack([both, np.ones(len(ours))]), think, rcond=None)
    pred = np.column_stack([both, np.ones(len(ours))]) @ beta
    print(f"{'추론모델 + ax31 결합':<28}{float(np.corrcoef(pred, think)[0, 1]):>8.3f}")

    print("\nfamily 별 (추론모델 -> think):")
    for f in sorted(set(fams)):
        sel = fams == f
        if sel.sum() < 20 or np.std(ours[sel]) == 0:
            continue
        print(f"  {f:<18}{sel.sum():>5}{float(np.corrcoef(ours[sel], think[sel])[0, 1]):>8.3f}")

    print(f"\nGATE: {corr:.3f} vs 0.319 (기존 최고, 실제 ax31 출력길이)")
    print("ADOPT — 대량 라벨링 진행" if corr >= 0.45 else
          "BELOW BAR — 이 축은 여기서 종료. 표를 그대로 붙여넣을 것")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
