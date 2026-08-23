# SPDX-License-Identifier: Apache-2.0
"""Time the heavy (lookup-miss) prediction path for one or more artifacts.

Seed-averaging the meta heads multiplies the exported tree count by the seed count, and the
container has a 90 s per-tier budget (E44), so the cost of that has to be measured rather than
assumed.  This times `predict_episode_augmented` -- the same call `holdout_eval.py` makes -- on
a sample of episodes, and reports per-episode microseconds plus the meta tree count.

Usage:
    PYTHONPATH=src python tools/time_heavy_path.py --artifact A.json B.json [--limit 300]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ossp_router import learned_router
from ossp_router.protocol import load_input


def meta_tree_count(raw) -> int:
    meta = raw.get("meta_gbm") or {}
    total = 0
    for key in ("trees", "delta_trees", "ordinal_trees", "rank_trees"):
        value = meta.get(key) or []
        if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
            total += sum(len(t) for t in value)
        else:
            total += len(value)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, nargs="+", required=True)
    ap.add_argument("--input", type=Path, default=Path("data/materialized/dev/inputs.json"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--tier", default="premium")
    a = ap.parse_args()

    episodes = list(load_input(a.input).episodes)[: a.limit]
    print(f"{'artifact':<28}{'meta trees':>12}{'MB':>8}{'us/episode':>13}{'880x3 tiers':>14}")
    for path in a.artifact:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.pop("public_lookup", None)          # force the miss path, as holdout_eval does
        artifact = learned_router.parse_artifact(raw, base_path=path.parent)
        learned_router.predict_episode_augmented(episodes[0], artifact, a.tier)   # warm caches
        t0 = time.perf_counter()
        for episode in episodes:
            learned_router.predict_episode_augmented(episode, artifact, a.tier)
        elapsed = time.perf_counter() - t0
        per = elapsed / len(episodes)
        print(f"{path.parent.name:<28}{meta_tree_count(raw):>12,}{path.stat().st_size/1e6:>8.1f}"
              f"{per*1e6:>13.0f}{per*880*3:>13.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
