# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a13 / provider view - part 11: construction forensics of the public splits.

- episode_id numbering: contiguous?  gaps?
- is the family sequence along episode_id i.i.d. (shuffled pool) or blocky
  (concatenated then split)?  runs test + lag-1 agreement.
- AIME source pool accounting (60 problems, how many consumed).
- duplicate prompts across splits.
Run: .venv/Scripts/python.exe experiments/lab/a13_provider_11_ids.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
sys.path.insert(0, str(ROOT / "src"))

import labdata  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

tr, dv = labdata.load_all()

print("=" * 88)
print("A. episode_id numbering")
print("=" * 88)
for sp in (tr, dv):
    nums = sorted(int(e.split("-")[1]) for e in sp.episode_ids)
    print(f"  {sp.name}: n={len(nums)} min {nums[0]} max {nums[-1]} "
          f"contiguous={nums == list(range(nums[0], nums[0]+len(nums)))} "
          f"gaps={len(set(range(nums[0], nums[-1]+1)) - set(nums))}")

print()
print("=" * 88)
print("B. is the family sequence shuffled along episode_id?")
print("=" * 88)
for sp in (tr, dv):
    fam = np.array([classify_family(t) for t in sp.texts])
    order = np.argsort([int(e.split("-")[1]) for e in sp.episode_ids])
    f = fam[order]
    lag1 = np.mean(f[1:] == f[:-1])
    p = np.array([np.mean(fam == x) for x in sorted(set(fam))])
    exp = float((p ** 2).sum())
    # runs test z
    n = len(f)
    z = (lag1 - exp) / np.sqrt(exp * (1 - exp) / n)
    print(f"  {sp.name}: P(same family as previous id) = {lag1:.4f}, "
          f"i.i.d. expectation {exp:.4f}, z = {z:+.2f}")
    # block check: family composition of first vs second half
    h = n // 2
    c1, c2 = Counter(f[:h]), Counter(f[h:])
    print(f"      first half {dict(sorted(c1.items()))}")
    print(f"      second half {dict(sorted(c2.items()))}")

print()
print("=" * 88)
print("C. AIME pool accounting")
print("=" * 88)
pools = {"aime24-public": set(range(60, 90)), "aime25-public": set(range(0, 30))}
used = {k: {"train": set(), "dev": set()} for k in pools}
for sp in ("train", "dev"):
    sel = json.loads((ROOT / f"data/{sp}/aime-selection.json").read_text(encoding="utf-8"))
    for e in sel["episodes"]:
        used[e["source_id"]][sp].add(int(e["source_key"]["source_id"]))
tot_left = 0
for k, pool in pools.items():
    t, d = used[k]["train"], used[k]["dev"]
    left = pool - t - d
    tot_left += len(left)
    print(f"  {k}: pool {len(pool)} (ids {min(pool)}..{max(pool)})  "
          f"train {len(t)}  dev {len(d)}  overlap {len(t & d)}  UNUSED {len(left)}")
    print(f"      unused ids: {sorted(left)}")
print(f"  TOTAL AIME pool 60, used 36 (24 train / 12 dev), unused {tot_left}")
print(f"  public AIME rate: train {24/1760*100:.4f}%  dev {12/880*100:.4f}%")
print(f"  a private split at the same rate would need "
      f"{24/(24/1760):.0f} episodes to consume all {tot_left} unused AIME problems")

print()
print("=" * 88)
print("D. duplicate prompts?")
print("=" * 88)
h = {}
for sp in (tr, dv):
    for e, t in zip(sp.episode_ids, sp.texts):
        h.setdefault(hashlib.sha256(t.encode("utf-8")).hexdigest(), []).append(e)
dups = {k: v for k, v in h.items() if len(v) > 1}
print(f"  distinct prompt hashes {len(h)} over {len(tr)+len(dv)} episodes; "
      f"duplicates {len(dups)}")

print()
print("=" * 88)
print("E. what a hash lookup over the PUBLIC SOURCES could plausibly cover")
print("=" * 88)
print("  (counts of public source items reproduced by colab-label/build_pool.py")
print("   --verify are recorded in EXPERIMENT_LOG E41; not re-measured here)")
print("  measured here: AIME is the only source whose pool is provably almost")
print("  exhausted by the public splits (36/60).")
