# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Leakage audit of the shipped prior_lookup.

If the table held the organiser's own labels, the prior score would reproduce the true
ax31-light score (corr ~1.0, agreement ~1.0) on Dev.  A genuine offline proxy sits far
below that.  Also reports Dev coverage and how much of the table's mass is Dev at all.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.heuristic import episode_text
from ossp_router.protocol import load_input, load_outcomes

art = json.loads((ROOT / "src/ossp_router/resources/learned-router.v1.json").read_text(encoding="utf-8"))
cols = art["prior_lookup"]["columns"]
print("provenance:", json.dumps(art["prior_lookup"]["provenance"]))
for i, c in enumerate(cols):
    print(f"column {i}: tag={c.get('tag')} entries={len(c['entries'])}")

for split in ("dev", "train"):
    inputs = load_input(ROOT / f"data/materialized/{split}/inputs.json")
    outcomes = load_outcomes(ROOT / f"data/{split}/outcomes.json")
    idx = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    eps = list(inputs.episodes)
    digests = [hashlib.sha256(episode_text(e).encode("utf-8")).hexdigest() for e in eps]
    s_light = np.array([float(idx[(e.episode_id, "ax31-light")].score) for e in eps])
    s_mid = np.array([float(idx[(e.episode_id, "ax31")].score) for e in eps])
    print(f"\n=== {split} ({len(eps)} episodes) ===")
    for i, c in enumerate(cols):
        ent = c["entries"]
        hit = np.array([d in ent for d in digests])
        scored = np.array([bool(d in ent and ent[d][0] >= 0.0) for d in digests])
        print(f"  column {i}: coverage {hit.mean():.3f}, with-score {scored.mean():.3f}")
        if scored.sum() > 10:
            p = np.array([ent[d][0] for d, s in zip(digests, scored) if s])
            t_l = s_light[scored]
            t_m = s_mid[scored]
            exact = float((np.abs(p - t_l) < 1e-9).mean())
            within = float((np.abs(p - t_l) <= 0.25).mean())
            print(f"     vs TRUE ax31-light: corr {np.corrcoef(p, t_l)[0,1]:.3f}  within.25 {within:.3f}  EXACT-equal {exact:.3f}")
            print(f"     vs TRUE ax31      : corr {np.corrcoef(p, t_m)[0,1]:.3f}")
            print(f"     prior mean {p.mean():.3f} vs true light mean {t_l.mean():.3f}")
