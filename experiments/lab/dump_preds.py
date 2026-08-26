# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Dump the deployed pipeline's per-episode predictions for a split."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import learned_router
from ossp_router.protocol import load_input, TIERS, MODEL_IDS

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", type=Path, required=True)
ap.add_argument("--input", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--keep-lookup", action="store_true")
a = ap.parse_args()

raw = json.loads(a.artifact.read_text(encoding="utf-8"))
if not a.keep_lookup:
    raw.pop("public_lookup", None)
art = learned_router.parse_artifact(raw, base_path=a.artifact.parent)
inp = load_input(a.input)
res = {}
for tier in TIERS:
    S = np.zeros((len(inp.episodes), 3)); C = np.zeros_like(S)
    for i, e in enumerate(inp.episodes):
        s, c = learned_router.predict_episode_augmented(e, art, tier)
        for j, m in enumerate(MODEL_IDS):
            S[i, j] = s[m]; C[i, j] = c[m]
    res[f"score_{tier}"] = S; res[f"cost_{tier}"] = C
np.savez_compressed(a.out, **res, episode_ids=np.array([e.episode_id for e in inp.episodes]))
print("wrote", a.out)
