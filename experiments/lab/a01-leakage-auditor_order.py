# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""A01 Q5: prove the runtime decision for an episode does not depend on its
position in the batch, its episode_id, or the declared split."""
import json, sys, random
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import learned_router
from ossp_router.protocol import load_input, load_bundled_policy, TIERS, MODEL_IDS, Episode, InputBatch

raw = json.loads((ROOT/"reports/holdout_local/learned-router.v1.json").read_text(encoding="utf-8"))
art = learned_router.parse_artifact(raw)
policy = load_bundled_policy()
inp = load_input(ROOT/"data/materialized/dev/inputs.json")
eps = list(inp.episodes)

# 1) per-episode prediction is a pure function of content: rename ids + shuffle
z = np.load(ROOT/"reports/lab/dev_preds_e43.npz", allow_pickle=True)
order = list(range(len(eps))); random.Random(5).shuffle(order)
renamed = [Episode(episode_id=f"zz-{k:05d}", prompt=eps[i].prompt, messages=eps[i].messages)
           for k, i in enumerate(order)]
maxdiff = 0.0
for tier in TIERS:
    for k, i in enumerate(order[:120]):
        s, c = learned_router.predict_episode_augmented(renamed[k], art, tier)
        ref_s = z[f"score_{tier}"][i]; ref_c = z[f"cost_{tier}"][i]
        maxdiff = max(maxdiff, max(abs(s[m]-ref_s[j]) for j, m in enumerate(MODEL_IDS)),
                              max(abs(c[m]-ref_c[j]) for j, m in enumerate(MODEL_IDS)))
print(f"per-episode prediction, ids renamed + batch shuffled: max abs diff vs reference = {maxdiff:.3e}")

# 2) batch selection is permutation-equivariant
shuf = InputBatch(schema_version=inp.schema_version, challenge_id="zzz-other-challenge",
                  split="private", episodes=tuple(renamed))
for tier in TIERS:
    sub_a = learned_router.make_submission(inp, policy, art, tier)
    sub_b = learned_router.make_submission(shuf, policy, art, tier)
    a = [d.model_id for d in sub_a.decisions]
    b = [d.model_id for d in sub_b.decisions]
    back = [None]*len(a)
    for k, i in enumerate(order):
        back[i] = b[k]
    print(f"tier={tier:9s} identical selection after shuffle+rename+split/challenge change: {a == back}")
