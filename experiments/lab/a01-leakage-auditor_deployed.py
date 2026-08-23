# SPDX-License-Identifier: Apache-2.0
"""A01: how much of the deployed artifact's dev score is self-memorisation?

Scores the DEPLOYED artifact (trained on train+dev = 2,640) on dev, with and
without the SHA-256 lookup, and compares to the honest train-only holdout.
Also measures kNN self-retrieval on dev under both indexes.
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labdata as L
from ossp_router import learned_router, similarity
from ossp_router.heuristic import episode_text
from ossp_router.protocol import load_input, TIERS, MODEL_IDS

SP = Path(r"C:/Users/PJ05/AppData/Local/Temp/claude/C--portable-skt-LLM1-LLM-ROUTE-0-7000/377d7fd8-9983-4bec-bb08-163cc405f7a3/scratchpad")
tr, dv = L.load_all()

def score(npz, safety, tag):
    S = {t: npz[f"score_{t}"] for t in L.TIERS}
    C = {t: npz[f"cost_{t}"] for t in L.TIERS}
    tot = 0.0; parts = []
    for t in L.TIERS:
        r = L.tier_result(S[t], C[t], dv, t, safety[t])
        tot += L.TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t}={r['tier_score']:.4f}(r={r['ratio']:.3f})")
    print(f"{tag:52s} final={tot:.6f}  " + " ".join(parts))
    return tot

hold = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
dep = np.load(SP / "a01_deployed_nolookup.npz", allow_pickle=True)

print("== dev score, allocator-exact ==")
sa_dep = dict(fast=.98, balanced=.87, premium=.85)
sa_hold = dict(fast=.98, balanced=.89, premium=.88)
a = score(hold, sa_hold, "honest holdout (train-only) @ .98/.89/.88")
b = score(hold, sa_dep,  "honest holdout (train-only) @ .98/.87/.85")
c = score(dep,  sa_dep,  "DEPLOYED (train+dev) no-lookup @ .98/.87/.85")
d = score(dep,  sa_hold, "DEPLOYED (train+dev) no-lookup @ .98/.89/.88")
print(f"\nin-sample optimism (same safety .98/.87/.85): {c-b:+.6f}")
print(f"in-sample optimism (same safety .98/.89/.88): {d-a:+.6f}")

# ---- lookup path: does the stored row reproduce the compute row exactly? ----
raw = json.loads((ROOT/"src/ossp_router/resources/learned-router.v1.json").read_text(encoding="utf-8"))
art = learned_router.parse_artifact(raw, base_path=ROOT/"src/ossp_router/resources")
inp = load_input(ROOT/"data/materialized/dev/inputs.json")
lut = art.public_lookup
hits = 0
maxdiff = 0.0
for i, e in enumerate(inp.episodes):
    key = hashlib.sha256(episode_text(e).encode("utf-8")).hexdigest()
    if key in lut:
        hits += 1
print(f"\npublic_lookup: {len(lut)} rows, dev prompt hits = {hits}/{len(inp.episodes)}")
tr_ep = load_input(ROOT/"data/materialized/train/inputs.json")
th = sum(1 for e in tr_ep.episodes if hashlib.sha256(episode_text(e).encode()).hexdigest() in lut)
print(f"                train prompt hits = {th}/{len(tr_ep.episodes)}")
# compare a few stored rows with the recomputed ones
for tier_i, tier in enumerate(TIERS):
    diffs = []
    for i, e in enumerate(inp.episodes[:60]):
        key = hashlib.sha256(episode_text(e).encode()).hexdigest()
        row = lut[key][tier_i*6:(tier_i+1)*6]
        s, c = learned_router.predict_episode_augmented(e, art, tier)
        got = [s[m] for m in MODEL_IDS] + [c[m] for m in MODEL_IDS]
        diffs.append(max(abs(x-y) for x, y in zip(row, got)))
    print(f"  lookup vs compute, tier={tier}: max abs diff over 60 dev rows = {max(diffs):.3e}")

# ---- kNN self-retrieval ----
aug = art.augmentation
print(f"\ndeployed kNN index: {len(aug.index.targets)} rows")
top1_dep, self_frac = [], 0
dev_texts = [episode_text(e) for e in inp.episodes]
for t in dev_texts:
    q = similarity.tfidf_vector(t, aug.idf)
    row, top1 = aug.index.predict(q)
    top1_dep.append(top1)
top1_dep = np.array(top1_dep)
raw_h = json.loads((ROOT/"reports/holdout_local/learned-router.v1.json").read_text(encoding="utf-8"))
art_h = learned_router.parse_artifact(raw_h)
aug_h = art_h.augmentation
print(f"holdout  kNN index: {len(aug_h.index.targets)} rows")
top1_h = []
for t in dev_texts:
    q = similarity.tfidf_vector(t, aug_h.idf)
    _row, top1 = aug_h.index.predict(q)
    top1_h.append(top1)
top1_h = np.array(top1_h)
print(f"dev top-1 similarity: deployed(2640) mean={top1_dep.mean():.4f} median={np.median(top1_dep):.4f} frac>0.99={float((top1_dep>0.99).mean()):.4f}")
print(f"                      holdout (1760) mean={top1_h.mean():.4f} median={np.median(top1_h):.4f} frac>0.99={float((top1_h>0.99).mean()):.4f}")
np.savez(SP/"a01_top1.npz", dep=top1_dep, hold=top1_h)
