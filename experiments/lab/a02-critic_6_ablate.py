# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a02-critic #6.  Honest per-layer ablation of the deployed stack.

Uses the TRAIN-ONLY artifact in reports/holdout_local/ (num_train_episodes=1760)
with the public lookup stripped, so every number is out-of-sample on dev.
Layers are switched off by editing the parsed artifact JSON in memory -- nothing
on disk is touched.
"""
from __future__ import annotations
import copy, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, TIER_MULT, tier_result
from ossp_router import learned_router
from ossp_router.protocol import load_input

ART = ROOT / "reports/holdout_local/learned-router.v1.json"
CACHE = Path(r"C:\Users\PJ05\AppData\Local\Temp\claude\C--portable-skt-LLM1-LLM-ROUTE-0-7000"
             r"\377d7fd8-9983-4bec-bb08-163cc405f7a3\scratchpad")
CACHE.mkdir(parents=True, exist_ok=True)

dv = load_split("dev")
N = len(dv)
raw0 = json.loads(ART.read_text(encoding="utf-8"))
raw0.pop("public_lookup", None)
inp = load_input(ROOT / "data/materialized/dev/inputs.json")


def mutate(raw, legacy_w=None, drop_aug=False, drop_meta=False,
           fam_w=None, conf=None, meta_blend=None, gain_alpha=None, rank_beta=None):
    r = copy.deepcopy(raw)
    if legacy_w is not None:
        r["legacy_blend_weight"] = legacy_w
    if drop_aug:
        r.pop("augmentation", None)
    if drop_meta:
        r.pop("meta_gbm", None)
    if r.get("augmentation") is not None:
        if fam_w is not None:
            r["augmentation"]["family_blend_weight"] = fam_w
        if conf is not None:
            r["augmentation"]["knn_conf_scale"] = conf
    if r.get("meta_gbm") is not None:
        if meta_blend is not None:
            r["meta_gbm"]["blend_weight"] = {t: meta_blend for t in TIERS}
        if gain_alpha is not None:
            r["meta_gbm"]["gain_alpha"] = gain_alpha
        if rank_beta is not None:
            r["meta_gbm"]["rank_beta"] = rank_beta
    return r


def predict(raw, tag):
    f = CACHE / f"a02_ablate_{tag}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    art = learned_router.parse_artifact(raw, base_path=ART.parent)
    out = {}
    t0 = time.time()
    for tier in TIERS:
        S = np.zeros((N, 3)); C = np.zeros_like(S)
        for i, e in enumerate(inp.episodes):
            s, c = learned_router.predict_episode_augmented(e, art, tier)
            for j, m in enumerate(MODEL_IDS):
                S[i, j] = s[m]; C[i, j] = c[m]
        out[f"score_{tier}"] = S; out[f"cost_{tier}"] = C
    np.savez_compressed(f, **out)
    print(f"    [{tag}: {time.time()-t0:.0f}s]", flush=True)
    return out


SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
SG = np.arange(0.60, 1.601, 0.005)


def evaluate(P, label):
    tot_dep = 0.0
    tot_opt = 0.0
    parts = []
    corr = []
    for t in TIERS:
        r = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, SAFE[t])
        tot_dep += TIER_WEIGHT[t] * r["tier_score"]
        best = 0.0
        for sf in SG:
            rr = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, float(sf))
            if rr["passed"]:
                best = max(best, rr["score"])
        tot_opt += TIER_WEIGHT[t] * best
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else '!'}")
    for j in range(3):
        corr.append(np.corrcoef(P["score_fast"][:, j], dv.score[:, j])[0, 1])
    lc = [np.corrcoef(np.log(P["cost_premium"][:, j]), np.log(dv.cost[:, j]))[0, 1] for j in range(3)]
    print(f"{label:44s} dep={tot_dep:.4f} oracle-safety={tot_opt:.4f}  "
          f"corr_s={'/'.join(f'{c:.2f}' for c in corr)} corr_logc={'/'.join(f'{c:.2f}' for c in lc)}")
    return tot_dep, tot_opt


CONFIGS = [
    ("legacy 256-bin hash-regex ONLY", dict(legacy_w=1.0, drop_aug=True, drop_meta=True)),
    ("ridge 16,414-dim ONLY", dict(legacy_w=0.0, drop_aug=True, drop_meta=True)),
    ("linear ensemble (legacy .9 + ridge .1)", dict(drop_aug=True, drop_meta=True)),
    ("+ family blend only (knn off)", dict(drop_meta=True, conf=0.0)),
    ("+ kNN only (family off)", dict(drop_meta=True, fam_w=0.0)),
    ("+ family + kNN (augmentation)", dict(drop_meta=True)),
    ("+ meta GBM regression heads only", dict(gain_alpha=0.0, rank_beta=0.0)),
    ("+ meta GBM + gain heads", dict(rank_beta=0.0)),
    ("FULL deployed E43 stack", dict()),
]
print(f"honest held-out dev, artifact = {ART} (Train-only), public lookup stripped\n")
res = {}
for label, kw in CONFIGS:
    tag = label.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("+", "p").replace(".", "")
    P = predict(mutate(raw0, **kw), tag)
    res[label] = evaluate(P, label)

print("\nmarginal contribution of each layer (oracle-safety column, so the safety")
print("knob cannot mask a layer's effect):")
order = [c[0] for c in CONFIGS]
prev = None
for lab in order:
    v = res[lab][1]
    d = "" if prev is None else f"  ({v-prev:+.4f})"
    print(f"  {lab:44s} {v:.4f}{d}")
    prev = v
