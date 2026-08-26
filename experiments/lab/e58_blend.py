# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""E58 - re-sweep the post-hoc constants now that the prior columns are in.

The tier blend weights were tuned (E43) for a meta stack without the offline
prior.  The prior lands entirely inside the meta features, so the optimum should
move toward the meta.  Selection is on the stress-priced train-only EV; dev is
read once at the end.
"""
import sys, json, itertools, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, DEPLOYED_EXP, DEPLOYED_CFG
import bench2 as B
import priorfeat as PF

CORE = dict(DEPLOYED_EXP, legacy_oof_meta=True, meta_seeds=(11, 23, 37, 53, 71))
lab = Lab()
pr = PF._prompts()
cols = [PF.load_column(["local-llm/labels_axlight.jsonl", "local-llm/labels_ext.jsonl"], pr),
        PF.load_column(["local-llm/labels_qwen14b.jsonl"], pr)]
lab.set_extra_features(PF.build_features(lab.texts, lab.fam_arr, cols))
cv, arr = B.stage(lab, CORE, tag="twocol")

cache = {}
def ev(cfg):
    k = tuple(sorted(cfg.items()))
    if k not in cache:
        cache[k] = B.run(lab, cv, arr, cfg, verbose=False, nboot=250)
    return cache[k]

base = ev(dict(DEPLOYED_CFG))
print(f"baseline  EV={base['EV']:.6f} dev={base['dev']:.6f} "
      f"safety={ {t: round(base['safety'][t],3) for t in TIERS} }", flush=True)

GRID = {"blend_fast":     [0.45, 0.6, 0.75, 0.85, 0.95],
        "blend_balanced": [0.3, 0.45, 0.6, 0.75, 0.9],
        "blend_premium":  [0.15, 0.3, 0.45, 0.6, 0.75],
        "gain_alpha":     [0.3, 0.5, 0.7],
        "rank_beta":      [0.2, 0.4, 0.6],
        "conf_scale":     [0.15, 0.25, 0.4],
        "legacy_w":       [0.8, 0.9, 1.0]}
cfg = dict(DEPLOYED_CFG); best = base["EV"]
t0 = time.perf_counter()
for rd in range(3):
    moved = False
    for name, vals in GRID.items():
        scores = [(ev({**cfg, name: v})["EV"], v) for v in vals]
        top = max(scores)
        vs = [s for s, _ in scores]
        unimodal = vs == sorted(vs) or vs == sorted(vs, reverse=True) or \
                   max(range(len(vs)), key=lambda i: vs[i]) not in (0, len(vs) - 1) or True
        if top[0] > best + 3e-4:
            cfg[name] = top[1]; best = top[0]; moved = True
            print(f"  round {rd}: {name} -> {top[1]} (EV {best:.6f})", flush=True)
    if not moved:
        break
r = ev(cfg)
print(f"\nswept    EV={r['EV']:.6f} dev={r['dev']:.6f} "
      f"safety={ {t: round(r['safety'][t],3) for t in TIERS} }")
print("  cfg:", json.dumps({k: round(v, 3) for k, v in cfg.items()}))
print(f"  dev tiers: " + " ".join(f"{t[:4]}={r['dev_tiers'][t]['score']:.4f}/r{r['dev_tiers'][t]['ratio']:.3f}"
                                  f"{'' if r['dev_tiers'][t]['passed'] else '!BUST'}" for t in TIERS))
print(f"  ({len(cache)} configs, {time.perf_counter()-t0:.0f}s)")
Path("reports/lab/e58_blend.json").write_text(json.dumps(
    {"baseline": {k: v for k, v in base.items() if k != "curves"},
     "swept": {k: v for k, v in r.items() if k != "curves"}, "cfg": cfg}, indent=2, default=float),
    encoding="utf-8")
lab.set_extra_features(None)
