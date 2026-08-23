# SPDX-License-Identifier: Apache-2.0
"""E46 - exhaustive joint sweep of the 8 post-hoc constants under the honest protocol.

E43 did this with a 3-point grid and a 200-sample bootstrap; the exact-envelope
evaluator makes each configuration cost ~1 s, so this uses a 5-9 point grid,
3 seeds x 300 bootstrap samples, coordinate descent from several starts, and a
random-restart search.  Selection is on cvEV only; dev is reported but never used
to choose.
"""
import sys, json, itertools, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, DEPLOYED_CFG
import gainlab as G

lab = Lab()
cv, arr = G.stage1(lab)
SEEDS = (7, 17, 23); NB = 300

GRID = {
    "legacy_w":       [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    "fam_w":          [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
    "conf_scale":     [0.0, 0.1, 0.2, 0.25, 0.35, 0.5, 0.7],
    "gain_alpha":     [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0],
    "rank_beta":      [0.0, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8],
    "blend_fast":     [0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0],
    "blend_balanced": [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0],
    "blend_premium":  [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
}

cache = {}
def ev(cfg):
    key = tuple(sorted(cfg.items()))
    if key not in cache:
        cache[key] = G.evaluate(lab, cv, arr, cfg, seeds=SEEDS, nboot=NB, verbose=False)
    return cache[key]

t0 = time.perf_counter()
base = ev(dict(DEPLOYED_CFG))
print(f"baseline cvEV={base['cv_ev']:.6f} dev={base['dev']:.6f} ({time.perf_counter()-t0:.1f}s/eval)", flush=True)

def descend(start, rounds=4):
    cfg = dict(start); best = ev(cfg)["cv_ev"]
    for rd in range(rounds):
        improved = False
        for name, vals in GRID.items():
            scores = []
            for v in vals:
                c2 = dict(cfg); c2[name] = v
                scores.append((ev(c2)["cv_ev"], v))
            top = max(scores)
            if top[0] > best + 2e-5:
                cfg[name] = top[1]; best = top[0]; improved = True
        if not improved:
            break
    return cfg, best

starts = [dict(DEPLOYED_CFG)]
rng = np.random.default_rng(5)
for _ in range(6):
    starts.append({k: float(rng.choice(v)) for k, v in GRID.items()})

results = []
for i, s in enumerate(starts):
    cfg, b = descend(s)
    r = ev(cfg)
    results.append((b, cfg, r))
    print(f"start{i}: cvEV={b:.6f} dev={r['dev']:.6f} cfg={json.dumps({k: round(v,3) for k,v in cfg.items()})}", flush=True)

results.sort(key=lambda x: -x[0])
print(f"\nevaluated {len(cache)} configs in {time.perf_counter()-t0:.0f}s")
print("=== top 8 distinct configs by cvEV ===")
seen = set()
top = []
for k, v in sorted(cache.items(), key=lambda kv: -kv[1]["cv_ev"])[:400]:
    key = tuple(round(x[1], 3) for x in k)
    if key in seen:
        continue
    seen.add(key); top.append((k, v))
    if len(top) >= 8:
        break
for k, v in top:
    print(f"  cvEV={v['cv_ev']:.6f} dev={v['dev']:.6f} safety={ {t: round(v['safety'][t],2) for t in TIERS} } "
          f"cfg={json.dumps({a: round(b,3) for a,b in k})}")
Path("reports/lab/e46_cfgsweep.json").write_text(json.dumps(
    [{"cfg": dict(k), "cv_ev": v["cv_ev"], "dev": v["dev"], "safety": v["safety"], "tiers": v["tiers"]}
     for k, v in sorted(cache.items(), key=lambda kv: -kv[1]["cv_ev"])[:60]], indent=2, default=float),
    encoding="utf-8")
