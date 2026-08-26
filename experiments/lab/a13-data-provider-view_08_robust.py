# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a13 / provider view - part 8: is the high bust probability a bootstrap
artefact (duplicate draws of one very expensive item), or real?

Checks
  1. subsample WITHOUT replacement (no duplicates possible) vs with replacement
  2. drop the single most influential item and repeat
  3. family-composition stress: what the provider could plausibly change
  4. the safety factor at which the REAL dev busts (how thin is the margin)
Run: .venv/Scripts/python.exe experiments/lab/a13_provider_08_robust.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
sys.path.insert(0, str(ROOT / "src"))

import labdata  # noqa: E402
from labdata import TIERS, TIER_MULT, TIER_WEIGHT, allocate, tier_result  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

dv = labdata.load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
fam = np.array([classify_family(t) for t in dv.texts])


def run_batch(tier, ix, safety):
    ps, pc = P[f"score_{tier}"][ix], P[f"cost_{tier}"][ix]
    tc, ts = dv.cost[ix], dv.score[ix]
    sel = allocate(ps, pc, tc, TIER_MULT[tier], safety)
    idx = np.arange(len(ix))
    ratio = tc[idx, sel].sum() / tc[:, 0].sum()
    ok = ratio <= TIER_MULT[tier] + 1e-15
    return ratio, (ts[idx, sel].mean() if ok else 0.0), ok


print("=" * 88)
print("1. WITH vs WITHOUT replacement (n=440 and n=660 out of dev's 880)")
print("=" * 88)
rng = np.random.default_rng(11)
for n in (440, 660):
    for tier in TIERS:
        rw, ro = [], []
        for _ in range(1500):
            rw.append(run_batch(tier, rng.integers(0, 880, n), SAFE[tier])[2])
            ro.append(run_batch(tier, rng.permutation(880)[:n], SAFE[tier])[2])
        print(f"  n={n} {tier:9s} bust with-replacement {100*(1-np.mean(rw)):5.1f}%   "
              f"without-replacement {100*(1-np.mean(ro)):5.1f}%")

print()
print("=" * 88)
print("2. drop the most influential single item (dev idx 204, k1 cost 0.906)")
print("=" * 88)
worst = int(np.argmax(dv.cost[:, 2]))
print(f"  most expensive k1 item: idx {worst}, k1 cost {dv.cost[worst,2]:.4f} "
      f"= {dv.cost[worst,2]/dv.cost[:,0].sum():.3f}x the whole light budget, "
      f"family {fam[worst]}, out/gen {dv.otok[worst,2]/dv.ngen[worst,2]:.0f}")
keep = np.setdiff1d(np.arange(880), [worst])
for tier in TIERS:
    b_all, b_drop = [], []
    rng = np.random.default_rng(5)
    for _ in range(1500):
        b_all.append(run_batch(tier, rng.integers(0, 880, 880), SAFE[tier])[2])
        b_drop.append(run_batch(tier, keep[rng.integers(0, 879, 880)], SAFE[tier])[2])
    print(f"  {tier:9s} bust all {100*(1-np.mean(b_all)):5.1f}%   "
          f"without the worst item {100*(1-np.mean(b_drop)):5.1f}%")

print()
print("=" * 88)
print("3. how thin is the safety margin on the REAL dev set?")
print("=" * 88)
for tier in TIERS:
    lo, hi = 0.5, 1.5
    last_ok = None
    for sf in np.arange(0.70, 1.201, 0.005):
        r = tier_result(P[f"score_{tier}"], P[f"cost_{tier}"], dv, tier, float(sf))
        if r["passed"]:
            last_ok = (float(sf), r["ratio"], r["score"])
        else:
            print(f"  {tier:9s} deployed {SAFE[tier]:.2f} -> ratio "
                  f"{tier_result(P[f'score_{tier}'], P[f'cost_{tier}'], dv, tier, SAFE[tier])['ratio']:.4f}; "
                  f"last passing safety {last_ok[0]:.3f} (ratio {last_ok[1]:.4f}, "
                  f"score {last_ok[2]:.4f}); first busting {sf:.3f}")
            break

print()
print("=" * 88)
print("4. family-composition stress (provider may reweight the private mix)")
print("=" * 88)
fams = sorted(set(fam))
scen = {"nominal": {}}
for f in fams:
    scen[f"2x {f}"] = {f: 2.0}
scen["no hrmcr+longdoc"] = {"hrmcr": 0.0, "longdoc": 0.0}
scen["2x aime+dmmath+code"] = {"aime": 2.0, "dmmath": 2.0, "code": 2.0}
for name, w in scen.items():
    p = np.ones(880)
    for f, mult in w.items():
        p[fam == f] *= mult
    if p.sum() == 0:
        continue
    p = p / p.sum()
    rng = np.random.default_rng(3)
    tot, busts = 0.0, {}
    for tier in TIERS:
        evs, ok = [], []
        for _ in range(400):
            ix = rng.choice(880, 880, p=p)
            _, s, o = run_batch(tier, ix, SAFE[tier])
            evs.append(s)
            ok.append(o)
        tot += TIER_WEIGHT[tier] * np.mean(evs)
        busts[tier] = 100 * (1 - np.mean(ok))
    print(f"  {name:22s} weighted EV {tot:.4f}   bust% "
          f"fast {busts['fast']:5.1f} bal {busts['balanced']:5.1f} prem {busts['premium']:5.1f}")
