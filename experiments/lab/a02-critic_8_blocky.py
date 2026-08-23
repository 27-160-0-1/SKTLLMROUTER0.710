# SPDX-License-Identifier: Apache-2.0
"""a02-critic #8.  Is the deployed router effectively a per-family policy?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, tier_result
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
fam = np.array([classify_family(t) for t in dv.texts])
fams = sorted(set(fam))
N = len(dv)

for t in TIERS:
    r = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, SAFE[t])
    sel = r["sel"]
    # entropy-style measure: how much of the selection is predicted by family alone?
    maj = np.zeros(N, dtype=int)
    for f in fams:
        m = fam == f
        maj[m] = np.bincount(sel[m], minlength=3).argmax()
    print(f"\n{t}: selection = {np.bincount(sel, minlength=3)}  "
          f"(family-majority rule reproduces {np.mean(maj == sel)*100:.1f}% of the picks)")
    print(f"  {'family':11s} {'n':>4s} {'%light':>7s} {'%mid':>6s} {'%k1':>5s} "
          f"{'true gain m-l':>13s} {'true gain k-m':>13s}")
    for f in fams:
        m = fam == f
        s = sel[m]
        print(f"  {f:11s} {m.sum():4d} {np.mean(s==0)*100:6.1f}% {np.mean(s==1)*100:5.1f}% "
              f"{np.mean(s==2)*100:4.1f}% {(dv.score[m,1]-dv.score[m,0]).mean():13.3f} "
              f"{(dv.score[m,2]-dv.score[m,1]).mean():13.3f}")

print("\n=== does the router beat a per-family quota policy? ===")
print("(same per-family upgrade counts, but items chosen at RANDOM inside each family)")
rng = np.random.default_rng(0)
for t in TIERS:
    r = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, SAFE[t])
    sel = r["sel"]
    real = dv.score[np.arange(N), sel].mean()
    vals = []
    for _ in range(400):
        s2 = np.zeros(N, dtype=int)
        for f in fams:
            idx = np.where(fam == f)[0]
            rng.shuffle(idx)
            cnt = np.bincount(sel[fam == f], minlength=3)
            pos = 0
            for j in (2, 1):
                s2[idx[pos:pos + cnt[j]]] = j
                pos += cnt[j]
        vals.append(dv.score[np.arange(N), s2].mean())
    print(f"  {t:9s} deployed score={real:.4f}   family-quota+random-inside={np.mean(vals):.4f}"
          f" +- {np.std(vals):.4f}   edge={real-np.mean(vals):+.4f}")
