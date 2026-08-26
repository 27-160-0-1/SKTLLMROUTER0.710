# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a11 step 14 - context-limit runaways: cost, score, and the value of a perfect
veto (an upper bound on what a runaway-detector head could be worth)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
from labdata import load_all, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router import similarity

ROOT = Path(__file__).resolve().parents[2]
tr, dv = load_all()
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv)
DEP = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
famd = np.array([similarity.classify_family(x) for x in dv.texts])
famt = np.array([similarity.classify_family(x) for x in tr.texts])

print("=== runaway items (k1 output tokens per generation >= 32,000) ===")
for sp, fm in ((tr, famt), (dv, famd)):
    m = (sp.otok[:, 2] / sp.ngen[:, 2]) >= 32000
    print(f"  {sp.name}: {int(m.sum())} items, families {sorted(set(fm[m]))}, "
          f"k1 scores {sp.score[m,2].tolist()}, light scores {sp.score[m,0].tolist()}")
    print(f"      per-item k1 cost / light-total = "
          f"{np.round(sp.cost[m,2]/sp.cost[:,0].sum(), 4).tolist()}")
    for f in sorted(set(fm[m])):
        print(f"      P(runaway | family={f}) = {int((m & (fm==f)).sum())}/{int((fm==f).sum())} "
              f"= {100*(m & (fm==f)).sum()/max((fm==f).sum(),1):.2f}%")

print("\n=== near-runaways: the same statistic at lower thresholds (train+dev) ===")
for thr in (32000, 20000, 10000, 5000):
    a = (tr.otok[:, 2] / tr.ngen[:, 2]) >= thr
    b = (dv.otok[:, 2] / dv.ngen[:, 2]) >= thr
    tot = int(a.sum() + b.sum())
    sc = np.concatenate([tr.score[a, 2], dv.score[b, 2]])
    cs = (tr.cost[a, 2].sum() / tr.cost[:, 0].sum() + dv.cost[b, 2].sum() / dv.cost[:, 0].sum())
    print(f"  out_tok/gen >= {thr:6d}: {tot:4d} items ({100*tot/(len(tr)+len(dv)):.2f}%), "
          f"mean k1 score {sc.mean():.3f}, mean k1 cost {np.mean(np.concatenate([tr.cost[a,2],dv.cost[b,2]])):.4f} "
          f"(= {cs/2:.3f} of a light-total each, on average across the two splits)")

print("\n=== are runaways selected by the deployed router?  (dev) ===")
run = (dv.otok[:, 2] / dv.ngen[:, 2]) >= 32000
for t in TIERS:
    sel = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, DEP[t])["sel"]
    hit = int(((sel == 2) & run).sum())
    print(f"  {t:9s}: {hit} of the {int(run.sum())} runaways selected as k1; "
          f"predicted k1 score for them = {np.round(P[f'score_{t}'][run,2],3).tolist()}; "
          f"predicted k1 cost / true = {np.round(P[f'cost_{t}'][run,2]/dv.cost[run,2],3).tolist()}")

print("\n=== value of a PERFECT runaway veto (oracle upper bound, dev) ===")
print("   veto = model 2 forbidden for the 3 known runaway items, safety re-tuned")
for t in TIERS:
    mult = TIER_MULT[t]
    for label, veto in (("no veto", np.zeros(n, bool)), ("oracle veto", run)):
        pc = P[f"cost_{t}"].copy()
        pc[veto, 2] = 1e9
        best = None
        for s in np.arange(0.60, 1.401, 0.005):
            r = tier_result(P[f"score_{t}"], pc, dv, t, float(s))
            if r["passed"] and (best is None or r["score"] > best[1]):
                best = (float(s), r["score"], r["ratio"])
        print(f"  {t:9s} {label:12s} best passing s={best[0]:.3f} score={best[1]:.4f} "
              f"ratio={best[2]:.4f}")

print("\n=== what does a runaway cost when it IS selected? (premium, forced) ===")
for i in np.flatnonzero(run):
    print(f"  item {i} fam={famd[i]} true k1 cost={dv.cost[i,2]:.4f} "
          f"= {dv.cost[i,2]/dv.cost[:,0].sum():.4f} of the light total "
          f"({100*dv.cost[i,2]/(4.0*dv.cost[:,0].sum()):.1f}% of the premium cap), "
          f"true k1 score={dv.score[i,2]:.2f}, pred k1 score={P['score_premium'][i,2]:.3f}")
