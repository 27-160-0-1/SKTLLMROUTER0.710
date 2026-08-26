# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a12 P5 -- the premium tier as a k1-cost problem.

(1) how big is ONE expensive k1 item relative to the whole light baseline?
(2) per-tier bust decomposition of the deployed configuration under item resampling
(3) realised efficiency of k1 as a function of the PREDICTED k1 cost decile
    (this is the runtime-observable version of the think-length law)
(4) how many k1 upgrades the premium budget can actually buy
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from labdata import load_split, TIERS, TIER_WEIGHT, TIER_MULT, tier_result, Split  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

ROOT = HERE.parents[1]
dv = load_split("dev")
tr = load_split("train")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
fam = np.array([classify_family(t) for t in dv.texts])
famtr = np.array([classify_family(t) for t in tr.texts])
fams = sorted(set(fam))
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
n = len(dv)
rng = np.random.default_rng(7)


def hdr(t):
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


hdr("P5.1  cost of a single k1 item in units of the ENTIRE light baseline of its split")
for sp, fm in ((tr, famtr), (dv, fam)):
    L = sp.cost[:, 0].sum()
    r = sp.cost[:, 2] / L
    o = np.argsort(-r)
    print(f"\n  {sp.name}: light baseline total = {L:.4f}; premium budget = 4x = {4*L:.4f}")
    print(f"   top-10 k1 items as a fraction of the light baseline (premium budget = 4.0 of these units):")
    for i in o[:10]:
        print(f"     {fm[i]:15s} out/gen={sp.otok[i,2]/sp.ngen[i,2]:8.0f} cost_k1/light_total={r[i]:.4f} "
              f"(= {r[i]/4*100:5.2f}% of the premium budget)  s_k1={sp.score[i,2]:.2f} s_mid={sp.score[i,1]:.2f}")
    print(f"   sum of the top 10 = {r[o[:10]].sum():.4f} light-units = {r[o[:10]].sum()/4*100:.1f}% of the premium budget")
    print(f"   sum of the top 30 = {r[o[:30]].sum():.4f} light-units = {r[o[:30]].sum()/4*100:.1f}% of the premium budget")
    print(f"   items with cost_k1 > 1% of the premium budget: {(r > 0.04).sum()}")
    print(f"   mean k1 cost = {sp.cost[:,2].mean()/sp.cost[:,0].mean():.1f}x mean light cost; "
          f"all-k1 ratio = {sp.cost[:,2].sum()/L:.2f} (premium cap 4.0)")
    print(f"   -> at most {int(4*L/sp.cost[:,2].mean())} of {len(sp)} items ({4*L/sp.cost[:,2].sum()*100:.1f}%) "
          f"can be k1 if k1 items were average; ")

hdr("P5.2  per-tier bust probability of the deployed configuration under item resampling of dev")
print("  (880 draws with replacement from dev; allocator re-run inside each resample)")
reps = 400
bust = {t: 0 for t in TIERS}
sc = {t: [] for t in TIERS}
ratios = {t: [] for t in TIERS}
tots = []
for _ in range(reps):
    take = rng.choice(n, size=n, replace=True)
    sub = Split(dv.name, [dv.episode_ids[i] for i in take], [dv.texts[i] for i in take],
                dv.score[take], dv.cost[take], dv.itok[take], dv.otok[take], dv.ngen[take])
    tot = 0.0
    for t in TIERS:
        r = tier_result(P[f"score_{t}"][take], P[f"cost_{t}"][take], sub, t, SAFE[t])
        bust[t] += 0 if r["passed"] else 1
        sc[t].append(r["score"])
        ratios[t].append(r["ratio"])
        tot += TIER_WEIGHT[t] * r["tier_score"]
    tots.append(tot)
for t in TIERS:
    rr = np.array(ratios[t])
    print(f"   {t:9s} bust={bust[t]/reps:6.3f}  ratio mean={rr.mean():.3f} sd={rr.std():.3f} "
          f"p95={np.percentile(rr,95):.3f} max={rr.max():.3f} cap={TIER_MULT[t]:.2f}  "
          f"score|pass={np.mean([s for s,b in zip(sc[t],[True]*reps)]):.4f}")
print(f"   weighted EV over resamples = {np.mean(tots):.4f}  (point estimate on dev itself = 0.7005)")
print("   NOTE: this resamples DEV ITEMS, which duplicates the fat k1-cost tail; the project's")
print("   official EV harness resamples the CV/OOF pool instead.  The comparison that matters")
print("   here is BETWEEN tiers, not the absolute level.")

hdr("P5.3  k1 efficiency as a function of the PREDICTED k1 cost (runtime-observable)")
Cp = P["cost_premium"]
Sp = P["score_premium"]
lightsum_pred = Cp[:, 0].sum()
rel = Cp[:, 2] / Cp[:, 0]           # predicted k1/light cost multiple
dec = np.digitize(rel, np.percentile(rel, np.arange(10, 100, 10)))
print(f"  {'dec':>3s} {'pred c2/c0':>14s} {'n':>4s} {'true c2/c0':>11s} {'true d21':>9s} "
      f"{'true d10':>9s} {'eff21 (d21/dc)':>15s} {'k1 cost share':>14s}  top families")
for d in range(10):
    m = dec == d
    dtrue = (dv.score[m, 2] - dv.score[m, 1]).mean()
    dc = (dv.cost[m, 2] - dv.cost[m, 1]).sum() / dv.cost[:, 0].mean() / m.sum()
    fs, ct = np.unique(fam[m], return_counts=True)
    top = ",".join(f"{a}:{b}" for a, b in sorted(zip(fs, ct), key=lambda z: -z[1])[:3])
    print(f"  {d:3d} {f'{rel[m].min():.1f}-{rel[m].max():.1f}':>14s} {m.sum():4d} "
          f"{dv.cost[m,2].mean()/dv.cost[m,0].mean():11.1f} {dtrue:+9.3f} "
          f"{(dv.score[m,1]-dv.score[m,0]).mean():+9.3f} {dtrue/max(dc,1e-9):15.4f} "
          f"{dv.cost[m,2].sum()/dv.cost[:,2].sum():14.3f}  {top}")

hdr("P5.4  same, but using the TRUE k1 output length (the law we would like to predict)")
og = dv.otok[:, 2] / dv.ngen[:, 2]
dec = np.digitize(og, np.percentile(og, np.arange(10, 100, 10)))
print(f"  {'dec':>3s} {'out/gen':>14s} {'n':>4s} {'true d21':>9s} {'eff21':>9s} {'k1 cost share':>14s}")
for d in range(10):
    m = dec == d
    dtrue = (dv.score[m, 2] - dv.score[m, 1]).mean()
    dc = (dv.cost[m, 2] - dv.cost[m, 1]).sum() / dv.cost[:, 0].mean() / m.sum()
    print(f"  {d:3d} {f'{og[m].min():.0f}-{og[m].max():.0f}':>14s} {m.sum():4d} {dtrue:+9.3f} "
          f"{dtrue/max(dc,1e-9):9.4f} {dv.cost[m,2].sum()/dv.cost[:,2].sum():14.3f}")

hdr("P5.5  ORACLE premium allocations: which families does a perfect router send to k1?")
o = tier_result(dv.score, dv.cost, dv, "premium", 1.0)
sel = o["sel"]
print(f"  oracle premium score={o['score']:.4f} ratio={o['ratio']:.3f}")
print(f"  {'family':16s} {'n':>4s} {'nL':>4s} {'nM':>4s} {'nK':>4s} {'k1 share of budget':>19s}")
Ltot = dv.cost[:, 0].sum()
for f in fams:
    m = fam == f
    s = sel[m]
    print(f"  {f:16s} {m.sum():4d} {(s==0).sum():4d} {(s==1).sum():4d} {(s==2).sum():4d} "
          f"{dv.cost[m][s==2,2].sum()/(4*Ltot):19.3f}")
print("\n  deployed premium selection for comparison:")
d = tier_result(P["score_premium"], P["cost_premium"], dv, "premium", SAFE["premium"])
sel2 = d["sel"]
print(f"  {'family':16s} {'nK oracle':>10s} {'nK deployed':>12s} {'overlap':>8s}")
for f in fams:
    m = fam == f
    print(f"  {f:16s} {int((sel[m]==2).sum()):10d} {int((sel2[m]==2).sum()):12d} "
          f"{int(((sel[m]==2)&(sel2[m]==2)).sum()):8d}")
print(f"  TOTAL k1 picks: oracle {(sel==2).sum()}  deployed {(sel2==2).sum()}  "
      f"overlap {((sel==2)&(sel2==2)).sum()}  precision={((sel==2)&(sel2==2)).sum()/max((sel2==2).sum(),1):.3f}")
