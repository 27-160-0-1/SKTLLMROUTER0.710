# SPDX-License-Identifier: Apache-2.0
"""a05 step 8: how much of the gain signal is family-level vs item-level?
Also verifies the two allocator invariances analytically claimed in the report."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.similarity import classify_family

tr = load_split("train"); dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
Cc = np.load(ROOT / "experiments/lab/a05-selection-failure_cache.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
N = len(dv); IDX = np.arange(N); FAM = Cc["fam"]; PHAT = Cc["phat"]
FAMTR = np.array([classify_family(t) for t in tr.texts])
L = dv.cost[:, 0].sum()


def auc(sc, lab):
    lab = np.asarray(lab, bool)
    n1 = lab.sum(); n0 = (~lab).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(sc)) + 1.0
    return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


print("=== (1) allocator invariances (empirical verification) ===")
S = P["score_premium"]; Cst = P["cost_premium"]
r0 = tier_result(S, Cst, dv, "premium", SAFE["premium"])
S1 = S + np.random.default_rng(0).normal(0, 1, (N, 1))     # per-item constant on all 3
r1 = tier_result(S1, Cst, dv, "premium", SAFE["premium"])
g1 = S[:, 1] - S[:, 0]; g2 = S[:, 2] - S[:, 1]
S2 = np.column_stack([S[:, 0], S[:, 0] + 3.0 * g1, S[:, 0] + 3.0 * (g1 + g2)])
r2 = tier_result(S2, Cst, dv, "premium", SAFE["premium"])
print(f"  I1 add per-item constant to all 3 scores : selections identical = {(r0['sel']==r1['sel']).all()}")
print(f"  I2 multiply BOTH gains by 3.0           : selections identical = {(r0['sel']==r2['sel']).all()}")
print("  => the allocator sees only the 2-vector (g1,g2) per item, up to one global scale.")

print("\n=== (2) family-level vs item-level content of the gain signals ===")
fm1 = {}; fm2 = {}
for f_ in set(FAMTR.tolist()):
    m = FAMTR == f_
    fm1[f_] = (tr.score[m, 1] - tr.score[m, 0]).mean()
    fm2[f_] = (tr.score[m, 2] - tr.score[m, 1]).mean()
FG1 = np.array([fm1.get(f, 0.0) for f in FAM])
FG2 = np.array([fm2.get(f, 0.0) for f in FAM])
PG1 = P["score_premium"][:, 1] - P["score_premium"][:, 0]
PG2 = P["score_premium"][:, 2] - P["score_premium"][:, 1]
lab1 = dv.score[:, 1] > dv.score[:, 0]
lab2 = dv.score[:, 2] > dv.score[:, 1]
elab1 = PHAT[:, 1] - PHAT[:, 0]
elab2 = PHAT[:, 2] - PHAT[:, 1]
print(f"  {'signal':28s} {'AUC(mid>light)':>15s} {'AUC(k1>mid)':>12s} "
      f"{'spear EBg1':>11s} {'spear EBg2':>11s}")
for nm, a, b in (("train family-mean gain", FG1, FG2),
                 ("deployed predicted gain", PG1, PG2),
                 ("deployed gain, family-mean removed",
                  PG1 - np.array([PG1[FAM == f].mean() for f in FAM]),
                  PG2 - np.array([PG2[FAM == f].mean() for f in FAM]))):
    print(f"  {nm:34s} {auc(a, lab1):15.3f} {auc(b, lab2):12.3f} "
          f"{spearmanr(a, elab1).statistic:11.3f} {spearmanr(b, elab2).statistic:11.3f}")

print("\n  within-family AUC (pooled over families, family effect removed):")
for nm, a, b in (("deployed predicted gain", PG1, PG2),):
    tot1 = w1 = tot2 = w2 = 0.0
    for f_ in sorted(set(FAM.tolist())):
        m = FAM == f_
        u1 = auc(a[m], lab1[m]); u2 = auc(b[m], lab2[m])
        n1 = (lab1[m].sum() * (~lab1[m]).sum()); n2 = (lab2[m].sum() * (~lab2[m]).sum())
        print(f"    {f_:15s} n={m.sum():4d}  AUC1={u1 if u1==u1 else float('nan'):.3f} "
              f"AUC2={u2 if u2==u2 else float('nan'):.3f}")
        if u1 == u1: tot1 += u1 * n1; w1 += n1
        if u2 == u2: tot2 += u2 * n2; w2 += n2
    print(f"    {'POOLED':15s}          AUC1={tot1/w1:.3f} AUC2={tot2/w2:.3f}")

print("\n=== (3) what does a family-only router score?  (train-fitted family gains) ===")
def run(mk_s, tune=True):
    tot_r = tot_e = 0.0; sfs = []
    for t in TIERS:
        if tune:
            best = None; bsf = None
            for sf in np.arange(0.5, 1.301, 0.005):
                r = tier_result(mk_s(t), P[f"cost_{t}"], dv, t, float(sf))
                if r["passed"] and (best is None or r["score"] > best["score"]):
                    best = r; bsf = float(sf)
            r = best; sfs.append(bsf)
        else:
            r = tier_result(mk_s(t), P[f"cost_{t}"], dv, t, SAFE[t]); sfs.append(SAFE[t])
        tot_r += TIER_WEIGHT[t] * r["tier_score"]
        tot_e += TIER_WEIGHT[t] * (PHAT[IDX, r["sel"]].mean() if r["passed"] else 0.0)
    return tot_r, tot_e, sfs

def fam_only(t):
    S = np.zeros((N, 3))
    S[:, 1] = FG1; S[:, 2] = FG1 + FG2
    return S
def item_only(t):
    S = P[f"score_{t}"].copy()
    a = S[:, 1] - S[:, 0]; b = S[:, 2] - S[:, 1]
    a = a - np.array([a[FAM == f].mean() for f in FAM]) + FG1
    b = b - np.array([b[FAM == f].mean() for f in FAM]) + FG2
    S[:, 1] = S[:, 0] + a; S[:, 2] = S[:, 1] + b
    return S
for nm, mk in (("deployed predictions", lambda t: P[f"score_{t}"]),
               ("train family-mean gains only", fam_only),
               ("deployed gains, family means replaced by train family means", item_only)):
    r, e, sfs = run(mk)
    print(f"  {nm:60s} realised={r:.4f} EB={e:.4f} safety={'/'.join(f'{x:.3f}' for x in sfs)}")

print("\n=== (4) upper bound if only the mid-vs-light decision were solved WITHIN family ===")
# replace g1 by the EB truth but keep the family mean of the prediction (no family-level leak)
def mk(t):
    S = P[f"score_{t}"].copy()
    tg = PHAT[:, 1] - PHAT[:, 0]
    a = tg - np.array([tg[FAM == f].mean() for f in FAM]) + \
        np.array([(S[:, 1] - S[:, 0])[FAM == f].mean() for f in FAM])
    b = S[:, 2] - S[:, 1]
    S[:, 1] = S[:, 0] + a; S[:, 2] = S[:, 1] + b
    return S
r, e, sfs = run(mk)
print(f"  perfect WITHIN-family mid-light gain (EB target): realised={r:.4f} EB={e:.4f} "
      f"safety={'/'.join(f'{x:.3f}' for x in sfs)}")
def mk2(t):
    S = P[f"score_{t}"].copy()
    tg = PHAT[:, 2] - PHAT[:, 1]
    b = tg - np.array([tg[FAM == f].mean() for f in FAM]) + \
        np.array([(S[:, 2] - S[:, 1])[FAM == f].mean() for f in FAM])
    S[:, 2] = S[:, 1] + b
    return S
r, e, sfs = run(mk2)
print(f"  perfect WITHIN-family k1-mid gain   (EB target): realised={r:.4f} EB={e:.4f} "
      f"safety={'/'.join(f'{x:.3f}' for x in sfs)}")
