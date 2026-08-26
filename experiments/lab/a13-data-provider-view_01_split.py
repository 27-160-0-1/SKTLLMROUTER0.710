# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a13 / provider view — part 1.

Q1  Are train and dev i.i.d. draws from one pool?  (family counts, score means,
    cost means, KS on prompt length, KS on token counts)
Q2  What is the 2:1 structure, exactly?
Q3  num_generations: distribution, per family, per model, and whether it leaks
    difficulty.
Run: .venv/Scripts/python.exe experiments/lab/a13_provider_01_split.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
sys.path.insert(0, str(ROOT / "src"))

import labdata  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

np.set_printoptions(suppress=True, precision=4)

tr, dv = labdata.load_all()
ftr = np.array([classify_family(t) for t in tr.texts])
fdv = np.array([classify_family(t) for t in dv.texts])
fams = sorted(set(ftr) | set(fdv))

print("=" * 78)
print("Q1a  per-family counts and the 2:1 hypothesis")
print("=" * 78)
ctr, cdv = Counter(ftr), Counter(fdv)
print(f"{'family':16s} {'train':>6s} {'dev':>5s} {'tr/dv':>6s} "
      f"{'exp_dv=tr/2':>11s} {'binom p':>9s}")
rows = []
for f in fams:
    a, b = ctr[f], cdv[f]
    tot = a + b
    # H0: each item of this family independently assigned to dev w.p. 1/3
    p = stats.binomtest(b, tot, 1.0 / 3.0).pvalue
    rows.append((f, a, b, tot, p))
    print(f"{f:16s} {a:6d} {b:5d} {a/max(b,1):6.3f} {a/2:11.1f} {p:9.4f}")
print(f"{'TOTAL':16s} {len(ftr):6d} {len(fdv):5d} {len(ftr)/len(fdv):6.3f}")

# chi-square: is the family composition of dev the same as train?
obs = np.array([[ctr[f] for f in fams], [cdv[f] for f in fams]], dtype=float)
chi2, pchi, dof, _ = stats.chi2_contingency(obs)
print(f"\nchi2 contingency train-vs-dev family composition: "
      f"chi2={chi2:.3f} dof={dof} p={pchi:.4f}")

# exact 2:1 per family?  how many families have train == 2*dev exactly
exact = [f for f in fams if ctr[f] == 2 * cdv[f]]
print(f"families with train == 2*dev EXACTLY: {len(exact)}/{len(fams)}  {exact}")
devs = [(f, ctr[f] - 2 * cdv[f]) for f in fams]
print("train - 2*dev per family:", devs)

print()
print("=" * 78)
print("Q1b  score / cost means per split (2-sample tests)")
print("=" * 78)
for j, m in enumerate(labdata.MODEL_IDS):
    s_t, s_d = tr.score[:, j], dv.score[:, j]
    t, p = stats.ttest_ind(s_t, s_d, equal_var=False)
    ks, pks = stats.ks_2samp(s_t, s_d)
    print(f"score {m:11s} train {s_t.mean():.4f} dev {s_d.mean():.4f} "
          f"diff {s_d.mean()-s_t.mean():+.4f}  welch p={p:.4f}  KS p={pks:.4f}")
for j, m in enumerate(labdata.MODEL_IDS):
    c_t, c_d = tr.cost[:, j], dv.cost[:, j]
    ks, pks = stats.ks_2samp(c_t, c_d)
    print(f"cost  {m:11s} train {c_t.mean():.6f} dev {c_d.mean():.6f} "
          f"ratio {c_d.mean()/c_t.mean():.4f}  KS p={pks:.4f}")

print("\nper-family score means (light/mid/k1), train vs dev, Welch p on light:")
print(f"{'family':16s} {'n_tr':>5s} {'n_dv':>5s} "
      f"{'light_tr':>9s} {'light_dv':>9s} {'p':>7s} "
      f"{'k1_tr':>7s} {'k1_dv':>7s} {'p':>7s}")
for f in fams:
    a = ftr == f
    b = fdv == f
    p0 = stats.ttest_ind(tr.score[a, 0], dv.score[b, 0], equal_var=False).pvalue
    p2 = stats.ttest_ind(tr.score[a, 2], dv.score[b, 2], equal_var=False).pvalue
    print(f"{f:16s} {a.sum():5d} {b.sum():5d} "
          f"{tr.score[a,0].mean():9.4f} {dv.score[b,0].mean():9.4f} {p0:7.3f} "
          f"{tr.score[a,2].mean():7.4f} {dv.score[b,2].mean():7.4f} {p2:7.3f}")

print()
print("=" * 78)
print("Q1c  KS on prompt length / input tokens, overall and per family")
print("=" * 78)
lt = np.array([len(t) for t in tr.texts], float)
ld = np.array([len(t) for t in dv.texts], float)
print(f"prompt chars: train mean {lt.mean():.1f} med {np.median(lt):.0f}  "
      f"dev mean {ld.mean():.1f} med {np.median(ld):.0f}  "
      f"KS D={stats.ks_2samp(lt, ld).statistic:.4f} "
      f"p={stats.ks_2samp(lt, ld).pvalue:.4f}")
print(f"light input tok: train {tr.itok[:,0].mean():.1f} dev {dv.itok[:,0].mean():.1f} "
      f"KS p={stats.ks_2samp(tr.itok[:,0], dv.itok[:,0]).pvalue:.4f}")
print(f"k1 output tok:   train {tr.otok[:,2].mean():.1f} dev {dv.otok[:,2].mean():.1f} "
      f"KS p={stats.ks_2samp(tr.otok[:,2], dv.otok[:,2]).pvalue:.4f}")
print("\nper family KS p on prompt chars:")
for f in fams:
    a = ftr == f
    b = fdv == f
    if a.sum() < 5 or b.sum() < 5:
        continue
    k = stats.ks_2samp(lt[a], ld[b])
    print(f"  {f:16s} D={k.statistic:.4f} p={k.pvalue:.4f} "
          f"(med {np.median(lt[a]):.0f} vs {np.median(ld[b]):.0f})")

print()
print("=" * 78)
print("Q3  num_generations")
print("=" * 78)
for sp, fa in ((tr, ftr), (dv, fdv)):
    print(f"-- {sp.name}")
    print("   ngen value counts per model:")
    for j, m in enumerate(labdata.MODEL_IDS):
        print(f"     {m:11s} {dict(Counter(sp.ngen[:, j].astype(int)))}")
    same = (sp.ngen[:, 0] == sp.ngen[:, 1]) & (sp.ngen[:, 1] == sp.ngen[:, 2])
    print(f"   ngen identical across the 3 models: {same.mean()*100:.2f}%")
    print(f"   {'family':16s} {'n':>5s} {'frac ngen=4':>12s}")
    for f in fams:
        a = fa == f
        print(f"   {f:16s} {a.sum():5d} {(sp.ngen[a,0]==4).mean():12.3f}")

# does ngen leak difficulty *within* the families that mix 2 and 4?
print("\nWithin gsm8k_or_other (the only mixed family): ngen=2 vs ngen=4 items")
for sp, fa in ((tr, ftr), (dv, fdv)):
    a = fa == "gsm8k_or_other"
    g4 = a & (sp.ngen[:, 0] == 4)
    g2 = a & (sp.ngen[:, 0] == 2)
    print(f" {sp.name}: n4={g4.sum()} n2={g2.sum()}")
    for j, m in enumerate(labdata.MODEL_IDS):
        p = stats.ttest_ind(sp.score[g4, j], sp.score[g2, j], equal_var=False).pvalue
        print(f"   {m:11s} score ngen4 {sp.score[g4,j].mean():.4f} "
              f"ngen2 {sp.score[g2,j].mean():.4f} p={p:.4g}")
    lc = np.array([len(t) for t in sp.texts], float)
    print(f"   prompt chars ngen4 {lc[g4].mean():.0f} ngen2 {lc[g2].mean():.0f}")
    print(f"   k1 outtok/gen ngen4 {(sp.otok[g4,2]/4).mean():.0f} "
          f"ngen2 {(sp.otok[g2,2]/2).mean():.0f}")
