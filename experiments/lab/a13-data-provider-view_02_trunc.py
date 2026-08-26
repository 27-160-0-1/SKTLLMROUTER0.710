# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a13 / provider view — part 2: the 32,768-token common-context condition.

Q  Do per-generation (input+output) token counts pile up at 32,768?
Q  What does a truncated generation score?  Is truncation ~= certain 0?
Q  Is truncation predictable from the prompt (family / length)?
Q  How much budget does the truncated tail consume?
Run: .venv/Scripts/python.exe experiments/lab/a13_provider_02_trunc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
sys.path.insert(0, str(ROOT / "src"))

import labdata  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

np.set_printoptions(suppress=True, precision=4)
CAP = 32768

tr, dv = labdata.load_all()
fam = {"train": np.array([classify_family(t) for t in tr.texts]),
       "dev": np.array([classify_family(t) for t in dv.texts])}
fams = sorted(set(fam["train"]))

print("=" * 78)
print("A. per-generation token accounting; is 32768 a hard ceiling?")
print("=" * 78)
for sp in (tr, dv):
    print(f"-- {sp.name}")
    for j, m in enumerate(labdata.MODEL_IDS):
        ipg = sp.itok[:, j] / sp.ngen[:, j]
        opg = sp.otok[:, j] / sp.ngen[:, j]
        tot = ipg + opg
        print(f"   {m:11s} in/gen max {ipg.max():8.1f}  out/gen max {opg.max():9.1f}  "
              f"(in+out)/gen max {tot.max():9.1f}  p99 {np.percentile(tot,99):8.1f}")
        # integrality: is out/gen an integer?  (evidence they are per-gen sums)
        frac = np.abs(opg - np.round(opg))
        print(f"                out/gen non-integer rows: {(frac>1e-9).sum()}  "
              f"| rows with (in+out)/gen > {CAP}: {(tot > CAP + .5).sum()}")

print()
print("=" * 78)
print("B. k1 truncation: how close to the ceiling, and what does it score?")
print("=" * 78)
for sp in (tr, dv):
    f = fam[sp.name]
    ipg = sp.itok[:, 2] / sp.ngen[:, 2]
    opg = sp.otok[:, 2] / sp.ngen[:, 2]
    tot = ipg + opg
    print(f"-- {sp.name}  n={len(sp)}")
    for thr in (0.999, 0.99, 0.95, 0.90, 0.75, 0.50):
        m = tot >= CAP * thr
        if m.sum() == 0:
            print(f"   ctx>= {thr:.3f}*cap : n=0")
            continue
        print(f"   ctx>= {thr:5.3f}*cap : n={m.sum():4d} "
              f"k1 score {sp.score[m,2].mean():.4f} "
              f"(light {sp.score[m,0].mean():.4f} mid {sp.score[m,1].mean():.4f}) "
              f"| k1 cost x-light {(sp.cost[m,2]/sp.cost[m,0]).mean():6.1f} "
              f"| share of total k1 cost {sp.cost[m,2].sum()/sp.cost[:,2].sum()*100:5.1f}%")
    # exact ceiling hits
    hit = tot >= CAP - 1.0
    print(f"   EXACT ceiling (>= {CAP}-1): n={hit.sum()}")
    if hit.sum():
        print(f"      k1 score mean {sp.score[hit,2].mean():.4f}  "
              f"score histogram {np.bincount((sp.score[hit,2]*4).astype(int), minlength=5)}")
        print(f"      families: {dict(zip(*np.unique(fam[sp.name][hit], return_counts=True)))}")
        print(f"      light score {sp.score[hit,0].mean():.4f} mid {sp.score[hit,1].mean():.4f}")

print()
print("=" * 78)
print("C. long-output tail regardless of ceiling: score vs out/gen decile (k1)")
print("=" * 78)
for sp in (tr, dv):
    opg = sp.otok[:, 2] / sp.ngen[:, 2]
    q = np.quantile(opg, np.linspace(0, 1, 11))
    print(f"-- {sp.name}")
    print(f"   {'decile':>7s} {'out/gen lo':>11s} {'hi':>9s} {'k1 score':>9s} "
          f"{'light':>7s} {'mid':>7s} {'k1 cost x light':>16s}")
    for d in range(10):
        m = (opg >= q[d]) & (opg <= q[d + 1] if d == 9 else opg < q[d + 1])
        print(f"   {d:7d} {q[d]:11.0f} {q[d+1]:9.0f} {sp.score[m,2].mean():9.4f} "
              f"{sp.score[m,0].mean():7.4f} {sp.score[m,1].mean():7.4f} "
              f"{(sp.cost[m,2]/sp.cost[m,0]).mean():16.1f}")

print()
print("=" * 78)
print("D. per-family truncation rate (k1) and the value of a 'never-k1' rule")
print("=" * 78)
for sp in (tr, dv):
    f = fam[sp.name]
    tot = (sp.itok[:, 2] + sp.otok[:, 2]) / sp.ngen[:, 2]
    print(f"-- {sp.name}")
    print(f"   {'family':16s} {'n':>5s} {'trunc>=0.99cap':>15s} {'score|trunc':>12s} "
          f"{'score|not':>10s}")
    for fx in fams:
        m = f == fx
        t = m & (tot >= CAP * 0.99)
        s_t = sp.score[t, 2].mean() if t.sum() else float("nan")
        s_n = sp.score[m & ~t, 2].mean() if (m & ~t).sum() else float("nan")
        print(f"   {fx:16s} {m.sum():5d} {t.sum():15d} {s_t:12.4f} {s_n:10.4f}")

print()
print("=" * 78)
print("E. what fraction of the k1 *cost* sits in generations that produced 0?")
print("=" * 78)
for sp in (tr, dv):
    zero = sp.score[:, 2] == 0
    print(f"-- {sp.name}: k1 score==0 rows {zero.sum():4d} "
          f"({zero.mean()*100:.1f}%), holding {sp.cost[zero,2].sum()/sp.cost[:,2].sum()*100:.1f}% "
          f"of total k1 cost; mean out/gen {(sp.otok[zero,2]/sp.ngen[zero,2]).mean():.0f} "
          f"vs {(sp.otok[~zero,2]/sp.ngen[~zero,2]).mean():.0f} for score>0")
