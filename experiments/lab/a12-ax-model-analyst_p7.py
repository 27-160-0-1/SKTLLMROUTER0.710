# SPDX-License-Identifier: Apache-2.0
"""a12 P7 -- the upgrade profile in the ALLOCATOR'S OWN CURRENCY.

Correction to P1.8: the budget denominator is the light cost of the WHOLE batch,
not of the family.  So the economically correct efficiency of an upgrade is

    eff = E[delta score] / ( E[delta cost] / mean_batch_light_cost )

i.e. score gained per 1% of the light baseline spent.  Under the per-family
normalisation belebele looks like the worst k1 buy; under the correct one it is
one of the best, because belebele prompts are cheap in absolute terms while
longdoc prompts (44% of the whole light baseline) are not.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from labdata import load_split, TIERS, TIER_WEIGHT, TIER_MULT, tier_result  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

dv, tr = load_split("dev"), load_split("train")
fam = np.concatenate([[classify_family(t) for t in tr.texts], [classify_family(t) for t in dv.texts]])
S = np.vstack([tr.score, dv.score])
C = np.vstack([tr.cost, dv.cost])
IT = np.vstack([tr.itok, dv.itok]) / np.vstack([tr.ngen, dv.ngen])
OT = np.vstack([tr.otok, dv.otok]) / np.vstack([tr.ngen, dv.ngen])
NG = np.vstack([tr.ngen, dv.ngen])[:, 0].astype(int)
fams = sorted(set(fam))
N = len(fam)
CL = C[:, 0].mean()          # mean light cost of the batch = the budget unit


def hdr(t):
    print("\n" + "=" * 112)
    print(t)
    print("=" * 112)


hdr("P7.1  budget anatomy: who PAYS for the budget and who SPENDS it")
print(f"  the budget unit is the batch mean light cost = {CL:.6f}; "
      f"budget(premium) = 4.0 x N x unit")
print(f"  {'family':16s} {'n':>5s} {'share of items':>15s} {'share of the light baseline':>28s} "
      f"{'mean light cost / unit':>23s} {'cost of all-k1 (light units)':>29s}")
for f in fams:
    m = fam == f
    print(f"  {f:16s} {m.sum():5d} {m.mean():15.3f} {C[m,0].sum()/C[:,0].sum():28.3f} "
          f"{C[m,0].mean()/CL:23.2f} {C[m,2].sum()/C[:,0].sum():29.3f}")
print(f"  {'TOTAL':16s} {N:5d} {1.0:15.3f} {1.0:28.3f} {1.0:23.2f} {C[:,2].sum()/C[:,0].sum():29.3f}")

hdr("P7.2  CORRECT upgrade economics (score gained per 1% of the light baseline spent)")
print("  eff = mean(delta score) / (mean(delta cost)/unit) * 100  ->  'score per 100 budget-units'")
rows = []
for f in fams:
    m = fam == f
    d10 = (S[m, 1] - S[m, 0]).mean()
    d21 = (S[m, 2] - S[m, 1]).mean()
    d20 = (S[m, 2] - S[m, 0]).mean()
    dc10 = (C[m, 1] - C[m, 0]).mean() / CL
    dc21 = (C[m, 2] - C[m, 1]).mean() / CL
    dc20 = (C[m, 2] - C[m, 0]).mean() / CL
    rows.append((f, int(m.sum()), d10, dc10, d10 / dc10, d21, dc21, d21 / dc21, d20, dc20, d20 / dc20))
print(f"  {'family':16s} {'n':>5s} | {'d10':>7s} {'dc10':>7s} {'eff10':>8s} | {'d21':>7s} {'dc21':>7s} "
      f"{'eff21':>8s} | {'d20':>7s} {'dc20':>7s} {'eff20':>8s}")
for r in sorted(rows, key=lambda z: -z[7]):
    print(f"  {r[0]:16s} {r[1]:5d} | {r[2]:+7.3f} {r[3]:7.2f} {r[4]:8.4f} | {r[5]:+7.3f} {r[6]:7.2f} "
          f"{r[7]:8.4f} | {r[8]:+7.3f} {r[9]:7.2f} {r[10]:8.4f}")
d10 = (S[:, 1] - S[:, 0]).mean(); dc10 = (C[:, 1] - C[:, 0]).mean() / CL
d21 = (S[:, 2] - S[:, 1]).mean(); dc21 = (C[:, 2] - C[:, 1]).mean() / CL
print(f"  {'ALL':16s} {N:5d} | {d10:+7.3f} {dc10:7.2f} {d10/dc10:8.4f} | {d21:+7.3f} {dc21:7.2f} "
      f"{d21/dc21:8.4f} |")

hdr("P7.3  the SAME ranking under the wrong (per-family) normalisation, for contrast")
print(f"  {'family':16s} {'eff21 correct':>14s} {'eff21 per-family':>18s} {'rank shift'}")
corr_rank = {r[0]: i for i, r in enumerate(sorted(rows, key=lambda z: -z[7]))}
wrong = []
for f in fams:
    m = fam == f
    e = (S[m, 2] - S[m, 1]).mean() / ((C[m, 2] - C[m, 1]).mean() / C[m, 0].mean())
    wrong.append((f, e))
wrong_rank = {f: i for i, (f, _) in enumerate(sorted(wrong, key=lambda z: -z[1]))}
for r in sorted(rows, key=lambda z: -z[7]):
    f = r[0]
    print(f"  {f:16s} {r[7]:14.4f} {dict(wrong)[f]:18.4f} "
          f"{corr_rank[f]+1} -> {wrong_rank[f]+1}")

hdr("P7.4  same table by prompt-length quintile and by language and by ngen")
for name, groups in (
        ("input-token quintile", [(f"Q{i+1}", np.digitize(IT[:, 0], np.percentile(IT[:, 0], [20, 40, 60, 80])) == i)
                                  for i in range(5)]),
        ("language", [("en", np.array([sum(1 for ch in t if 0xAC00 <= ord(ch) <= 0xD7A3) / max(len(t), 1) <= 0.05
                                       for t in list(tr.texts) + list(dv.texts)])),
                      ("ko", np.array([sum(1 for ch in t if 0xAC00 <= ord(ch) <= 0xD7A3) / max(len(t), 1) > 0.05
                                       for t in list(tr.texts) + list(dv.texts)]))]),
        ("num_generations", [("ngen=2", NG == 2), ("ngen=4", NG == 4)])):
    print(f"\n  -- {name}")
    print(f"  {'group':10s} {'n':>5s} | {'d10':>7s} {'dc10':>7s} {'eff10':>8s} | {'d21':>7s} {'dc21':>7s} {'eff21':>8s}")
    for g, m in groups:
        if m.sum() == 0:
            continue
        a = (S[m, 1] - S[m, 0]).mean(); ca = (C[m, 1] - C[m, 0]).mean() / CL
        b = (S[m, 2] - S[m, 1]).mean(); cb = (C[m, 2] - C[m, 1]).mean() / CL
        print(f"  {g:10s} {m.sum():5d} | {a:+7.3f} {ca:7.2f} {a/ca:8.4f} | {b:+7.3f} {cb:7.2f} {b/cb:8.4f}")

hdr("P7.5  marginal-value curve: exact Lagrangian sweep with the TRUE score/cost matrix")
print(f"  {'lambda':>10s} {'spent (x light)':>16s} {'mean score':>11s} {'#mid':>6s} {'#k1':>6s}  k1 basket by family")
L = C[:, 0].sum()
prev = None
for lam in [3.0, 1.0, 0.3, 0.1, 0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001, 0.0]:
    util = S - lam * C / L * N
    sel = util.argmax(axis=1)
    spent = C[np.arange(N), sel].sum() / L
    sc = S[np.arange(N), sel].mean()
    k1 = sel == 2
    top = ",".join(f"{k}:{v}" for k, v in sorted(
        [(f, int(((fam == f) & k1).sum())) for f in fams], key=lambda z: -z[1])[:5] if v > 0)
    print(f"  {lam:10.4f} {spent:16.3f} {sc:11.4f} {int((sel==1).sum()):6d} {int(k1.sum()):6d}  {top}")
print("  the premium cap is 4.0 light-units, balanced 2.0, fast 1.25 -> read off the rows.")
print("  NOTE: TRUE-score oracle, inflated by label noise; the transferable part is WHICH")
print("  families enter the k1 basket first.")
