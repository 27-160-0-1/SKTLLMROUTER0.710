# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a13 / provider view - part 5: how pure are the 9 regex 'families'?

The provider handed us two exact source labels: data/sources/
deepmind-mathematics-selection.v1.json (456 episode_ids) and
data/{train,dev}/aime-selection.json (36 episode_ids).  Use them as ground truth
for the regex classifier used by the deployed pipeline (family means, one-hot).

Run: .venv/Scripts/python.exe experiments/lab/a13_provider_05_family.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
sys.path.insert(0, str(ROOT / "src"))

import labdata  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

tr, dv = labdata.load_all()
splits = {"train": tr, "dev": dv}

# ---- provider-supplied ground truth ---------------------------------------
truth = {}
dm = json.loads((ROOT / "data/sources/deepmind-mathematics-selection.v1.json").read_text(encoding="utf-8"))
for sp, rows in dm["splits"].items():
    for r in rows:
        truth[r["episode_id"]] = ("dmmath", r["sample_id"].split(":")[1])
for sp in ("train", "dev"):
    sel = json.loads((ROOT / f"data/{sp}/aime-selection.json").read_text(encoding="utf-8"))
    for e in sel["episodes"]:
        truth[e["episode_id"]] = ("aime", e["source_id"])

print("=" * 88)
print("A. regex family vs provider ground truth (dmmath 456, aime 36)")
print("=" * 88)
conf = defaultdict(Counter)
for name, sp in splits.items():
    fam = [classify_family(t) for t in sp.texts]
    for eid, f in zip(sp.episode_ids, fam):
        if eid in truth:
            conf[truth[eid][0]][f] += 1
for t, c in conf.items():
    tot = sum(c.values())
    print(f"  TRUE {t:8s} n={tot:4d} -> regex says {dict(c)}")
    print(f"       recall = {c[t]/tot:.3f}")

# purity the other way: of everything the regex calls dmmath / aime, how many are
# confirmed?  (we only have positive ground truth, so this is an upper bound on
# contamination)
print()
for name, sp in splits.items():
    fam = np.array([classify_family(t) for t in sp.texts])
    for f in ("dmmath", "aime"):
        m = fam == f
        known = np.array([eid in truth and truth[eid][0] == f for eid in sp.episode_ids])
        other = np.array([eid in truth and truth[eid][0] != f for eid in sp.episode_ids])
        print(f"  {name:5s} regex={f:7s} n={m.sum():4d}: confirmed {int((m&known).sum()):4d}, "
              f"confirmed-WRONG {int((m&other).sum()):3d}, unlabelled {int((m&~known&~other).sum()):4d}")

print()
print("=" * 88)
print("B. the fallback bucket 'gsm8k_or_other': what is actually in it?")
print("=" * 88)
import re
RT = re.compile(r"^[A-Z][a-z]+ is [a-z]+\.")          # ruletaker fact style
DM_ISH = re.compile(r"(Work out|Calculate|What is|Which is|Let |Solve|Simplify|Collect|Does \d|divided by|Round )")
for name, sp in splits.items():
    fam = np.array([classify_family(t) for t in sp.texts])
    g = fam == "gsm8k_or_other"
    ln = np.array([len(t) for t in sp.texts])
    rt = np.array([bool(RT.match(t)) for t in sp.texts])
    dmk = np.array([eid in truth and truth[eid][0] == "dmmath" for eid in sp.episode_ids])
    n4 = sp.ngen[:, 0] == 4
    print(f"-- {name}: |gsm8k_or_other| = {g.sum()}")
    print(f"   ruletaker-style ('Name is adj.'): {int((g&rt).sum())}")
    print(f"   confirmed dmmath episodes       : {int((g&dmk).sum())}")
    print(f"   ngen==4 (true gsm8k/aime proto) : {int((g&n4).sum())}")
    rest = g & ~rt & ~dmk & ~n4
    print(f"   remainder                       : {int(rest.sum())} "
          f"(median chars {np.median(ln[rest]) if rest.sum() else float('nan'):.0f})")
    for nm, m in (("ruletaker-style", g & rt), ("confirmed dmmath", g & dmk),
                  ("ngen4", g & n4), ("remainder", rest)):
        if m.sum() == 0:
            continue
        print(f"     {nm:17s} n={m.sum():4d} score {sp.score[m,0].mean():.3f}/"
              f"{sp.score[m,1].mean():.3f}/{sp.score[m,2].mean():.3f}  "
              f"k1/light cost {(sp.cost[m,2]/sp.cost[m,0]).mean():7.1f}")

print()
print("=" * 88)
print("C. how much does the polluted bucket distort the family mean it feeds?")
print("=" * 88)
for name, sp in splits.items():
    fam = np.array([classify_family(t) for t in sp.texts])
    g = fam == "gsm8k_or_other"
    n4 = sp.ngen[:, 0] == 4
    print(f"-- {name}")
    print(f"   family mean as used today (all of gsm8k_or_other): "
          f"{np.round(sp.score[g].mean(0),3)}  logcost k1 {np.log(sp.cost[g,2]).mean():.3f}")
    print(f"   if split: sub-A (ngen4)  n={int((g&n4).sum()):4d} "
          f"{np.round(sp.score[g&n4].mean(0),3)}  logcost k1 {np.log(sp.cost[g&n4,2]).mean():.3f}")
    print(f"             sub-B (ngen2)  n={int((g&~n4).sum()):4d} "
          f"{np.round(sp.score[g&~n4].mean(0),3)}  logcost k1 {np.log(sp.cost[g&~n4,2]).mean():.3f}")

print()
print("=" * 88)
print("D. same check for every regex family: within-family score dispersion")
print("=" * 88)
fam_tr = np.array([classify_family(t) for t in tr.texts])
fam_dv = np.array([classify_family(t) for t in dv.texts])
print(f"{'family':16s} {'n':>5s} {'mean light':>10s} {'sd light':>9s} "
      f"{'mean gain m-l':>13s} {'sd gain':>8s} {'mean log k1cost':>16s} {'sd':>6s}")
for f in sorted(set(fam_tr)):
    m = fam_tr == f
    gain = tr.score[m, 1] - tr.score[m, 0]
    lc = np.log(tr.cost[m, 2])
    print(f"{f:16s} {m.sum():5d} {tr.score[m,0].mean():10.3f} {tr.score[m,0].std():9.3f} "
          f"{gain.mean():13.3f} {gain.std():8.3f} {lc.mean():16.3f} {lc.std():6.3f}")
