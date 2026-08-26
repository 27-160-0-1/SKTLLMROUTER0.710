# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a08 probe 17: robustness of the repaired family classifier.

Does each rescued item actually behave like its destination bucket?  Reported
separately for train (where the regexes were written) and dev (held out from
that inspection in the sense that the same rule is applied unchanged).
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_all  # noqa: E402
from ossp_router import similarity  # noqa: E402
sys.path.insert(0, str(HERE))
from a08_probe13 import classify_v3, _DM_EXTRA, _DM_OP, _RT_Q, _RT_FACT, _LATEX  # noqa: E402

tr, dv = load_all()
texts = tr.texts + dv.texts
ntr = len(tr)
score = np.vstack([tr.score, dv.score])
cost = np.vstack([tr.cost, dv.cost])
ngen = np.vstack([tr.ngen, dv.ngen])[:, 0]
old = np.array([similarity.classify_family(t) for t in texts])
new = np.array([classify_v3(t) for t in texts])
moved = old != new

print("=== rescued items: do they look like their destination? ===")
print(f"  {'move':38s} {'split':>6s} {'n':>4s} {'ngen2%':>7s} "
      f"{'light':>6s} {'mid':>6s} {'k1':>6s}  vs destination bucket")
for dest in ("ruletaker", "dmmath", "gsm8k_or_other"):
    for split, sl in (("train", slice(0, ntr)), ("dev", slice(ntr, None))):
        m = np.zeros(len(texts), bool)
        m[sl] = True
        r = m & moved & (new == dest)
        if r.sum() == 0:
            continue
        base = m & ~moved & (new == dest)
        print(f"  {'-> ' + dest:38s} {split:>6s} {r.sum():4d} "
              f"{100*np.mean(ngen[r]==2):7.1f} "
              f"{score[r,0].mean():6.3f} {score[r,1].mean():6.3f} {score[r,2].mean():6.3f}"
              f"   [{score[base,0].mean():.3f} {score[base,1].mean():.3f} "
              f"{score[base,2].mean():.3f}]")

print("\n=== which rule fires for each dmmath rescue? ===")
resc = np.where(moved & (new == "dmmath"))[0]
by_op = sum(1 for i in resc if _DM_OP.match(texts[i].strip()))
by_extra = sum(1 for i in resc if _DM_EXTRA.match(texts[i].strip()) and not _DM_OP.match(texts[i].strip()))
print(f"  bare-arithmetic pattern _DM_OP : {by_op}")
print(f"  extended verb list _DM_EXTRA   : {by_extra}")
lead = {}
for i in resc:
    k = " ".join(texts[i].split()[:2])
    lead[k] = lead.get(k, 0) + 1
print("  leading bigrams:", sorted(lead.items(), key=lambda kv: -kv[1])[:12])

print("\n=== false-positive risk: rules applied to the CORRECTLY labelled buckets ===")
for f in ("gsm8k_or_other", "aime"):
    keep = np.where((old == f) & (new == f))[0]
    print(f"  kept in {f}: {len(keep)}  ngen4%={100*np.mean(ngen[keep]==4):.1f}  "
          f"scores={np.round(score[keep].mean(0),3)}")
print(f"  items where _RT_Q+3facts fires but old label was NOT gsm8k_or_other: "
      f"{sum(1 for i in range(len(texts)) if old[i] not in ('gsm8k_or_other','ruletaker') and _RT_Q.search(texts[i]) and len(_RT_FACT.findall(texts[i]))>=3)}")
print(f"  items with LaTeX outside aime: "
      f"{sum(1 for i in range(len(texts)) if _LATEX.search(texts[i]) and old[i]!='aime')}")
