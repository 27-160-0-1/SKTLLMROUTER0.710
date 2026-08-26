# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a08 probe 9: audit + repair of the regex family classifier.

The `gsm8k_or_other` bucket turns out to be a *catch-all* holding real GSM8K
(ngen=4) plus RuleTaker and DeepMind-Mathematics items the regexes missed
(ngen=2).  This script prints the misses, defines a repaired classifier, and
measures the repair with the allocator (family means as the only predictor).
"""
from __future__ import annotations
import re
import sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import (load_all, MODEL_IDS, TIERS, TIER_WEIGHT, tier_result)  # noqa: E402
from ossp_router import similarity  # noqa: E402

tr, dv = load_all()
texts = tr.texts + dv.texts
ntr = len(tr)
fams = np.array([similarity.classify_family(t) for t in texts])
ngen = np.vstack([tr.ngen, dv.ngen])[:, 0]
score = np.vstack([tr.score, dv.score])
cost = np.vstack([tr.cost, dv.cost])

g = np.where((fams == "gsm8k_or_other") & (ngen == 2))[0]
print(f"=== the {len(g)} ngen=2 items inside gsm8k_or_other, by leading pattern ===")
c = Counter()
for i in g:
    t = texts[i]
    if "\nQuestion: " in t:
        c["ruletaker-like (has \\nQuestion:)"] += 1
    else:
        c["dmmath-like: " + " ".join(t.split()[:3])[:34]] += 1
for k, v in c.most_common(30):
    print(f"   {v:4d}  {k}")

print("\n=== conversely: items OUTSIDE gsm8k_or_other with ngen=4 ===")
o = np.where((fams != "gsm8k_or_other") & (ngen == 4))[0]
print("   ", Counter(fams[o]).most_common())

# ------------------------------------------------------------------ repaired
_RT_Q = re.compile(r"\nQuestion: ")
_RT_FACT = re.compile(r"\b\w+ is (?:not )?\w+\.")
_DM_EXTRA = re.compile(
    r"^(?:Work out|Which is|What is|Add|Subtract|Total of|Product of|Divide|Multiply|"
    r"Calculate|Simplify|Solve|Evaluate|Round|Sort|Put|Let |Suppose|Differentiate|"
    r"Factor|Expand|In base|Convert|How many|What comes next|List the prime|"
    r"Is \d|Find |Give |Print |Sum |Take |Subtract|\-?\d[\d ,.eE/*+-]* (?:divided by|times|plus|minus))")
_DM_OP = re.compile(r"^-?[\d./]+ (?:divided by|times|plus|minus) -?[\d./]+\.?$")


def classify2(text: str) -> str:
    head = text[:600]
    if similarity._CODE.search(head):
        return "code"
    if similarity._HRMCR_AGE.search(head) or similarity._HRMCR_CAL.search(head[:200]):
        return "hrmcr"
    if similarity._TRUTHFULQA.match(text):
        return "truthfulqa"
    if _RT_Q.search(text) and len(_RT_FACT.findall(text)) >= 3:
        return "ruletaker"
    if similarity._RULETAKER.search(head) and " is " in head:
        return "ruletaker"
    if sum("가" <= ch <= "힣" for ch in head) > 40:
        return "belebele"
    if len(text) > 6_000:
        return "longdoc"
    if similarity._AIME.search(head) and len(text) < 2_000:
        return "aime"
    body = text.strip()
    if len(body) < 400 and (_DM_EXTRA.match(body) or _DM_OP.match(body)):
        return "dmmath"
    if similarity._DMMATH.match(head) and len(text) < 400:
        return "dmmath"
    return "gsm8k_or_other"


fams2 = np.array([classify2(t) for t in texts])
print("\n=== reassignment table (old -> new) ===")
ch = Counter((a, b) for a, b in zip(fams, fams2) if a != b)
for (a, b), v in ch.most_common():
    print(f"   {a:16s} -> {b:16s} {v:5d}")
print(f"   total moved: {sum(ch.values())} / {len(texts)} "
      f"({100*sum(ch.values())/len(texts):.1f}%)")

print("\n=== gsm8k_or_other purity, before vs after ===")
for tag, f in (("old", fams), ("new", fams2)):
    m = f == "gsm8k_or_other"
    print(f"   {tag}: n={m.sum():4d}  ngen4 frac={np.mean(ngen[m]==4):.3f}  "
          f"light score mean={score[m,0].mean():.3f} sd={score[m,0].std():.3f}")

print("\n=== per-family light-score sd (lower = purer buckets) ===")
print(f"   {'family':16s} {'old n':>6s} {'old sd':>7s} | {'new n':>6s} {'new sd':>7s}")
for f in sorted(set(fams) | set(fams2)):
    a = fams == f
    b = fams2 == f
    print(f"   {f:16s} {a.sum():6d} {score[a,0].std() if a.sum() else float('nan'):7.3f} | "
          f"{b.sum():6d} {score[b,0].std() if b.sum() else float('nan'):7.3f}")

# ------------------------------------------------- allocator value of the labels
print("\n=== allocator test: family means as the ONLY predictor (train-fit -> dev) ===")
dev_slice = np.arange(ntr, len(texts))


def fam_pred(f_all):
    ftr, fdv = f_all[:ntr], f_all[ntr:]
    gs = score[:ntr].mean(0)
    gc = np.log(cost[:ntr]).mean(0)
    ps = np.zeros((len(fdv), 3))
    pc = np.zeros((len(fdv), 3))
    for name in sorted(set(f_all)):
        m = ftr == name
        s = score[:ntr][m].mean(0) if m.sum() >= 8 else gs
        c = np.log(cost[:ntr][m]).mean(0) if m.sum() >= 8 else gc
        k = fdv == name
        ps[k] = s
        pc[k] = np.exp(c)
    return ps, pc


for tag, f_all in (("old regex (9 families)", fams), ("repaired regex", fams2)):
    ps, pc = fam_pred(f_all)
    tot = 0.0
    parts = []
    for t in TIERS:
        best = None
        for s in np.arange(0.60, 1.401, 0.005):
            r = tier_result(ps, pc, dv, t, float(s))
            if r["passed"] and (best is None or r["score"] > best[0]):
                best = (r["score"], float(s))
        tot += TIER_WEIGHT[t] * best[0]
        parts.append(f"{t[:4]}={best[0]:.4f}@{best[1]:.3f}")
    corr = [np.corrcoef(ps[:, j], dv.score[:, j])[0, 1] for j in range(3)]
    lc = [np.std(np.log(pc[:, j]) - np.log(dv.cost[:, j])) for j in range(3)]
    print(f"   {tag:24s} final={tot:.4f}  " + " ".join(parts))
    print(f"   {'':24s} score corr={np.round(corr,3)}  logcost err sd={np.round(lc,3)}")
    # with true cost, safety 1.0 (BRIEF reference row)
    tot2 = 0.0
    for t in TIERS:
        r = tier_result(ps, dv.cost, dv, t, 1.0)
        tot2 += TIER_WEIGHT[t] * r["tier_score"]
    print(f"   {'':24s} with TRUE cost, safety 1.0: {tot2:.4f}")
