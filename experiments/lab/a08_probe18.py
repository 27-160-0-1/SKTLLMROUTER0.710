# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a08 probe 18: tighten the dmmath rescue rule.  "How many"/"Find " are also
GSM8K/AIME openers, so measure purity and value for several verb lists.
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
from labdata import load_all, TIERS, TIER_WEIGHT, tier_result  # noqa: E402
from ossp_router import similarity  # noqa: E402

tr, dv = load_all()
texts = tr.texts + dv.texts
ntr = len(tr)
score = np.vstack([tr.score, dv.score])
cost = np.vstack([tr.cost, dv.cost])
ngen = np.vstack([tr.ngen, dv.ngen])[:, 0]
old = np.array([similarity.classify_family(t) for t in texts])

_RT_Q = re.compile(r"\nQuestion: ")
_RT_FACT = re.compile(r"\b\w+ is (?:not )?\w+\.")
_LATEX = re.compile(r"\\(?:frac|sqrt|sum|int|cdot|left|right|text|angle|triangle|overline|log|pi|"
                    r"binom|mathbb|dfrac|le|ge|neq|equiv|pmod)")
_DM_OP = re.compile(r"^-?[\d./]+ (?:divided by|times|plus|minus) -?[\d./]+\.?$")

WIDE = (r"Work out|Which is|What is|Add|Subtract|Total of|Product of|Divide|Multiply|"
        r"Calculate|Simplify|Solve|Evaluate|Round|Sort|Put|Let |Suppose|Differentiate|"
        r"Factor|Expand|In base|Convert|How many|What comes next|List the prime|"
        r"Is \d|Find |Give |Print |Sum |Take ")
NARROW = (r"Work out|Which is|Total of|Product of|Divide|Multiply|Calculate|Simplify|"
          r"Rearrange|Collect the terms|Round|Sort|In base|What comes next|"
          r"List the prime|Is \d|Express |Determine [a-z] so|Are -?\d")
MID = NARROW + r"|How many (?:minutes|hours|seconds|days|weeks|months|years|centuries|"    \
      r"millimet|centimet|kilomet|met[re]|litres|millilitres|grams|kilograms|nanosecond|"  \
      r"microsecond|millisecond|decades|millennia)"

RULES = {"wide (probe13)": WIDE, "mid": MID, "narrow": NARROW}


def build(pattern):
    dm = re.compile("^(?:" + pattern + ")")
    lab = old.copy()
    for i, t in enumerate(texts):
        if lab[i] == "aime":
            lab[i] = "aime" if (_LATEX.search(t) and len(t) < 2000) else "gsm8k_or_other"
    for i, t in enumerate(texts):
        if lab[i] != "gsm8k_or_other":
            continue
        if _RT_Q.search(t) and len(_RT_FACT.findall(t)) >= 3:
            lab[i] = "ruletaker"
            continue
        b = t.strip()
        if len(b) < 400 and (dm.match(b) or _DM_OP.match(b)):
            lab[i] = "dmmath"
    return lab


def eval_labels(lab):
    gs, gc = score[:ntr].mean(0), np.log(cost[:ntr]).mean(0)
    ps = np.zeros((len(texts) - ntr, 3))
    pc = np.zeros_like(ps)
    for name in sorted(set(lab)):
        m = lab[:ntr] == name
        s = score[:ntr][m].mean(0) if m.sum() >= 8 else gs
        cc = np.log(cost[:ntr][m]).mean(0) if m.sum() >= 8 else gc
        k = lab[ntr:] == name
        ps[k], pc[k] = s, np.exp(cc)
    tot = sum(TIER_WEIGHT[t] * tier_result(ps, dv.cost, dv, t, 1.0)["tier_score"] for t in TIERS)
    corr = [np.corrcoef(ps[:, j], dv.score[:, j])[0, 1] for j in range(3)]
    return tot, corr


print(f"{'rule':16s} {'->dm':>5s} {'->rt':>5s} {'aime->g':>8s} {'dm ngen2%':>10s} "
      f"{'true-cost final':>16s}  corr(l/m/k)")
t0, c0 = eval_labels(old)
print(f"{'deployed':16s} {0:5d} {0:5d} {0:8d} {'':>10s} {t0:16.4f}  {np.round(c0,3)}")
for name, pat in RULES.items():
    lab = build(pat)
    mv = old != lab
    ndm = int(((lab == "dmmath") & mv).sum())
    nrt = int(((lab == "ruletaker") & mv).sum())
    nag = int(((old == "aime") & (lab == "gsm8k_or_other")).sum())
    pure = 100 * np.mean(ngen[(lab == "dmmath") & mv] == 2) if ndm else float("nan")
    t1, c1 = eval_labels(lab)
    print(f"{name:16s} {ndm:5d} {nrt:5d} {nag:8d} {pure:10.1f} {t1:16.4f}  {np.round(c1,3)}")
