# SPDX-License-Identifier: Apache-2.0
"""a08 probe 10: is the `aime` bucket really AIME, or GSM8K money problems?

similarity._AIME = r"\$[^$]+\$" matches any two '$' characters, so a word
problem mentioning $20,000 ... $1000 is labelled `aime`.
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

tr, dv = load_all()
texts = tr.texts + dv.texts
ntr = len(tr)
fams = np.array([similarity.classify_family(t) for t in texts])
score = np.vstack([tr.score, dv.score])
cost = np.vstack([tr.cost, dv.cost])
ngen = np.vstack([tr.ngen, dv.ngen])[:, 0]

a = np.where(fams == "aime")[0]
LATEX = re.compile(r"\\(?:frac|sqrt|sum|int|cdot|left|right|text|angle|triangle|overline|log|pi|"
                   r"binom|mathbb|dfrac|le|ge|neq|equiv|pmod)")
MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
has_latex = np.array([bool(LATEX.search(texts[i])) for i in a])
money_only = np.array([bool(MONEY.search(texts[i])) and not LATEX.search(texts[i]) for i in a])
print(f"=== `aime` bucket n={len(a)} ===")
print(f"  with LaTeX commands      : {has_latex.sum():4d}  light={score[a[has_latex],0].mean():.3f} "
      f"mid={score[a[has_latex],1].mean():.3f} k1={score[a[has_latex],2].mean():.3f} "
      f"logcost_k1={np.log(cost[a[has_latex],2]).mean():.2f}")
print(f"  $money, no LaTeX         : {money_only.sum():4d}  light={score[a[money_only],0].mean():.3f} "
      f"mid={score[a[money_only],1].mean():.3f} k1={score[a[money_only],2].mean():.3f} "
      f"logcost_k1={np.log(cost[a[money_only],2]).mean():.2f}")
rest = ~has_latex & ~money_only
print(f"  neither                  : {rest.sum():4d}  light={score[a[rest],0].mean() if rest.sum() else float('nan'):.3f}")
print(f"  ngen: latex {np.unique(ngen[a[has_latex]], return_counts=True)}  "
      f"money {np.unique(ngen[a[money_only]], return_counts=True)}")
print("\n  --- samples with LaTeX ---")
for i in a[has_latex][:3]:
    print(f"   {texts[i][:180]!r}")
print("  --- samples $money only ---")
for i in a[money_only][:3]:
    print(f"   {texts[i][:180]!r}")

print("\n=== same test applied to the WHOLE corpus: LaTeX-bearing items ===")
lat_all = np.array([bool(LATEX.search(t)) for t in texts])
print(f"  total with LaTeX: {lat_all.sum()}  by current family:")
for f in sorted(set(fams)):
    m = (fams == f)
    if (m & lat_all).sum():
        print(f"    {f:16s} {(m&lat_all).sum():4d}/{m.sum():4d}  "
              f"light score latex={score[m&lat_all,0].mean():.3f} "
              f"vs non-latex={score[m&~lat_all,0].mean() if (m&~lat_all).sum() else float('nan'):.3f}")

print("\n=== value of splitting aime into aime_latex / gsm_money (family means, train->dev) ===")
from labdata import TIERS, TIER_WEIGHT, tier_result  # noqa: E402


def eval_labels(lab):
    lab = np.asarray(lab)
    gs = score[:ntr].mean(0)
    gc = np.log(cost[:ntr]).mean(0)
    ps = np.zeros((len(texts) - ntr, 3)); pc = np.zeros_like(ps)
    for name in sorted(set(lab)):
        m = lab[:ntr] == name
        s = score[:ntr][m].mean(0) if m.sum() >= 8 else gs
        c = np.log(cost[:ntr][m]).mean(0) if m.sum() >= 8 else gc
        k = lab[ntr:] == name
        ps[k] = s; pc[k] = np.exp(c)
    tot = 0.0
    for t in TIERS:
        r = tier_result(ps, dv.cost, dv, t, 1.0)
        tot += TIER_WEIGHT[t] * r["tier_score"]
    corr = [np.corrcoef(ps[:, j], dv.score[:, j])[0, 1] for j in range(3)]
    return tot, corr


base = fams.copy()
t0, c0 = eval_labels(base)
split = base.copy()
for k, i in enumerate(a):
    split[i] = "aime" if has_latex[k] else "gsm_money"
t1, c1 = eval_labels(split)
print(f"  9 families            : true-cost final={t0:.4f} corr={np.round(c0,3)}")
print(f"  + aime/$money split   : true-cost final={t1:.4f} corr={np.round(c1,3)}")

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


rep = np.array([classify2(t) for t in texts])
t2, c2 = eval_labels(rep)
rep2 = rep.copy()
for i in np.where(rep == "aime")[0]:
    if not LATEX.search(texts[i]):
        rep2[i] = "gsm_money"
t3, c3 = eval_labels(rep2)
print(f"  repaired (rt+dm)      : true-cost final={t2:.4f} corr={np.round(c2,3)}")
print(f"  repaired + aime split : true-cost final={t3:.4f} corr={np.round(c3,3)}")
