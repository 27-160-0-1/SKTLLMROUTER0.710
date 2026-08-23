# SPDX-License-Identifier: Apache-2.0
"""a08 probe 13: final repaired classifier (9 buckets, no new family), split
statistics train vs dev, and the family-mean allocator value of each repair step.
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
from labdata import load_all, TIERS, TIER_WEIGHT, tier_result  # noqa: E402
from ossp_router import similarity  # noqa: E402

tr, dv = load_all()
texts = tr.texts + dv.texts
ntr = len(tr)
score = np.vstack([tr.score, dv.score])
cost = np.vstack([tr.cost, dv.cost])
ngen = np.vstack([tr.ngen, dv.ngen])[:, 0]

_RT_Q = re.compile(r"\nQuestion: ")
_RT_FACT = re.compile(r"\b\w+ is (?:not )?\w+\.")
_DM_EXTRA = re.compile(
    r"^(?:Work out|Which is|What is|Add|Subtract|Total of|Product of|Divide|Multiply|"
    r"Calculate|Simplify|Solve|Evaluate|Round|Sort|Put|Let |Suppose|Differentiate|"
    r"Factor|Expand|In base|Convert|How many|What comes next|List the prime|"
    r"Is \d|Find |Give |Print |Sum |Take |\-?\d[\d ,.eE/*+-]* (?:divided by|times|plus|minus))")
_DM_OP = re.compile(r"^-?[\d./]+ (?:divided by|times|plus|minus) -?[\d./]+\.?$")
_LATEX = re.compile(r"\\(?:frac|sqrt|sum|int|cdot|left|right|text|angle|triangle|overline|log|pi|"
                    r"binom|mathbb|dfrac|le|ge|neq|equiv|pmod)")


def classify_v3(text: str) -> str:
    """9 buckets, same names as the deployed classifier."""
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
    if _LATEX.search(text) and len(text) < 2_000:      # real AIME, not $money
        return "aime"
    body = text.strip()
    if len(body) < 400 and (_DM_EXTRA.match(body) or _DM_OP.match(body)):
        return "dmmath"
    if similarity._DMMATH.match(head) and len(text) < 400:
        return "dmmath"
    return "gsm8k_or_other"


old = np.array([similarity.classify_family(t) for t in texts])
new = np.array([classify_v3(t) for t in texts])

print("=== reassignment (old -> new) ===")
c = Counter((a, b) for a, b in zip(old, new) if a != b)
for (a, b), v in c.most_common():
    ktr = sum(1 for i in range(ntr) if old[i] == a and new[i] == b)
    print(f"  {a:16s} -> {b:16s} total={v:4d}  (train {ktr}, dev {v-ktr})")
print(f"  moved {sum(c.values())}/{len(texts)} = {100*sum(c.values())/len(texts):.1f}%")

print("\n=== bucket profile old vs new (light/mid/k1 mean score; k1 cost ratio) ===")
print(f"  {'family':16s} | {'old n':>5s} {'old scores':>20s} {'ck1':>6s} | "
      f"{'new n':>5s} {'new scores':>20s} {'ck1':>6s}")
for f in sorted(set(old) | set(new)):
    a, b = old == f, new == f
    sa = np.round(score[a].mean(0), 3) if a.sum() else np.array([np.nan] * 3)
    sb = np.round(score[b].mean(0), 3) if b.sum() else np.array([np.nan] * 3)
    ca = cost[a, 2].mean() / cost[a, 0].mean() if a.sum() else np.nan
    cb = cost[b, 2].mean() / cost[b, 0].mean() if b.sum() else np.nan
    print(f"  {f:16s} | {a.sum():5d} {str(sa):>20s} {ca:6.1f} | "
          f"{b.sum():5d} {str(sb):>20s} {cb:6.1f}")


def eval_labels(lab):
    lab = np.asarray(lab)
    gs, gc = score[:ntr].mean(0), np.log(cost[:ntr]).mean(0)
    ps = np.zeros((len(texts) - ntr, 3)); pc = np.zeros_like(ps)
    for name in sorted(set(lab)):
        m = lab[:ntr] == name
        s = score[:ntr][m].mean(0) if m.sum() >= 8 else gs
        cc = np.log(cost[:ntr][m]).mean(0) if m.sum() >= 8 else gc
        k = lab[ntr:] == name
        ps[k], pc[k] = s, np.exp(cc)
    tot = sum(TIER_WEIGHT[t] * tier_result(ps, dv.cost, dv, t, 1.0)["tier_score"] for t in TIERS)
    tot2 = 0.0
    for t in TIERS:
        best = max((tier_result(ps, pc, dv, t, float(s))["score"]
                    for s in np.arange(0.6, 1.401, 0.01)
                    if tier_result(ps, pc, dv, t, float(s))["passed"]), default=0.0)
        tot2 += TIER_WEIGHT[t] * best
    corr = [np.corrcoef(ps[:, j], dv.score[:, j])[0, 1] for j in range(3)]
    lce = [np.std(np.log(pc[:, j]) - np.log(dv.cost[:, j])) for j in range(3)]
    return tot, tot2, corr, lce


print("\n=== family means as the only predictor, train-fit -> dev ===")
print(f"  {'labels':34s} {'true-cost':>10s} {'pred-cost':>10s}  corr(l/m/k)      logcost sd")
steps = {}
steps["deployed regex"] = old
rt_only = old.copy()
for i in range(len(texts)):
    if old[i] == "gsm8k_or_other" and _RT_Q.search(texts[i]) and len(_RT_FACT.findall(texts[i])) >= 3:
        rt_only[i] = "ruletaker"
steps["+ ruletaker rescue"] = rt_only
dm = rt_only.copy()
for i in range(len(texts)):
    if dm[i] == "gsm8k_or_other":
        b = texts[i].strip()
        if len(b) < 400 and (_DM_EXTRA.match(b) or _DM_OP.match(b)):
            dm[i] = "dmmath"
steps["+ dmmath rescue"] = dm
steps["+ aime/$money fix (= full v3)"] = new
for name, lab in steps.items():
    t1, t2, corr, lce = eval_labels(lab)
    print(f"  {name:34s} {t1:10.4f} {t2:10.4f}  {np.round(corr,3)}  {np.round(lce,3)}")
