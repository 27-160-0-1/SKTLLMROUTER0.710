# SPDX-License-Identifier: Apache-2.0
"""a06 step 9: robustness of the sub-family discriminators + leakage check.

Which prompt-only predicate best separates true-AIME from the GSM8K money
problems that the `\\$...\\$` regex sweeps into the `aime` bucket, and are
there true-AIME items hiding in other buckets?
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a06_features import build  # noqa: E402

BS = re.compile(r"\\[a-zA-Z]{2,}")           # a real LaTeX command
MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")   # "$20,000"

PREDS = {
    "no '?'": lambda t: "?" not in t,
    "no '?' or 2+ '?'": lambda t: t.count("?") != 1,
    "LaTeX cmd present": lambda t: bool(BS.search(t)),
    "no money-$ token": lambda t: not bool(MONEY.search(t)),
    "no '?' AND no money-$": lambda t: ("?" not in t) and not bool(MONEY.search(t)),
    "LaTeX cmd OR no '?'": lambda t: bool(BS.search(t)) or ("?" not in t),
}

for nm in ("train", "dev"):
    d = build(nm)
    sp, fam = d["split"], d["fam"]
    m = fam == "aime"
    txt = [sp.texts[i] for i in np.where(m)[0]]
    S, C = sp.score[m], sp.cost[m]
    lb = sp.cost[:, 0].sum()
    print(f"\n===== {nm}: aime bucket n={m.sum()} =====")
    print(f"{'predicate':26s} {'n_hard':>6s} | {'hard s(l/m/k)':>22s} "
          f"{'hard k1 cost(lb)':>16s} | {'easy s(l/m/k)':>22s}")
    for name, fn in PREDS.items():
        sel = np.array([fn(t) for t in txt])
        if sel.sum() == 0 or sel.sum() == len(sel):
            print(f"{name:26s} degenerate ({sel.sum()}/{len(sel)})")
            continue
        a, b = S[sel].mean(0), S[~sel].mean(0)
        print(f"{name:26s} {sel.sum():6d} | {a[0]:6.3f}{a[1]:8.3f}{a[2]:8.3f} "
              f"{C[sel,2].sum()/lb:16.2f} | {b[0]:6.3f}{b[1]:8.3f}{b[2]:8.3f}")

    print(" leakage check: items OUTSIDE the aime bucket that look like true AIME")
    other = ~m
    idx = np.where(other)[0]
    hard = np.array([("?" not in sp.texts[i]) and bool(BS.search(sp.texts[i]))
                     and len(sp.texts[i]) < 2000 for i in idx])
    if hard.sum():
        ii = idx[hard]
        print(f"   n={hard.sum()}  families={dict(zip(*np.unique(fam[ii], return_counts=True)))}"
              f"  s={np.round(sp.score[ii].mean(0),3)}  k1cost={sp.cost[ii,2].sum()/lb:.2f} lb")
        for i in ii[:3]:
            print("     ", sp.texts[i][:110].replace("\n", " | "))
    else:
        print("   none")

    print(" gsm8k_or_other bucket, split by frac_digit>=.08 vs alternatives")
    g = fam == "gsm8k_or_other"
    gt = [sp.texts[i] for i in np.where(g)[0]]
    Sg, Cg = sp.score[g], sp.cost[g]
    ALT = {
        "frac_digit>=.08": lambda t, i: d["X"][np.where(g)[0][i], d["names"].index("frac_digit")] >= 0.08,
        "no '?'": lambda t, i: "?" not in t,
        "(a) (b) options present": lambda t, i: ("(a)" in t and "(b)" in t),
        "starts with a DM-math verb": lambda t, i: bool(re.match(
            r"^(Solve|Calculate|Simplify|Work out|Let |Suppose|What is|Which is|Sort|Put|Round|"
            r"Divide|Multiply|Add|Subtract|Evaluate|Differentiate|Factor|Expand|Find|List|Is )", t)),
    }
    for name, fn in ALT.items():
        sel = np.array([fn(t, i) for i, t in enumerate(gt)])
        if sel.sum() in (0, len(sel)):
            print(f"   {name:28s} degenerate")
            continue
        a, b = Sg[sel].mean(0), Sg[~sel].mean(0)
        print(f"   {name:28s} n={sel.sum():4d} hard s={np.round(a,3)} | "
              f"easy s={np.round(b,3)}  ngen(hard)={sp.ngen[g][sel,0].mean():.1f}")
