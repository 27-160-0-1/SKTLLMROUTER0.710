# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the b01 fast-tier study."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP  # noqa: E402
import protocol as P  # noqa: E402
import bench2 as B  # noqa: E402

FAST = "fast"
MF = MULTS["fast"]


# ------------------------------------------------------------------ latent p
def eb_posterior(lab):
    """E[p | k,n] under a Beta prior moment-matched per (family, model).

    score = k/n with n = num_generations in {2,4}.  Returns (n,3) array.
    """
    fam = lab.fam_arr
    s = lab.true_s
    ngen = lab.ngen
    out = np.zeros_like(s)
    for f in np.unique(fam):
        rows = np.where(fam == f)[0]
        for m in range(3):
            x = s[rows, m]
            n = ngen[rows, m]
            mu = float(x.mean())
            v = float(x.var())
            ei = float(np.mean(1.0 / n))
            if mu <= 1e-9 or mu >= 1 - 1e-9:
                out[rows, m] = np.clip(x, 1e-6, 1 - 1e-6)
                continue
            vp = (v - mu * (1 - mu) * ei) / max(1.0 - ei, 1e-9)
            vp = min(max(vp, 1e-6), mu * (1 - mu) * 0.999)
            conc = mu * (1 - mu) / vp - 1.0
            conc = float(np.clip(conc, 0.05, 500.0))
            a = mu * conc
            b = (1 - mu) * conc
            k = x * n
            out[rows, m] = (a + k) / (a + b + n)
    return out


# --------------------------------------------------------------- fast tier EV
def fast_stats(lab, ps, pc, rows, safety):
    """Deterministic fast-tier outcome of one prediction set on `rows`."""
    pick = P.exact_allocate(ps, pc, MF, safety)
    r = np.arange(len(rows))
    tc = lab.true_c[rows]
    ts = lab.true_s[rows]
    ratio = tc[r, pick].sum() / tc[:, 0].sum()
    sc = ts[r, pick].mean()
    ok = ratio <= MF + 1e-15
    return dict(pick=pick, score=float(sc), ratio=float(ratio), passed=bool(ok),
                final=float(sc if ok else 0.0),
                counts=np.bincount(pick, minlength=3).tolist())


def fast_ev(lab, ps, pc, rows, grid, seeds=(7, 17, 23), nboot=400):
    """Bootstrap EV / bust curve of the fast tier over `grid`."""
    ts = lab.true_s[rows]; tc = lab.true_c[rows]; m = len(rows)
    ev = np.zeros(len(grid)); bu = np.zeros(len(grid)); raw = np.zeros(len(grid))
    for s in seeds:
        smp = np.asarray(lab.samples_for(m, s, nboot, 880))
        e, b, r = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MF, grid)
        ev += e / len(seeds); bu += b / len(seeds); raw += r / len(seeds)
    return ev, bu, raw


GRID = np.arange(0.80, 1.041, 0.005)


def pick_safety(ev, grid=GRID):
    gi = int(np.argmax(ev))
    return float(grid[gi]), gi


# ------------------------------------------------------- repaired classifier
# copied verbatim from experiments/lab/a08_probe13.py (a08's classify_v3)
import re
sys.path.insert(0, str(HERE.parents[1] / "src"))
from ossp_router import similarity  # noqa: E402

_RT_Q = re.compile(r"\nQuestion: ")
_RT_FACT = re.compile(r"\b\w+ is (?:not )?\w+\.")
_DM_EXTRA = re.compile(
    r"^(?:Work out|Which is|What is|Add|Subtract|Total of|Product of|Divide|Multiply|"
    r"Calculate|Simplify|Solve|Evaluate|Round|Sort|Put|Let |Suppose|Differentiate|"
    r"Factor|Expand|In base|Convert|How many|What comes next|List the prime|"
    r"Is \d|Find |Give |Print |Sum |Take |\-?\d[\d ,.eE/*+-]* (?:divided by|times|plus|minus))")
_DM_OP = re.compile(r"^-?[\d./]+ (?:divided by|times|plus|minus) -?[\d./]+\.?$")
_LATEX = re.compile("\\\\(?:frac|sqrt|sum|int|cdot|left|right|text|angle|triangle|overline|log|pi|"
                    r"binom|mathbb|dfrac|le|ge|neq|equiv|pmod)")


def classify_v3(text: str) -> str:
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
    if sum("\uac00" <= ch <= "\ud7a3" for ch in head) > 40:
        return "belebele"
    if len(text) > 6_000:
        return "longdoc"
    if _LATEX.search(text) and len(text) < 2_000:
        return "aime"
    body = text.strip()
    if len(body) < 400 and (_DM_EXTRA.match(body) or _DM_OP.match(body)):
        return "dmmath"
    if similarity._DMMATH.match(head) and len(text) < 400:
        return "dmmath"
    return "gsm8k_or_other"


FOLDS, SEED = 10, 123


def fam_matrix(lab, part, cv, arr):
    """Honest per-row family-mean target vectors (6 cols) for cv rows and arr rows."""
    tg = lab.targets; tr = lab.train_idx
    fold_of = np.random.default_rng(SEED).integers(0, FOLDS, size=len(tr))
    cvm = np.zeros((len(cv["idx"]), 6))
    pos = {int(i): k for k, i in enumerate(cv["idx"])}
    for f in range(FOLDS):
        fit = tr[fold_of != f]; hold = tr[fold_of == f]
        gm = tg[fit].mean(axis=0); means = {}
        for name in np.unique(part[fit]):
            sel = fit[part[fit] == name]
            means[name] = tg[sel].mean(axis=0) if len(sel) >= 8 else gm
        for i in hold:
            cvm[pos[int(i)]] = means.get(part[i], gm)
    gm = tg[tr].mean(axis=0); means = {}
    for name in np.unique(part[tr]):
        sel = tr[part[tr] == name]
        means[name] = tg[sel].mean(axis=0) if len(sel) >= 8 else gm
    dvm = np.array([means.get(part[i], gm) for i in arr["idx"]])
    return cvm, dvm
