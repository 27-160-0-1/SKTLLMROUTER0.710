# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a06: interpretable prompt-feature table + target table.

Everything here is computable from the prompt text alone, EXCEPT the two
columns explicitly named `oracle_*` (num_generations, true input tokens),
which are kept separately so association tables can be honest about them.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_split, MODEL_IDS  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

_NUM = re.compile(r"\d+(?:\.\d+)?")
_MC = re.compile(r"(?m)^\s*(?:[A-Da-d]|\([A-Da-d]\)|[1-9])[.)]\s+\S")
_ANSFMT = re.compile(
    r"(?i)\b(answer|정답|답)\s*[:：]|\bfinal answer\b|\banswer with\b|"
    r"\boutput only\b|\bonly the\b|\brespond with\b|\[ANSWER\]|boxed")
_CONSTRAINT = re.compile(
    r"(?i)\b(must|should|do not|don't|cannot|only|exactly|at least|at most|"
    r"without|ensure|required|note that|assume)\b")
_LATEX = re.compile(r"\\[a-zA-Z]+|\$")
_CODEISH = re.compile(r"\bdef \w+\(|\breturn\b|\bassert\b|```|\bprint\(")
_URL = re.compile(r"https?://")
_HANGUL = lambda ch: "\uac00" <= ch <= "\ud7a3"

FEATURE_NAMES = [
    "n_chars", "log_chars", "n_words", "n_lines", "n_para", "n_sent",
    "mean_word_len", "ttr", "frac_hangul", "frac_latin", "frac_digit",
    "frac_punct", "frac_upper", "frac_space", "n_numbers", "log_maxnum",
    "n_qmark", "n_colon", "latex_hits", "code_hits", "n_mc_options",
    "n_constraints", "ansfmt", "has_url", "char_entropy", "head_len",
    "tail_len", "n_eq", "n_math_ops", "n_commas", "frac_stop_en",
    "longest_line", "n_caps_words", "n_backslash", "n_paren",
]


def _feat_one(t: str) -> list:
    n = len(t)
    words = t.split()
    lines = t.split("\n")
    n_hangul = sum(1 for ch in t if _HANGUL(ch))
    n_latin = sum(1 for ch in t if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))
    n_digit = sum(1 for ch in t if ch.isdigit())
    n_punct = sum(1 for ch in t if (not ch.isalnum()) and (not ch.isspace()) and ord(ch) < 128)
    n_upper = sum(1 for ch in t if "A" <= ch <= "Z")
    n_space = sum(1 for ch in t if ch.isspace())
    nums = _NUM.findall(t)
    maxnum = max((float(x) for x in nums), default=0.0)
    # char entropy over a coarse alphabet
    counts = {}
    for ch in t[:8000]:
        counts[ch] = counts.get(ch, 0) + 1
    tot = max(1, sum(counts.values()))
    ent = -sum((c / tot) * math.log(c / tot + 1e-12) for c in counts.values())
    stop = ("the ", " of ", " and ", " to ", " is ", " a ", " in ")
    n_stop = sum(t.count(s) for s in stop)
    return [
        float(n), math.log1p(n), float(len(words)), float(len(lines)),
        float(t.count("\n\n") + 1), float(t.count(". ") + t.count("? ") + t.count("! ") + 1),
        (sum(len(w) for w in words) / max(1, len(words))),
        (len(set(w.lower() for w in words)) / max(1, len(words))),
        n_hangul / max(1, n), n_latin / max(1, n), n_digit / max(1, n),
        n_punct / max(1, n), n_upper / max(1, n), n_space / max(1, n),
        float(len(nums)), math.log10(1.0 + maxnum),
        float(t.count("?")), float(t.count(":")),
        float(len(_LATEX.findall(t))), float(len(_CODEISH.findall(t))),
        float(len(_MC.findall(t))), float(len(_CONSTRAINT.findall(t))),
        float(bool(_ANSFMT.search(t))), float(bool(_URL.search(t))),
        ent, float(len(lines[0])), float(len(lines[-1])),
        float(t.count("=")),
        float(sum(t.count(c) for c in "+-*/^")),
        float(t.count(",")), n_stop / max(1.0, len(words) / 100.0),
        float(max(len(x) for x in lines)),
        float(sum(1 for w in words if len(w) > 1 and w.isupper())),
        float(t.count("\\")), float(t.count("(")),
    ]


def build(split_name: str):
    sp = load_split(split_name)
    X = np.asarray([_feat_one(t) for t in sp.texts], dtype=float)
    fam = np.asarray([classify_family(t) for t in sp.texts])
    ngen = sp.ngen[:, 0]
    itok = sp.itok[:, 0] / np.maximum(ngen, 1)      # per generation
    otok = sp.otok / np.maximum(sp.ngen, 1)         # (n,3) per generation
    T = {
        "s_light": sp.score[:, 0], "s_mid": sp.score[:, 1], "s_k1": sp.score[:, 2],
        "g_mid_light": sp.score[:, 1] - sp.score[:, 0],
        "g_k1_mid": sp.score[:, 2] - sp.score[:, 1],
        "g_k1_light": sp.score[:, 2] - sp.score[:, 0],
        "logc_light": np.log(sp.cost[:, 0]), "logc_mid": np.log(sp.cost[:, 1]),
        "logc_k1": np.log(sp.cost[:, 2]),
        "log_otok_light": np.log1p(otok[:, 0]),
        "log_otok_mid": np.log1p(otok[:, 1]),
        "log_otok_k1": np.log1p(otok[:, 2]),
        "log_cratio_k1": np.log(sp.cost[:, 2] / sp.cost[:, 0]),
        "eff_mid": (sp.score[:, 1] - sp.score[:, 0]) / np.maximum(sp.cost[:, 1] - sp.cost[:, 0], 1e-9),
        "eff_k1": (sp.score[:, 2] - sp.score[:, 1]) / np.maximum(sp.cost[:, 2] - sp.cost[:, 1], 1e-9),
    }
    extra = np.column_stack([ngen, np.log(itok)])
    return dict(split=sp, X=X, names=list(FEATURE_NAMES), fam=fam,
                targets=T, extra=extra, extra_names=["oracle_ngen", "oracle_log_itok"],
                itok=itok, otok=otok)


if __name__ == "__main__":
    for nm in ("train", "dev"):
        d = build(nm)
        print(nm, d["X"].shape, "families:",
              {f: int((d["fam"] == f).sum()) for f in sorted(set(d["fam"]))})
