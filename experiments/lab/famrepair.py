# SPDX-License-Identifier: Apache-2.0
"""Repaired source-family classifier (a08).

The deployed ``similarity.classify_family`` mislabels 7.0% of items:

* ``_AIME`` is a bare "two dollar signs" pattern, so GSM8K money word problems
  land in ``aime`` (88 items; light score .756 vs real AIME .103, and ~9x
  cheaper on k1);
* ``_RULETAKER`` requires "If something|someone|the <word> ", so RuleTaker items
  whose rules start with a named entity fall into the ``gsm8k_or_other``
  catch-all (38 items);
* ``_DMMATH`` has no "Work out | Which is | In base | Rearrange | ..." so
  DeepMind-Mathematics items fall into the same catch-all (54 items).

Same nine bucket names, so nothing downstream changes shape.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router import similarity  # noqa: E402

_RT_Q = re.compile("\nQuestion: ")
_RT_FACT = re.compile(r"\b\w+ is (?:not )?\w+\.")
_LATEX = re.compile(
    "\\\\(?:frac|sqrt|sum|int|cdot|left|right|text|angle|triangle|overline|log|pi|"
    "binom|mathbb|dfrac|le|ge|neq|equiv|pmod)")
_DM_OP = re.compile(r"^-?[\d./]+ (?:divided by|times|plus|minus) -?[\d./]+\.?$")
_NARROW = re.compile(
    r"^(?:Work out|Which is|Total of|Product of|Divide|Multiply|Calculate|Simplify|"
    r"Rearrange|Collect the terms|Round|Sort|In base|What comes next|"
    r"List the prime|Is \d|Express |Determine [a-z] so|Are -?\d)")


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
    if sum("가" <= ch <= "힣" for ch in head) > 40:
        return "belebele"
    if len(text) > 6_000:
        return "longdoc"
    if _LATEX.search(text) and len(text) < 2_000:
        return "aime"
    body = text.strip()
    if len(body) < 400 and (_NARROW.match(body) or _DM_OP.match(body)):
        return "dmmath"
    if similarity._DMMATH.match(head) and len(text) < 400:
        return "dmmath"
    return "gsm8k_or_other"
