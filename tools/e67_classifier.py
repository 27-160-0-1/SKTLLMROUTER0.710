# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""E67 -- a text-only family classifier built from the data analysis, measured against the
current one on the 2,640 public episodes.

The analysis (analysis/SKT_DATA_ANALYSIS.md §3.2) measured `similarity.classify_family` at 90.83 %
against the true source label and located every error in a regex:

    dmmath -> gsm8k_or_other   98   _DMMATH demands a whitelisted first word; "Which", "How",
                                     "Does", a bare expression all fall through
    gsm8k  -> aime             76   _AIME is \$[^$]+\$ -- two dollar amounts match it
    ruletaker -> gsm8k_or_other 40  _RULETAKER wants "If (something|someone|the \w+) ";
                                     "If Erin is red" has a proper noun
    gsm8k  -> ruletaker         8   "If the car ..." matches that same pattern
    babilong -> ruletaker       2   the ruletaker test runs BEFORE the length test

And it supplied a 7-step cascade that reproduces the recorded provenance 2443/2443.  Two of its
steps are unavailable at routing time (aime-selection.json, num_generations), so this is the
text-only projection of it, with the aime/dmmath/gsm8k split -- the part the cascade resolved
with num_generations -- done on text features instead.

Why it matters for the router: the family one-hot is a meta-GBM input.  dmmath is the family
where the router loses most to the oracle (dev gap 0.0291 of 0.103), and 38 of dev's 153 dmmath
items reach the GBM labelled gsm8k_or_other.

Usage: PYTHONPATH=src python tools/e67_classifier.py            (measure only)
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ossp_router import similarity  # noqa: E402
from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import load_input  # noqa: E402

# --- the analysis's structural markers, text-only -------------------------------------------
_CODE = re.compile(r"\bdef \w+\(|assert f\(")
_OPT_ABCD = re.compile(r"^[A-D]\. ", re.M)
_OPT_AB = re.compile(r"^[AB]\. ", re.M)
_OPT_AE = re.compile(r"^[A-E]\. ", re.M)
_HANGUL = re.compile(r"[가-힣]")
_HRMCR = re.compile(r"한국 나이|선/후배|존댓말|음력|양력|윤년|3·1절|생일|띠")
# AIME: LaTeX math, not dollar amounts.  A dollar *amount* is "$12", "$20,000", "$1.5"; LaTeX is
# "$n$", "$\tfrac{m}{n}$", "$17_b$".  Require a non-digit, non-currency character right after $.
# A dollar AMOUNT is "$20,000" / "$1.5" / "$12 fee" -- digits then a word boundary.  LaTeX is
# "$n$", "$\tfrac{m}{n}$", "$12$", "$2 \times 2$" -- the closing $ follows within the same span.
_LATEX = re.compile(r"\$(?![\d,.\s]+\b(?!\$))[^$]{1,80}\$|\\\(|\\\[|\\begin\{|\\frac|\\tfrac|\\sqrt|\\times")
# deepmind-mathematics: one short line, symbolic.  The whitelist is gone; instead we ask whether
# the text *looks like* a math expression or a terse math question.
_DM_SYMBOL = re.compile(r"[=*/^()]|\d+/\d+|\bbase \d+\b|\bprime factors\b|\bderivative\b|\bprob(?:ability)?\b|\bsort\b|\bround\b", re.I)
# deepmind-mathematics idioms a word-problem filter must not veto
_DM_IDIOM = re.compile(
    r"\bpicked without replacement\b|\bprob of\b|\bin (?:ascending|decreasing|increasing|descending) order\b|"
    r"\bhow many \w+ are there in\b|\bhow many minutes are there between\b|^total of\b|^rearrange\b|"
    r"^let [a-z]\b|^suppose\b|\*\*|\bnanometers?\b|\bmillilitres?\b|\bmillennium\b|\bcenturies\b", re.I)
# gsm8k word problems: people, money, narrative verbs
_WORD_PROBLEM = re.compile(r"\b(?:he|she|they|his|her|their|each|every|per|costs?|buys?|sells?|pays?|earns?|spends?|total|how (?:many|much))\b", re.I)
# AIME without LaTeX: competition-geometry / combinatorics phrasing, no money
_AIME_PHRASE = re.compile(
    r"\b(?:regular (?:dodecagon|hexagon|octagon|polygon)|equilateral|convex|quadrants?|diameters?|circumcircle|"
    r"incircle|unit squares?|lattice|remainder when .{1,60} is divided by|find the number of|"
    r"relatively prime positive integers|\bm\+n\b|\bm \+ n\b|\\times|\\tfrac|\\frac)", re.I)


def classify_text(text: str) -> str:
    """Text-only family label (runtime-safe)."""
    if len(text) > 6_000:
        return "longdoc"
    head = text[:600]
    if _CODE.search(text[:400]):
        return "code"
    if len(_OPT_ABCD.findall(text)) == 4:
        return "belebele"
    if len(_OPT_AB.findall(text)) == 2 and len(_OPT_AE.findall(text)) == 2:
        return "truthfulqa"
    if "\nQuestion:" in text:
        return "ruletaker"
    if _HANGUL.search(text):
        return "hrmcr" if _HRMCR.search(head) or len(text) < 1_000 else "belebele"
    # the math trio, resolved on text
    if _LATEX.search(head):
        return "aime"
    words = len(text.split())
    if _DM_IDIOM.search(text) and words <= 60:
        return "dmmath"
    if words <= 40 and "\n" not in text.strip() and _DM_SYMBOL.search(text) and not _WORD_PROBLEM.search(text):
        return "dmmath"
    if words <= 12 and not _WORD_PROBLEM.search(text):
        return "dmmath"
    if _AIME_PHRASE.search(text) and 25 <= words <= 120 and not re.search(r"\$\d", text):
        return "aime"
    return "gsm8k_or_other"


# true-source -> runtime family name, so the two can be compared
SRC_TO_FAM = {
    "cruxeval": "code", "hrmcr": "hrmcr", "ruletaker": "ruletaker", "truthfulqa": "truthfulqa",
    "belebele": "belebele", "babilong": "longdoc", "aime": "aime",
    "deepmind-mathematics": "dmmath", "gsm8k": "gsm8k_or_other",
}


def main() -> int:
    src = {}
    with (ROOT / "analysis/episodes.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            src[row["episode_id"]] = row["source"]
    episodes = []
    for split in ("train", "dev"):
        episodes += list(load_input(ROOT / f"data/materialized/{split}/inputs.json").episodes)

    truth = [SRC_TO_FAM[src[e.episode_id]] for e in episodes]
    old = [similarity.classify_family(episode_text(e)) for e in episodes]
    new = [classify_text(episode_text(e)) for e in episodes]

    for name, pred in (("current classify_family", old), ("E67 text-only", new)):
        ok = sum(p == t for p, t in zip(pred, truth))
        print(f"\n=== {name}: {ok}/{len(truth)} = {ok/len(truth):.4f} ===")
        conf = defaultdict(Counter)
        for p, t in zip(pred, truth):
            if p != t:
                conf[t][p] += 1
        for t in sorted(conf, key=lambda k: -sum(conf[k].values())):
            print(f"  {t:<16} -> " + ", ".join(f"{p} {n}" for p, n in conf[t].most_common()))
        # per-bucket precision: what the GBM sees
        print("  bucket precision:")
        for fam in similarity.FAMILY_NAMES:
            n_pred = sum(p == fam for p in pred)
            n_hit = sum(p == fam and t == fam for p, t in zip(pred, truth))
            if n_pred:
                print(f"    {fam:<16} {n_hit:>4}/{n_pred:<4} = {n_hit/n_pred:.3f}")

    # where they disagree and both are wrong: show a few for the record
    bad = [(e.episode_id, t, o, nn, episode_text(e)[:90].replace("\n", "⏎"))
           for e, t, o, nn in zip(episodes, truth, old, new) if nn != t]
    print(f"\nE67 remaining errors: {len(bad)}")
    for row in bad[:25]:
        print("  ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
