# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python hashed character tf-idf and public-train kNN prediction.

CHALLENGE_RULES.md allows classifiers, vocabularies/IDF tables, and search
indexes built from public data inside the submission image.  This module
matches prompts against the public Train prompts by hashed character-n-gram
cosine similarity; the training tool and the runtime share this exact
implementation, so offline validation reflects container behavior.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

HASH_BITS = 15
HASH_BINS = 1 << HASH_BITS
NGRAM_SIZES = (3, 4, 5)
TEXT_LIMIT = 4_000
TOP_COMPONENTS = 256
NEIGHBORS = 16  # E20: k=16이 k=8 대비 CV EV +0.0014 (단봉 곡선 확인)

_FNV_OFFSET = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
_MASK64 = (1 << 64) - 1
_SPACE = re.compile(r"\s+")

_GRAM_BINS: Dict[str, int] = {}
_GRAM_CACHE_LIMIT = 2_000_000

# Content-only source-family classifier (E67; see tools/e67_classifier.py for the measurement).
_F_CODE = re.compile(r"\bdef \w+\(|assert f\(")
_F_OPT_ABCD = re.compile(r"^[A-D]\. ", re.M)
_F_OPT_AB = re.compile(r"^[AB]\. ", re.M)
_F_OPT_AE = re.compile(r"^[A-E]\. ", re.M)
_F_HANGUL = re.compile(r"[가-힣]")
_F_HRMCR = re.compile(r"한국 나이|선/후배|존댓말|음력|양력|윤년|3·1절|생일|띠")
# AIME: LaTeX math, not dollar amounts.  A dollar *amount* is "$12", "$20,000", "$1.5"; LaTeX is
# "$n$", "$\tfrac{m}{n}$", "$17_b$".  Require a non-digit, non-currency character right after $.
# A dollar AMOUNT is "$20,000" / "$1.5" / "$12 fee" -- digits then a word boundary.  LaTeX is
# "$n$", "$\tfrac{m}{n}$", "$12$", "$2 \times 2$" -- the closing $ follows within the same span.
_F_LATEX = re.compile(r"\$(?![\d,.\s]+\b(?!\$))[^$]{1,80}\$|\\\(|\\\[|\\begin\{|\\frac|\\tfrac|\\sqrt|\\times")
# deepmind-mathematics: one short line, symbolic.  The whitelist is gone; instead we ask whether
# the text *looks like* a math expression or a terse math question.
_F_DM_SYMBOL = re.compile(r"[=*/^()]|\d+/\d+|\bbase \d+\b|\bprime factors\b|\bderivative\b|\bprob(?:ability)?\b|\bsort\b|\bround\b", re.I)
# deepmind-mathematics idioms a word-problem filter must not veto
_F_DM_IDIOM = re.compile(
    r"\bpicked without replacement\b|\bprob of\b|\bin (?:ascending|decreasing|increasing|descending) order\b|"
    r"\bhow many \w+ are there in\b|\bhow many minutes are there between\b|^total of\b|^rearrange\b|"
    r"^let [a-z]\b|^suppose\b|\*\*|\bnanometers?\b|\bmillilitres?\b|\bmillennium\b|\bcenturies\b", re.I)
# gsm8k word problems: people, money, narrative verbs
_F_WORD_PROBLEM = re.compile(r"\b(?:he|she|they|his|her|their|each|every|per|costs?|buys?|sells?|pays?|earns?|spends?|total|how (?:many|much))\b", re.I)
# AIME without LaTeX: competition-geometry / combinatorics phrasing, no money
_F_AIME_PHRASE = re.compile(
    r"\b(?:regular (?:dodecagon|hexagon|octagon|polygon)|equilateral|convex|quadrants?|diameters?|circumcircle|"
    r"incircle|unit squares?|lattice|remainder when .{1,60} is divided by|find the number of|"
    r"relatively prime positive integers|\bm\+n\b|\bm \+ n\b|\\times|\\tfrac|\\frac)", re.I)


def classify_family(text: str) -> str:
    """Assign one of the public source-family labels from prompt content only.

    E67: rebuilt from the data analysis's structural markers.  99.85 % against the true
    source on the public 2,640 (the previous heuristic: 91.44 %, with an `aime` bucket that
    was 69 % GSM8K).  Order matters: length first, then unambiguous structure, then the
    math trio resolved on text.
    """
    if len(text) > 6_000:
        return "longdoc"
    head = text[:600]
    if _F_CODE.search(text[:400]):
        return "code"
    if len(_F_OPT_ABCD.findall(text)) == 4:
        return "belebele"
    if len(_F_OPT_AB.findall(text)) == 2 and len(_F_OPT_AE.findall(text)) == 2:
        return "truthfulqa"
    if "\nQuestion:" in text:
        return "ruletaker"
    if _F_HANGUL.search(text):
        return "hrmcr" if _F_HRMCR.search(head) or len(text) < 1_000 else "belebele"
    # the math trio, resolved on text
    if _F_LATEX.search(head):
        return "aime"
    words = len(text.split())
    if _F_DM_IDIOM.search(text) and words <= 60:
        return "dmmath"
    if words <= 40 and "\n" not in text.strip() and _F_DM_SYMBOL.search(text) and not _F_WORD_PROBLEM.search(text):
        return "dmmath"
    if words <= 12 and not _F_WORD_PROBLEM.search(text):
        return "dmmath"
    if _F_AIME_PHRASE.search(text) and 25 <= words <= 120 and not re.search(r"\$\d", text):
        return "aime"
    return "gsm8k_or_other"


FAMILY_NAMES = (
    "code", "hrmcr", "ruletaker", "truthfulqa", "belebele",
    "longdoc", "aime", "dmmath", "gsm8k_or_other",
)


def _gram_bin(gram: str) -> int:
    cached = _GRAM_BINS.get(gram)
    if cached is not None:
        return cached
    digest = _FNV_OFFSET
    for byte in gram.encode("utf-8"):
        digest ^= byte
        digest = (digest * _FNV_PRIME) & _MASK64
    result = digest & (HASH_BINS - 1)
    if len(_GRAM_BINS) < _GRAM_CACHE_LIMIT:
        _GRAM_BINS[gram] = result
    return result


def hashed_counts(text: str) -> Dict[int, int]:
    """Hashed character n-gram counts of the normalized prompt head."""

    normalized = _SPACE.sub(" ", text[:TEXT_LIMIT].casefold()).strip()
    padded = f" {normalized} "
    counts: Dict[int, int] = {}
    length = len(padded)
    for size in NGRAM_SIZES:
        for start in range(0, max(0, length - size + 1)):
            index = _gram_bin(padded[start:start + size])
            counts[index] = counts.get(index, 0) + 1
    return counts


def document_frequencies(texts: Iterable[str]) -> Tuple[Dict[int, int], int]:
    frequencies: Dict[int, int] = {}
    total = 0
    for text in texts:
        total += 1
        for index in hashed_counts(text):
            frequencies[index] = frequencies.get(index, 0) + 1
    return frequencies, total


def idf_table(frequencies: Mapping[int, int], total_documents: int) -> Dict[int, float]:
    return {
        index: math.log((1 + total_documents) / (1 + df)) + 1.0
        for index, df in frequencies.items()
    }


def tfidf_vector(
    text: str,
    idf: Mapping[int, float],
    *,
    top_components: int = 0,
) -> Dict[int, float]:
    """L2-normalized sublinear tf-idf vector; optionally keep top components."""

    weights = {}
    for index, count in hashed_counts(text).items():
        factor = idf.get(index)
        if factor is None:
            continue
        weights[index] = (1.0 + math.log(count)) * factor
    if top_components and len(weights) > top_components:
        kept = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
        weights = dict(kept[:top_components])
    norm = math.sqrt(math.fsum(value * value for value in weights.values()))
    if not norm:
        return {}
    return {index: value / norm for index, value in weights.items()}


def evaluate_trees(baseline: float, trees: Sequence[Sequence[Sequence[float]]], features: Sequence[float]) -> float:
    """Evaluate exported HistGradientBoosting trees.

    Each node is [is_leaf, value, feature_idx, threshold, left, right]; the
    traversal reproduces sklearn's numeric-threshold predict exactly for
    finite inputs.
    """

    total = baseline
    for tree in trees:
        node = tree[0]
        while not node[0]:
            node = tree[int(node[4])] if features[int(node[2])] <= node[3] else tree[int(node[5])]
        total += node[1]
    return total


class KnnIndex:
    """Inverted index over stored train vectors for cosine kNN."""

    def __init__(
        self,
        vectors: Sequence[Mapping[int, float]],
        targets: Sequence[Sequence[float]],
    ) -> None:
        if len(vectors) != len(targets):
            raise ValueError("kNN vectors and targets must align")
        self.targets = [tuple(float(v) for v in row) for row in targets]
        postings: Dict[int, List[Tuple[int, float]]] = {}
        for document, vector in enumerate(vectors):
            for index, value in vector.items():
                postings.setdefault(index, []).append((document, value))
        self.postings = postings

    def predict(
        self, query: Mapping[int, float], neighbors: int = NEIGHBORS
    ) -> Tuple[Tuple[float, ...], float]:
        """Return (weighted neighbor target row, top-1 similarity)."""

        if not query:
            return (), 0.0
        scores: Dict[int, float] = {}
        for index, value in query.items():
            for document, stored in self.postings.get(index, ()):
                scores[document] = scores.get(document, 0.0) + value * stored
        if not scores:
            return (), 0.0
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:neighbors]
        total = math.fsum(similarity for _document, similarity in ranked)
        if total <= 1e-9:
            return (), 0.0
        width = len(self.targets[0])
        blended = [0.0] * width
        for document, similarity in ranked:
            weight = similarity / total
            row = self.targets[document]
            for position in range(width):
                blended[position] += weight * row[position]
        return tuple(blended), ranked[0][1]
