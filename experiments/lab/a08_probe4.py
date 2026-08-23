# SPDX-License-Identifier: Apache-2.0
"""a08 probe 4: what does the kNN representation actually see?

similarity.hashed_counts truncates at TEXT_LIMIT=4,000 chars, HEAD ONLY.
learned_router._normalized_char_text keeps head 3k + tail 3k.
The word block has NO limit at all.
Measure per family: fraction of the prompt visible to each path, and for
longdoc specifically whether the question (at the tail) is inside the window.
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
from ossp_router import similarity, learned_router  # noqa: E402

tr, dv = load_all()
texts = tr.texts + dv.texts
fams = [similarity.classify_family(t) for t in texts]

print(f"{'family':16s} {'n':>5s} {'medlen':>8s} {'knn_cov%':>9s} {'char_cov%':>10s} {'wordtok_med':>12s}")
for f in sorted(set(fams)):
    idx = [i for i, g in enumerate(fams) if g == f]
    L = np.array([len(texts[i]) for i in idx])
    knn_cov = np.minimum(similarity.TEXT_LIMIT, L) / L
    char_cov = np.minimum(6000, L) / L
    ntok = np.array([len(re.findall(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", texts[i])) for i in idx])
    print(f"{f:16s} {len(idx):5d} {np.median(L):8.0f} {100*np.median(knn_cov):9.1f} "
          f"{100*np.median(char_cov):10.1f} {np.median(ntok):12.0f}")

print()
print("=== longdoc: is the question inside each window? ===")
li = [i for i, g in enumerate(fams) if g == "longdoc"]
qpat = re.compile(r"\n(?:How many|Where is|What is|Where was|Is |Who )", re.IGNORECASE)
inhead4k = inhead6k = intail = 0
qs = []
for i in li:
    t = texts[i]
    m = list(qpat.finditer(t))
    pos = m[-1].start() if m else None
    if pos is None:
        continue
    qs.append(t[pos:pos + 60].strip().replace("\n", " "))
    if pos < 4000:
        inhead4k += 1
    if pos < 3000:
        inhead6k += 1
    if pos > len(t) - 3000:
        intail += 1
print(f"  n with a detected trailing question: {len(qs)} / {len(li)}")
print(f"  question start < 4000 (kNN window)          : {inhead4k}")
print(f"  question start < 3000 (char head half)      : {inhead6k}")
print(f"  question inside last 3000 chars (char tail) : {intail}")
from collections import Counter  # noqa: E402
print("  question templates:", Counter(q.split("?")[0][:40] for q in qs).most_common(8))

print()
print("=== how distinguishable are longdoc items to the kNN path? ===")
freqs, total = similarity.document_frequencies([texts[i] for i in li])
idf = similarity.idf_table(freqs, total)
V = [similarity.tfidf_vector(texts[i], idf, top_components=256) for i in li]


def cos(a, b):
    return sum(v * b.get(k, 0.0) for k, v in a.items())


sims = []
for a in range(0, len(V), 3):
    for b in range(a + 1, len(V), 7):
        sims.append(cos(V[a], V[b]))
sims = np.array(sims)
print(f"  head-4k tf-idf pairwise cosine: mean={sims.mean():.3f} p90={np.percentile(sims,90):.3f} "
      f"max={sims.max():.3f}")

tails = [texts[i][-4000:] for i in li]
freqs2, total2 = similarity.document_frequencies(tails)
idf2 = similarity.idf_table(freqs2, total2)
V2 = [similarity.tfidf_vector(t, idf2, top_components=256) for t in tails]
sims2 = []
for a in range(0, len(V2), 3):
    for b in range(a + 1, len(V2), 7):
        sims2.append(cos(V2[a], V2[b]))
sims2 = np.array(sims2)
print(f"  tail-4k tf-idf pairwise cosine: mean={sims2.mean():.3f} p90={np.percentile(sims2,90):.3f} "
      f"max={sims2.max():.3f}")

print()
print("=== longdoc score/cost spread (is there anything to predict?) ===")
S = np.vstack([tr.score, dv.score])[li]
C = np.vstack([tr.cost, dv.cost])[li]
print(f"  n={len(li)} mean score={np.round(S.mean(0),3)} sd={np.round(S.std(0),3)}")
print(f"  mean cost ratio vs light={np.round(C.mean(0)/C[:,0].mean(),2)}")
