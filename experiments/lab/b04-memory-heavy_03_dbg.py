# SPDX-License-Identifier: Apache-2.0
"""b04 - locate the discrepancy between the replica and the shipped kNN."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router import similarity

lab = lib.MemLab()
fit = lab.train_idx
texts_fit = [lab.texts[i] for i in fit]
freqs, total = similarity.document_frequencies(texts_fit)
idf = similarity.idf_table(freqs, total)
vecs = [similarity.tfidf_vector(t, idf, top_components=similarity.TOP_COMPONENTS) for t in texts_fit]

Q, V = lib.tfidf_view(lab.C, fit, 256)
print("V indices sorted:", V.has_sorted_indices, " Q sorted:", Q.has_sorted_indices)

# compare vector 0
v0 = vecs[0]
a, b = V.indptr[0], V.indptr[0 + 1]
mine = dict(zip(V.indices[a:b].tolist(), V.data[a:b].tolist()))
print("ref nnz", len(v0), "mine nnz", len(mine))
common = set(v0) & set(mine)
print("common", len(common), "ref-only", len(set(v0) - set(mine)), "mine-only", len(set(mine) - set(v0)))
if common:
    d = max(abs(v0[k] - mine[k]) for k in common)
    print("max |diff| on common:", d)

# query vector for a dev row
di = int(lab.dev_idx[0])
q0 = similarity.tfidf_vector(lab.texts[di], idf)
a, b = Q.indptr[di], Q.indptr[di + 1]
mq = dict(zip(Q.indices[a:b].tolist(), Q.data[a:b].tolist()))
print("query ref nnz", len(q0), "mine", len(mq),
      "maxdiff", max((abs(q0[k] - mq.get(k, 0.0)) for k in q0), default=0))

# similarity row
sc = {}
for g, v in q0.items():
    for d_, s in lib.knn_index_postings(V).get(g, ()) if False else ():
        pass
S = (Q[[di]] @ V.T).toarray().ravel()
ref = {}
idxobj = similarity.KnnIndex(vecs, lab.targets[fit].tolist())
for g, v in q0.items():
    for d_, s in idxobj.postings.get(g, ()):
        ref[d_] = ref.get(d_, 0.0) + v * s
rr = np.zeros(len(fit))
for d_, s in ref.items():
    rr[d_] = s
print("sim max|diff|", np.abs(rr - S).max())
o1 = sorted(ref.items(), key=lambda kv: (-kv[1], kv[0]))[:16]
o2 = np.argsort(-S, kind="stable")[:16]
print("ref top16", [(d, round(s, 6)) for d, s in o1[:5]])
print("mine top16", [(int(d), round(float(S[d]), 6)) for d in o2[:5]])
