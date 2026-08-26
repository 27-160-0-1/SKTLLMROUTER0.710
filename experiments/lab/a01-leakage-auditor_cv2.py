# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""A01: mechanism check -- is a dev item's cost easier to predict when the
neighbour pool contains the other dev items (the E39/E43b 5-fold-over-combined
regime) than when it contains only train (the honest regime)?"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labdata as L
from ossp_router import learned_router, similarity

tr, dv = L.load_all()
raw = json.loads((ROOT/"reports/holdout_local/learned-router.v1.json").read_text(encoding="utf-8"))
art = learned_router.parse_artifact(raw)
idf = art.augmentation.idf                     # built on train only, held fixed for both arms

tgt_tr = np.hstack([tr.score, np.log(tr.cost)])
tgt_dv = np.hstack([dv.score, np.log(dv.cost)])

vec_tr = [similarity.tfidf_vector(t, idf, top_components=similarity.TOP_COMPONENTS) for t in tr.texts]
vec_dv = [similarity.tfidf_vector(t, idf, top_components=similarity.TOP_COMPONENTS) for t in dv.texts]

def knn_predict(pool_vecs, pool_tgts, query_texts, skip_offset=None):
    idx = similarity.KnnIndex(pool_vecs, pool_tgts.tolist())
    out = []
    for i, t in enumerate(query_texts):
        q = similarity.tfidf_vector(t, idf)
        sc = {}
        for g, v in q.items():
            for d, s in idx.postings.get(g, ()):
                if skip_offset is not None and d == skip_offset + i:
                    continue
                sc[d] = sc.get(d, 0.0) + v * s
        if not sc:
            out.append(pool_tgts.mean(0)); continue
        ranked = sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:16]
        tot = sum(s for _d, s in ranked)
        out.append(sum((s/tot)*pool_tgts[d] for d, s in ranked))
    return np.asarray(out)

A = knn_predict(vec_tr, tgt_tr, dv.texts)                                   # honest: train-only pool
pool_v = vec_tr + vec_dv
pool_t = np.vstack([tgt_tr, tgt_dv])
B = knn_predict(pool_v, pool_t, dv.texts, skip_offset=len(vec_tr))          # CV-over-combined-like

names = ["s_light","s_mid","s_k1","logc_light","logc_mid","logc_k1"]
truth = tgt_dv
print("dev-item kNN feature quality (k=16), same idf, same code:")
print(f"{'target':12s} {'train-only pool':>16s} {'train+dev pool':>16s} {'delta':>9s}")
for j, nm in enumerate(names):
    ca = np.corrcoef(A[:, j], truth[:, j])[0, 1]
    cb = np.corrcoef(B[:, j], truth[:, j])[0, 1]
    print(f"{nm:12s} {ca:16.4f} {cb:16.4f} {cb-ca:+9.4f}")
print()
for j, nm in enumerate(names[3:], start=3):
    ra = float(np.sqrt(((A[:, j]-truth[:, j])**2).mean()))
    rb = float(np.sqrt(((B[:, j]-truth[:, j])**2).mean()))
    print(f"{nm:12s} log-RMSE  train-only {ra:.4f}   train+dev {rb:.4f}   ({100*(rb-ra)/ra:+.1f}%)")
