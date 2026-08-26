# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""A01 Q3: is the meta-GBM's kNN feature distribution-matched between the
leave-one-out training rows and the runtime (dev) rows?"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import learned_router, similarity
from ossp_router.heuristic import episode_text
from ossp_router.protocol import load_input

raw = json.loads((ROOT/"reports/holdout_local/learned-router.v1.json").read_text(encoding="utf-8"))
art = learned_router.parse_artifact(raw)
aug = art.augmentation
idx = aug.index
print("holdout kNN index rows:", len(idx.targets))

tr = load_input(ROOT/"data/materialized/train/inputs.json")
dv = load_input(ROOT/"data/materialized/dev/inputs.json")
tr_texts = [episode_text(e) for e in tr.episodes]
dv_texts = [episode_text(e) for e in dv.episodes]

def loo_top1(texts):
    out = []
    for i, t in enumerate(texts):
        q = similarity.tfidf_vector(t, aug.idf)
        scores = {}
        for g, v in q.items():
            for d, s in idx.postings.get(g, ()):
                if d == i:
                    continue
                scores[d] = scores.get(d, 0.0) + v * s
        out.append(max(scores.values()) if scores else 0.0)
    return np.array(out)

def plain_top1(texts):
    out = []
    for t in texts:
        q = similarity.tfidf_vector(t, aug.idf)
        _r, top1 = idx.predict(q)
        out.append(top1)
    return np.array(out)

a = loo_top1(tr_texts)      # exactly what build_meta_gbm._loo_knn feeds the GBM
b = plain_top1(dv_texts)    # what the runtime feeds the GBM on dev
c = plain_top1(tr_texts)    # self-inclusive, for reference
for name, v in (("train LOO (meta training feature)", a), ("dev runtime", b), ("train self-inclusive", c)):
    print(f"{name:36s} n={len(v):5d} mean={v.mean():.4f} med={np.median(v):.4f} "
          f"p10={np.quantile(v,.1):.4f} p90={np.quantile(v,.9):.4f} frac>0.5={float((v>0.5).mean()):.3f}")
print()
# how well does the kNN row predict the truth in each regime?
import labdata as L
sys.path.insert(0, str(Path(__file__).resolve().parent))
trs, dvs = L.load_all()

def knn_rows(texts, loo):
    tg = np.asarray(idx.targets)
    rows = []
    for i, t in enumerate(texts):
        q = similarity.tfidf_vector(t, aug.idf)
        scores = {}
        for g, v in q.items():
            for d, s in idx.postings.get(g, ()):
                if loo and d == i:
                    continue
                scores[d] = scores.get(d, 0.0) + v * s
        if not scores:
            rows.append(tg.mean(0)); continue
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:16]
        tot = sum(s for _d, s in ranked)
        rows.append(sum((s/tot)*tg[d] for d, s in ranked))
    return np.asarray(rows)

kt = knn_rows(tr_texts, True)
kd = knn_rows(dv_texts, False)
for j, nm in enumerate(["s_light","s_mid","s_k1"]):
    ct = np.corrcoef(kt[:, j], trs.score[:, j])[0, 1]
    cd = np.corrcoef(kd[:, j], dvs.score[:, j])[0, 1]
    print(f"kNN score corr {nm}: train-LOO {ct:.4f}  dev {cd:.4f}  gap {ct-cd:+.4f}")
for j, nm in enumerate(["c_light","c_mid","c_k1"]):
    ct = np.corrcoef(kt[:, 3+j], np.log(trs.cost[:, j]))[0, 1]
    cd = np.corrcoef(kd[:, 3+j], np.log(dvs.cost[:, j]))[0, 1]
    print(f"kNN logcost corr {nm}: train-LOO {ct:.4f}  dev {cd:.4f}  gap {ct-cd:+.4f}")
