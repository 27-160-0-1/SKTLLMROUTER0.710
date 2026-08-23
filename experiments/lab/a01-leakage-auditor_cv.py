# SPDX-License-Identifier: Apache-2.0
"""A01: why do E39/E43b (5-fold CV over combined 2,640) report far lower bust
risk than a train-only holdout?  Two candidate causes, both measured."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labdata as L
from ossp_router import learned_router, similarity
from ossp_router.heuristic import episode_text
from ossp_router.protocol import load_input

tr, dv = L.load_all()

# ---- (1) is a dev item closer to other DEV items than to TRAIN items? ----
raw = json.loads((ROOT/"reports/holdout_local/learned-router.v1.json").read_text(encoding="utf-8"))
art = learned_router.parse_artifact(raw); aug = art.augmentation
dev_texts = dv.texts
dev_vecs = [similarity.tfidf_vector(t, aug.idf, top_components=similarity.TOP_COMPONENTS) for t in dev_texts]
dev_idx = similarity.KnnIndex(dev_vecs, [[0.0]*6]*len(dev_vecs))
top1_dd = []
for i, t in enumerate(dev_texts):
    q = similarity.tfidf_vector(t, aug.idf)
    sc = {}
    for g, v in q.items():
        for d, s in dev_idx.postings.get(g, ()):
            if d == i: continue
            sc[d] = sc.get(d, 0.0) + v*s
    top1_dd.append(max(sc.values()) if sc else 0.0)
top1_dd = np.array(top1_dd)
top1_dt = []
for t in dev_texts:
    q = similarity.tfidf_vector(t, aug.idf)
    _r, s = aug.index.predict(q)
    top1_dt.append(s)
top1_dt = np.array(top1_dt)
print("dev item's top-1 neighbour similarity")
print(f"  vs TRAIN pool (1,760): mean={top1_dt.mean():.4f} med={np.median(top1_dt):.4f} frac>0.6={float((top1_dt>0.6).mean()):.3f}")
print(f"  vs other DEV  (  879): mean={top1_dd.mean():.4f} med={np.median(top1_dd):.4f} frac>0.6={float((top1_dd>0.6).mean()):.3f}")
print(f"  -> a 5-fold CV over the combined 2,640 gives every held-out dev item access to 4/5 of the other dev items")

# ---- (2) cost tail: dev vs train ----
print("\nk1 (axk1-think) true cost, normalised by the split's mean light cost")
for sp in (tr, dv):
    c = sp.cost[:,2]/sp.cost[:,0].mean()
    print(f"  {sp.name:5s} mean={c.mean():6.2f} p90={np.quantile(c,.9):6.2f} p99={np.quantile(c,.99):7.2f} "
          f"max={c.max():7.2f}  cv={c.std()/c.mean():.3f}")
# bootstrap sd of the premium ratio when the pool is dev(880) vs combined(2640)
print("\nsd of a size-880 bootstrap mean of k1 cost, drawing from:")
rng = np.random.default_rng(3)
for name, c in (("dev only (my harness)", dv.cost[:,2]/dv.cost[:,0].mean()),
                ("train+dev (E39/E43b)", np.concatenate([tr.cost[:,2]/tr.cost[:,0].mean(),
                                                          dv.cost[:,2]/dv.cost[:,0].mean()]))):
    m = np.array([c[rng.integers(0,len(c),880)].mean() for _ in range(4000)])
    print(f"  {name:22s} mean={m.mean():.3f} sd={m.std():.3f}  (relative sd {m.std()/m.mean():.4f})")
