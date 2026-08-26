# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a08 probe 15: does a tail/head+tail kNN window rescue the longdoc family?

Leave-one-out kNN (k=16, similarity weighting, exactly similarity.KnnIndex
semantics) restricted to longdoc, under three text windows.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_all  # noqa: E402
from ossp_router import similarity  # noqa: E402

tr, dv = load_all()
texts = tr.texts + dv.texts
fams = np.array([similarity.classify_family(t) for t in texts])
score = np.vstack([tr.score, dv.score])
cost = np.vstack([tr.cost, dv.cost])
targets = np.hstack([score, np.log(cost)])

WINDOWS = {
    "head 4k (deployed)": lambda t: t[:4000],
    "tail 4k": lambda t: t[-4000:],
    "head 2k + tail 2k": lambda t: (t[:2000] + " … " + t[-2000:]) if len(t) > 4000 else t,
    "tail 1k": lambda t: t[-1000:],
}

for scope in ("longdoc", "all"):
    idx = np.where(fams == "longdoc")[0] if scope == "longdoc" else np.arange(len(texts))
    print(f"\n=== kNN quality, scope={scope} (n={len(idx)}), leave-one-out ===")
    print(f"  {'window':22s} {'top1 sim':>9s} {'corr s0':>8s} {'corr s1':>8s} {'corr s2':>8s} "
          f"{'logc2 err sd':>13s}")
    for name, fn in WINDOWS.items():
        wt = [fn(texts[i]) for i in idx]
        freqs, tot = similarity.document_frequencies(wt)
        idf = similarity.idf_table(freqs, tot)
        vecs = [similarity.tfidf_vector(t, idf, top_components=similarity.TOP_COMPONENTS)
                for t in wt]
        post = {}
        for d, v in enumerate(vecs):
            for g, val in v.items():
                post.setdefault(g, []).append((d, val))
        pred = np.zeros((len(idx), 6))
        top1 = np.zeros(len(idx))
        gmean = targets[idx].mean(axis=0)
        for k in range(len(idx)):
            q = similarity.tfidf_vector(wt[k], idf)
            sc = {}
            for g, val in q.items():
                for d, st in post.get(g, ()):
                    if d == k:
                        continue
                    sc[d] = sc.get(d, 0.0) + val * st
            if not sc:
                pred[k] = gmean
                continue
            rk = sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:similarity.NEIGHBORS]
            s = sum(v for _d, v in rk)
            top1[k] = rk[0][1]
            if s <= 1e-9:
                pred[k] = gmean
                continue
            row = np.zeros(6)
            for d, v in rk:
                row += (v / s) * targets[idx[d]]
            pred[k] = row
        cs = [np.corrcoef(pred[:, j], score[idx, j])[0, 1] for j in range(3)]
        lce = np.std(pred[:, 5] - np.log(cost[idx, 2]))
        print(f"  {name:22s} {np.median(top1):9.3f} {cs[0]:8.3f} {cs[1]:8.3f} {cs[2]:8.3f} "
              f"{lce:13.3f}")
