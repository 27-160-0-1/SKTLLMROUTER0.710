# SPDX-License-Identifier: Apache-2.0
"""b04 - runtime cost of the memory-heavy kNN options, in the units the stdlib
runtime actually pays: number of (posting, multiply-add) operations per query.

`similarity.KnnIndex.predict` scans, for every non-zero bin of the query vector,
the whole postings list of that bin.  So the per-episode work is
    sum_over_query_bins len(postings[bin])
which is exactly measurable from the index, independent of the host's numpy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")

lab = lib.MemLab(verbose=False)
fit = np.arange(lab.n)                       # deployed index = train+dev (E11)
dev = lab.dev_idx
print(f"{'index':>10} {'nnz':>10} {'MB(raw)':>8} {'ops/query mean':>15} {'p95':>10} {'x vs 256':>9}")
base = None
for tc in (128, 256, 512, 1024, 0):
    Q, V = lib.tfidf_view(lab.C, fit, tc)
    Vc = V.tocsc()
    plen = np.diff(Vc.indptr).astype(float)          # postings length per bin
    ops = np.asarray((Q[dev] > 0).astype(float) @ plen).ravel()
    if tc == 256:
        base = ops.mean()
    print(f"{tc if tc else 'exact':>10} {V.nnz:10d} {V.nnz*8/1024/1024:8.2f} "
          f"{ops.mean():15.0f} {np.percentile(ops,95):10.0f} "
          f"{ops.mean()/base if base else float('nan'):9.2f}")

print("\nk only changes the top-k selection, not the scan:")
Q, V = lib.tfidf_view(lab.C, fit, 256)
Vc = V.tocsc(); plen = np.diff(Vc.indptr).astype(float)
ops = np.asarray((Q[dev] > 0).astype(float) @ plen).ravel()
cand = np.asarray(((Q[dev] > 0).astype(float) @ (Vc > 0).astype(float).T > 0).sum(axis=1)).ravel()
print(f"  candidate docs touched per query: mean {cand.mean():.0f} p95 {np.percentile(cand,95):.0f} "
      f"of {len(fit)}   (sort cost ~ cand*log(cand), k only slices the sorted head)")
print(f"  multiply-adds per query at top-256: mean {ops.mean():.0f}")
