# SPDX-License-Identifier: Apache-2.0
"""b04 - what the memory budget is actually spent on today, and what each
memory-heavy option would cost.  Everything here is measured, not estimated."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")

ROOT = Path(".")
RES = ROOT / "src/ossp_router/resources"


def mb(x):
    return x / 1024 / 1024


print("== shipped artifacts ==")
tot = 0
for f in sorted(RES.glob("*.json")):
    tot += f.stat().st_size
    print(f"  {f.name:34s} {mb(f.stat().st_size):8.2f} MB")
print(f"  {'TOTAL':34s} {mb(tot):8.2f} MB")

art = json.loads((RES / "learned-router.v1.json").read_text(encoding="utf-8"))
heavy = json.loads((RES / "learned-router-heavy.v1.json").read_text(encoding="utf-8"))
print("\n== top-level keys ==")
for name, d in (("light", art), ("heavy", heavy)):
    for k, v in d.items():
        s = len(json.dumps(v))
        if s > 20_000:
            print(f"  {name:6s} {k:32s} {mb(s):8.2f} MB")


def count_nodes(obj):
    if isinstance(obj, list):
        return sum(count_nodes(o) for o in obj)
    return 1


for name, d in (("light", art), ("heavy", heavy)):
    for k in d:
        if "tree" in k.lower() or "head" in k.lower() or "model" in k.lower():
            try:
                print(f"  {name} {k}: json bytes {mb(len(json.dumps(d[k]))):.2f} MB")
            except Exception:
                pass

print("\n== kNN index sizes (measured nnz) ==")
lab = lib.MemLab(verbose=False)
fit = np.arange(2640)          # the deployed router indexes train+dev (E11)
for tc in (128, 256, 512, 1024, 0):
    Q, V = lib.tfidf_view(lab.C, fit, tc)
    nnz = V.nnz
    # runtime representation: postings list of (doc:int32, value:float32)
    print(f"  top_components={tc if tc else 'exact':>6}  nnz={nnz:9d}  "
          f"mean nnz/doc={nnz/len(fit):7.1f}  raw(4B idx+4B val)={mb(nnz*8):7.2f} MB  "
          f"json(~14B/pair)={mb(nnz*14):7.2f} MB")

print("\n== per-episode vector sizes ==")
nz = np.diff(lab.C.indptr)
print(f"  distinct char-ngram bins per episode: mean {nz.mean():.0f} median {np.median(nz):.0f} "
      f"p95 {np.percentile(nz,95):.0f} max {nz.max()}")
nzw = np.diff(lab.Wd.indptr)
print(f"  distinct word-ngram bins per episode: mean {nzw.mean():.0f} median {np.median(nzw):.0f} "
      f"p95 {np.percentile(nzw,95):.0f} max {nzw.max()}")
print(f"  prompt chars: mean {np.mean([len(t) for t in lab.texts]):.0f} "
      f"total {sum(len(t) for t in lab.texts)/1e6:.2f} M chars "
      f"({mb(sum(len(t.encode('utf-8')) for t in lab.texts)):.2f} MB utf-8)")
