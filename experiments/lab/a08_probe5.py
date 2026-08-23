# SPDX-License-Identifier: Apache-2.0
"""a08 probe 5: exact tokenizer vs cheap statistics for input-token estimation.

Uses the locally cached skt/A.X-3.1-Light tokenizer.json (Apache-2.0, the same
family the challenge evaluates) purely as an analysis tool.
"""
from __future__ import annotations
import glob
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_all, MODEL_IDS  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub"))
paths = {
    "ax31light": glob.glob(str(CACHE / "models--skt--A.X-3.1-Light/snapshots/*/tokenizer.json")),
    "axk1": glob.glob(str(CACHE / "models--skt--A.X-K1/snapshots/*/tokenizer.json")),
}
tr, dv = load_all()
texts = tr.texts + dv.texts
fams = np.array([classify_family(t) for t in texts])
itok = np.vstack([tr.itok, dv.itok])
ntr = len(tr)

for tag, p in paths.items():
    if not p:
        print(f"{tag}: tokenizer not cached")
        continue
    tk = Tokenizer.from_file(p[0])
    enc = tk.encode_batch_fast(texts) if hasattr(tk, "encode_batch_fast") else tk.encode_batch(texts)
    ours = np.array([len(e.ids) for e in enc], dtype=float)
    print(f"\n=== tokenizer {tag} ({Path(p[0]).parent.name[:8]}) ===")
    for j, m in enumerate(MODEL_IDS):
        d = itok[:, j] - ours
        print(f"  vs {m:11s}: offset median={np.median(d):7.1f} mean={d.mean():8.1f} sd={d.std():8.1f}")
    j = 0
    d = itok[:, j] - ours
    print(f"  per-family offset (itok[light] - our token count):")
    print(f"  {'family':16s} {'n':>5s} {'median':>8s} {'iqr':>8s} {'p05':>8s} {'p95':>8s} "
          f"{'const?':>7s}")
    off = {}
    for f in sorted(set(fams)):
        msk = fams == f
        v = d[msk]
        off[f] = float(np.median(v))
        print(f"  {f:16s} {msk.sum():5d} {np.median(v):8.1f} "
              f"{np.percentile(v,75)-np.percentile(v,25):8.1f} {np.percentile(v,5):8.1f} "
              f"{np.percentile(v,95):8.1f} "
              f"{'YES' if np.percentile(v,95)-np.percentile(v,5) < 4 else 'no':>7s}")
    # accuracy of "tokenizer + family median offset"
    pred = ours + np.array([off[f] for f in fams])
    for j, m in enumerate(MODEL_IDS):
        y = itok[:, j]
        # fit offsets on train only, evaluate on dev
        offt = {f: float(np.median((y - ours)[(fams == f) & (np.arange(len(y)) < ntr)]))
                for f in sorted(set(fams))}
        pd_ = ours + np.array([offt[f] for f in fams])
        dev = slice(ntr, None)
        yy, pp = y[dev], pd_[dev]
        r2 = 1 - ((pp - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
        ape = np.abs(pp - yy) / np.maximum(yy, 1)
        lg = np.log(np.maximum(pp, 1)) - np.log(np.maximum(yy, 1))
        print(f"  [train-fit offsets -> dev] {m:11s} R2={r2:.6f} medAPE={np.median(ape):.5f} "
              f"sum_ratio={pp.sum()/yy.sum():.5f} log-err sd={lg.std():.4f} "
              f"exact={np.mean(np.abs(pp-yy)<1e-9):.3f}")

print("\n=== cheap-stat baseline for the same target (train-fit -> dev) ===")
import re  # noqa: E402
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)


def stats(t):
    ws = _SPACE.sub(" ", t)
    w = _WORD.findall(t)
    return [len(t), len(t.encode("utf-8")), len(ws), len(w), t.count("\n"),
            sum(1 for c in t if "가" <= c <= "힣"), sum(c.isdigit() for c in t),
            sum(1 for c in t if ord(c) >= 128),
            sum(1 for c in t if (not c.isalnum()) and (not c.isspace())),
            sum(len(x) for x in w), len(t) - len(ws)]


X = np.array([stats(t) for t in texts], dtype=float)
FS = sorted(set(fams))
F = np.array([[float(f == g) for g in FS] for f in fams])
A = np.hstack([X, F, np.ones((len(X), 1))])
for j, m in enumerate(MODEL_IDS):
    y = itok[:, j]
    coef, *_ = np.linalg.lstsq(A[:ntr], y[:ntr], rcond=None)
    pp = A[ntr:] @ coef
    yy = y[ntr:]
    r2 = 1 - ((pp - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    lg = np.log(np.maximum(pp, 1)) - np.log(np.maximum(yy, 1))
    print(f"  {m:11s} R2={r2:.6f} medAPE={np.median(np.abs(pp-yy)/np.maximum(yy,1)):.5f} "
          f"sum_ratio={pp.sum()/yy.sum():.5f} log-err sd={lg.std():.4f}")
