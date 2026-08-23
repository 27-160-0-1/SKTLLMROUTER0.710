# SPDX-License-Identifier: Apache-2.0
"""a08 probe 19: does Korean stemming add signal?  Cheap within-family test.

Word 1/2-gram hashed block only (8,192 bins, signed, L2), ridge, 5-fold OOF,
restricted to belebele+hrmcr (the Korean families), target = each model's score
and log cost.  Three tokenisations: deployed, particle-stripped, 2-syllable stem
(both stem forms are emitted *in addition* to the raw token in the "+stem" rows
so no information is lost).
"""
from __future__ import annotations
import re
import sys
import zlib
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_all  # noqa: E402
from ossp_router import similarity  # noqa: E402

TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", re.UNICODE)
PARTICLES = ("에서는", "으로는", "이라고", "에게서", "에서", "으로", "에게", "까지", "부터",
             "라고", "이나", "처럼", "보다", "마다", "조차", "밖에", "이다", "입니다",
             "는", "은", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만", "로")
BINS = 8192

tr, dv = load_all()
texts = tr.texts + dv.texts
fams = np.array([similarity.classify_family(t) for t in texts])
score = np.vstack([tr.score, dv.score])
cost = np.vstack([tr.cost, dv.cost])
Y = np.hstack([score, np.log(cost)])


def base_tokens(t):
    return [x.casefold() if not x.isdecimal() else "<number>" for x in TOKEN.findall(t)]


def strip_particle(tok):
    if not ("가" <= tok[0] <= "힣"):
        return None
    for p in PARTICLES:
        if len(tok) > len(p) + 1 and tok.endswith(p):
            return tok[: -len(p)]
    return None


def syl2(tok):
    if not ("가" <= tok[0] <= "힣") or len(tok) <= 2:
        return None
    return tok[:2]


MODES = {
    "deployed": None,
    "+particle stem": strip_particle,
    "+2-syllable stem": syl2,
    "replace w/ 2-syl": "replace",
}


def row(t, mode):
    toks = base_tokens(t)
    if mode == "replace":
        toks = [syl2(x) or x for x in toks]
        extra = []
    elif mode is None:
        extra = []
    else:
        extra = [f"w1:{mode(x)}" for x in toks if mode(x)]
    vals = [f"w1:{x}" for x in toks]
    vals += [f"w2:{a}\x1f{b}" for a, b in zip(toks, toks[1:])]
    vals += extra
    counts = {}
    for v, c in Counter(vals).items():
        d = zlib.crc32(v.encode("utf-8"))
        i = (d & 0x7FFFFFFF) & (BINS - 1)
        counts[i] = counts.get(i, 0.0) + (-c if d & 0x80000000 else c)
    n = np.sqrt(sum(x * x for x in counts.values())) or 1.0
    return {k: v / n for k, v in counts.items() if v}


for scope, mask in (("belebele+hrmcr", (fams == "belebele") | (fams == "hrmcr")),
                    ("belebele only", fams == "belebele")):
    idx = np.where(mask)[0]
    print(f"\n=== scope={scope} n={len(idx)} ===")
    print(f"  {'tokenisation':20s} {'nnz cols':>9s} " +
          " ".join(f"{'corr s' + str(j):>8s}" for j in range(3)) +
          " " + " ".join(f"{'clog' + str(j):>7s}" for j in range(3)))
    rng = np.random.default_rng(123)
    fold = rng.integers(0, 5, size=len(idx))
    for name, mode in MODES.items():
        r_, c_, v_ = [], [], []
        for k, i in enumerate(idx):
            for c, v in row(texts[i], mode).items():
                r_.append(k); c_.append(c); v_.append(v)
        X = sparse.csr_matrix((v_, (r_, c_)), shape=(len(idx), BINS))
        oof = np.zeros((len(idx), 6))
        for f in range(5):
            te = fold == f
            m = Ridge(alpha=1.0, solver="sparse_cg").fit(X[~te], Y[idx][~te])
            oof[te] = m.predict(X[te])
        cs = [np.corrcoef(oof[:, j], score[idx, j])[0, 1] for j in range(3)]
        ce = [np.std(oof[:, 3 + j] - np.log(cost[idx, j])) for j in range(3)]
        print(f"  {name:20s} {int((X != 0).sum(axis=0).astype(bool).sum()):9d} " +
              " ".join(f"{x:8.3f}" for x in cs) + " " + " ".join(f"{x:7.3f}" for x in ce))
