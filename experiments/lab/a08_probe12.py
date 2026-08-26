# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a08 probe 12: Korean tokenisation.  learned_router._TOKEN emits a whole Hangul
run as ONE token, so agglutinated forms ("탐사선의 / 탐사선을 / 탐사선이") are
distinct types.  How much of the word block is wasted on hapax legomena, and how
much would a 2-syllable stem (or particle stripping) recover?
"""
from __future__ import annotations
import re
import sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_all  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", re.UNICODE)
PARTICLES = ("에서는", "으로는", "이라고", "에게서", "에서", "으로", "에게", "까지", "부터",
             "라고", "이나", "처럼", "보다", "마다", "조차", "밖에", "이다", "입니다",
             "는", "은", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만", "로")

tr, dv = load_all()
texts = tr.texts + dv.texts
fams = np.array([classify_family(t) for t in texts])


def toks(t):
    return [x.casefold() if not x.isdecimal() else "<number>" for x in TOKEN.findall(t)]


def stem(tok):
    if not ("가" <= tok[0] <= "힣"):
        return tok
    for p in PARTICLES:
        if len(tok) > len(p) + 1 and tok.endswith(p):
            return tok[: -len(p)]
    return tok


def stem2(tok):
    if not ("가" <= tok[0] <= "힣"):
        return tok
    return tok[:2] if len(tok) > 2 else tok


print(f"{'family':16s} {'items':>6s} {'tokens':>9s} {'types':>8s} {'hapax%':>7s} "
      f"{'hapax% part-strip':>18s} {'hapax% 2-syl stem':>18s}")
for f in sorted(set(fams)):
    idx = np.where(fams == f)[0]
    all_t = []
    for i in idx:
        all_t.extend(toks(texts[i]))
    c = Counter(all_t)
    c1 = Counter(stem(x) for x in all_t)
    c2 = Counter(stem2(x) for x in all_t)

    def hap(cc):
        return 100.0 * sum(1 for v in cc.values() if v == 1) / max(sum(cc.values()), 1)

    print(f"{f:16s} {len(idx):6d} {len(all_t):9d} {len(c):8d} {hap(c):7.2f} "
          f"{hap(c1):18.2f} {hap(c2):18.2f}")

print("\n=== hangul-only view (belebele + hrmcr) ===")
idx = np.where((fams == "belebele") | (fams == "hrmcr"))[0]
hk = []
for i in idx:
    hk.extend([x for x in toks(texts[i]) if "가" <= x[0] <= "힣"])
c = Counter(hk)
print(f"  hangul tokens={len(hk)} types={len(c)} "
      f"type/token={len(c)/len(hk):.3f} hapax={100*sum(1 for v in c.values() if v==1)/len(hk):.2f}%")
c1 = Counter(stem(x) for x in hk)
c2 = Counter(stem2(x) for x in hk)
print(f"  particle-strip: types={len(c1)} type/token={len(c1)/len(hk):.3f} "
      f"hapax={100*sum(1 for v in c1.values() if v==1)/len(hk):.2f}%")
print(f"  2-syllable stem: types={len(c2)} type/token={len(c2)/len(hk):.3f} "
      f"hapax={100*sum(1 for v in c2.values() if v==1)/len(hk):.2f}%")
print("  most common:", c.most_common(10))

print("\n=== how many distinct hashed word features does an average item use? ===")
for f in sorted(set(fams)):
    idx = np.where(fams == f)[0]
    u = [len(set(toks(texts[i]))) for i in idx]
    n = [len(toks(texts[i])) for i in idx]
    print(f"  {f:16s} median distinct types/item={np.median(u):6.0f} of "
          f"{np.median(n):6.0f} tokens  (8192 bins)")
