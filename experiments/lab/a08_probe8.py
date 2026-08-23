# SPDX-License-Identifier: Apache-2.0
"""a08 probe 8: what are the two sub-sources inside `gsm8k_or_other`, and is any
other regex family similarly heterogeneous?
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_all, MODEL_IDS  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

tr, dv = load_all()
texts = tr.texts + dv.texts
fams = np.array([classify_family(t) for t in texts])
ngen = np.vstack([tr.ngen, dv.ngen])[:, 0]
score = np.vstack([tr.score, dv.score])
cost = np.vstack([tr.cost, dv.cost])
n = len(texts)

g = np.where(fams == "gsm8k_or_other")[0]
print("=== sample gsm8k_or_other, ngen=4 ===")
for i in g[ngen[g] == 4][:4]:
    print(f"  [{len(texts[i]):5d}c] {texts[i][:220]!r}")
print("\n=== sample gsm8k_or_other, ngen=2 ===")
for i in g[ngen[g] == 2][:8]:
    print(f"  [{len(texts[i]):5d}c] {texts[i][:220]!r}")

print("\n=== simple separators ===")
sub = g
y = (ngen[sub] == 4)
tests = {
    "ends with '?'": np.array([texts[i].rstrip().endswith("?") for i in sub]),
    "has '$'": np.array(["$" in texts[i] for i in sub]),
    "has backslash": np.array(["\\" in texts[i] for i in sub]),
    "len<400": np.array([len(texts[i]) < 400 for i in sub]),
    "starts 'Find'/'Let'/'Compute'": np.array(
        [bool(re.match(r"(Find|Let|Compute|Suppose|Determine|Prove)\b", texts[i])) for i in sub]),
    "contains ' $ ' name+verb (gsm style)": np.array(
        [bool(re.search(r"\b(?:has|had|buys|sells|costs|each|total)\b", texts[i])) for i in sub]),
}
for name, t in tests.items():
    if t.sum() == 0 or (~t).sum() == 0:
        continue
    print(f"  {name:38s} P(n4|T)={y[t].mean():.3f} (n={t.sum():3d})  "
          f"P(n4|F)={y[~t].mean():.3f} (n={(~t).sum():3d})")

print("\n=== every family: heterogeneity of the realised light score ===")
print(f"  {'family':16s} {'n':>5s} {'mean':>6s} {'sd':>6s} {'frac0':>6s} {'frac1':>6s} "
      f"{'ngen4%':>7s} {'k1-light':>9s}")
for f in sorted(set(fams)):
    m = fams == f
    s = score[m, 0]
    print(f"  {f:16s} {m.sum():5d} {s.mean():6.3f} {s.std():6.3f} {(s==0).mean():6.3f} "
          f"{(s==1).mean():6.3f} {100*(ngen[m]==4).mean():7.1f} "
          f"{(score[m,2]-score[m,0]).mean():9.3f}")

print("\n=== within-family 2-means split on cheap features: how separable is difficulty? ===")
from sklearn.cluster import KMeans  # noqa: E402


def cheap(t):
    return [len(t), t.count("\n"), len(re.findall(r"\d", t)), t.count("?"),
            len(re.findall(r"[A-Za-z]+", t)), sum(1 for c in t if "가" <= c <= "힣"),
            t.count("$"), t.count("\\"), t.count(","), t.count(".")]


for f in sorted(set(fams)):
    m = np.where(fams == f)[0]
    if len(m) < 60:
        continue
    X = np.array([cheap(texts[i]) for i in m], dtype=float)
    X = (X - X.mean(0)) / np.where(X.std(0) > 1e-9, X.std(0), 1)
    lab = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(X)
    a = score[m[lab == 0], 0].mean()
    b = score[m[lab == 1], 0].mean()
    ca = np.log(cost[m[lab == 0], 2]).mean()
    cb = np.log(cost[m[lab == 1], 2]).mean()
    print(f"  {f:16s} n={len(m):4d} split={(lab==0).sum():4d}/{(lab==1).sum():4d} "
          f"light score {a:.3f} vs {b:.3f} (gap {abs(a-b):.3f})  "
          f"k1 logcost {ca:.2f} vs {cb:.2f} (gap {abs(ca-cb):.2f})")
