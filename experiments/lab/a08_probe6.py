# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a08 probe 6: num_generations is a hidden multiplier on BOTH tokens and cost.

  - is ngen identical across the three models for an episode?
  - per-family ngen distribution
  - after dividing by ngen, does tokenizer(text) + family constant reproduce
    the per-generation input token count exactly?  (E41 offsets)
  - how much of the cost log-error is pure ngen (a factor of 2)?
  - is ngen predictable from the prompt inside gsm8k_or_other?
"""
from __future__ import annotations
import glob
import os
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
from tokenizers import Tokenizer  # noqa: E402

tr, dv = load_all()
texts = tr.texts + dv.texts
fams = np.array([classify_family(t) for t in texts])
itok = np.vstack([tr.itok, dv.itok])
otok = np.vstack([tr.otok, dv.otok])
ngen = np.vstack([tr.ngen, dv.ngen])
score = np.vstack([tr.score, dv.score])
ntr = len(tr)
n = len(texts)

print("=== ngen consistency across models ===")
same = (ngen[:, 0] == ngen[:, 1]) & (ngen[:, 1] == ngen[:, 2])
print(f"  ngen identical across all 3 models: {same.mean():.4f} ({same.sum()}/{n})")
print("  values:", np.unique(ngen, return_counts=True))

print("\n=== ngen per family ===")
print(f"  {'family':16s} {'n':>5s} {'n=2':>6s} {'n=4':>6s} {'other':>6s}")
for f in sorted(set(fams)):
    m = fams == f
    v = ngen[m, 0]
    print(f"  {f:16s} {m.sum():5d} {(v==2).sum():6d} {(v==4).sum():6d} "
          f"{((v!=2)&(v!=4)).sum():6d}")

CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub"))
p = glob.glob(str(CACHE / "models--skt--A.X-3.1-Light/snapshots/*/tokenizer.json"))
tk = Tokenizer.from_file(p[0])
enc = tk.encode_batch_fast(texts) if hasattr(tk, "encode_batch_fast") else tk.encode_batch(texts)
ours = np.array([len(e.ids) for e in enc], dtype=float)

print("\n=== per-generation input tokens vs tokenizer(text): instruction offset ===")
print(f"  {'family':16s} {'n':>5s} {'median':>8s} {'p05':>8s} {'p95':>8s} {'const?':>7s}")
offs = {}
for f in sorted(set(fams)):
    m = fams == f
    d = itok[m, 0] / ngen[m, 0] - ours[m]
    offs[f] = float(np.median(d))
    const = "YES" if (np.percentile(d, 95) - np.percentile(d, 5)) < 4 else "no"
    print(f"  {f:16s} {m.sum():5d} {np.median(d):8.1f} {np.percentile(d,5):8.1f} "
          f"{np.percentile(d,95):8.1f} {const:>7s}")

print("\n=== accuracy of  itok = ngen * (tokenizer(text) + family offset)  (train-fit -> dev) ===")
dev = np.arange(ntr, n)
for j, m in enumerate(MODEL_IDS):
    ot = {f: float(np.median((itok[:ntr, j] / ngen[:ntr, j] - ours[:ntr])[fams[:ntr] == f]))
          for f in sorted(set(fams))}
    pred = ngen[dev, j] * (ours[dev] + np.array([ot[f] for f in fams[dev]]))
    y = itok[dev, j]
    r2 = 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    lg = np.log(np.maximum(pred, 1)) - np.log(np.maximum(y, 1))
    print(f"  {m:11s} R2={r2:.6f} medAPE={np.median(np.abs(pred-y)/y):.5f} "
          f"sum_ratio={pred.sum()/y.sum():.5f} log-err sd={lg.std():.4f} "
          f"exact={np.mean(np.abs(pred-y)<0.5):.3f}")
print("  (same, but with ngen UNKNOWN -> assume family-modal ngen)")
for j, m in enumerate(MODEL_IDS):
    ot = {f: float(np.median((itok[:ntr, j] / ngen[:ntr, j] - ours[:ntr])[fams[:ntr] == f]))
          for f in sorted(set(fams))}
    gm = {f: float(np.median(ngen[:ntr, j][fams[:ntr] == f])) for f in sorted(set(fams))}
    pred = np.array([gm[f] for f in fams[dev]]) * (ours[dev] + np.array([ot[f] for f in fams[dev]]))
    y = itok[dev, j]
    r2 = 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    lg = np.log(np.maximum(pred, 1)) - np.log(np.maximum(y, 1))
    print(f"  {m:11s} R2={r2:.6f} log-err sd={lg.std():.4f}")

print("\n=== how much cost log-variance is pure ngen? ===")
from labdata import RATES, TOKEN_UNIT  # noqa: E402
cost = np.vstack([tr.cost, dv.cost])
for j, m in enumerate(MODEL_IDS):
    lc = np.log(cost[:, j])
    lcpg = np.log(cost[:, j] / ngen[:, j])
    # residual sd around family mean, with and without the ngen factor
    r1 = lc.copy(); r2_ = lcpg.copy()
    for f in set(fams):
        k = fams == f
        r1[k] -= lc[k].mean(); r2_[k] -= lcpg[k].mean()
    print(f"  {m:11s} sd[log c | family]={r1.std():.3f}   "
          f"sd[log (c/ngen) | family]={r2_.std():.3f}   "
          f"reduction={(1-r2_.std()/r1.std())*100:5.1f}%")

print("\n=== is ngen predictable from the prompt inside gsm8k_or_other? ===")
m = fams == "gsm8k_or_other"
y = (ngen[m, 0] == 4).astype(int)
sub = np.where(m)[0]
print(f"  n={m.sum()} P(ngen=4)={y.mean():.3f}")
# simple text features
feat = []
for i in sub:
    t = texts[i]
    feat.append([len(t), t.count("$"), t.count("\\"), len(re.findall(r"\d", t)),
                 t.count("?"), t.count("\n"), len(re.findall(r"[A-Za-z]+", t)),
                 float("\\frac" in t or "\\sqrt" in t or "$" in t),
                 float(t.rstrip().endswith("?")),
                 len(re.findall(r"\b(?:[Ff]ind|[Cc]ompute|[Ll]et|[Pp]rove|[Dd]enote)\b", t))])
feat = np.array(feat, dtype=float)
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
pp = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15,
                                                      min_samples_leaf=20, random_state=0),
                       feat, y, cv=5, method="predict_proba")[:, 1]
print(f"  10 cheap features, 5-fold AUC = {roc_auc_score(y, pp):.4f}  "
      f"acc@0.5 = {((pp>0.5).astype(int)==y).mean():.4f}  base = {max(y.mean(),1-y.mean()):.4f}")
# does the '$' marker alone do it?
has_dollar = feat[:, 7] > 0
print(f"  math-marker rule: P(ngen=4 | marker)={y[has_dollar].mean():.3f} (n={has_dollar.sum()}), "
      f"P(ngen=4 | no marker)={y[~has_dollar].mean():.3f} (n={(~has_dollar).sum()})")
print(f"  score spread by ngen: ngen2 {np.round(score[sub][y==0].mean(0),3)}  "
      f"ngen4 {np.round(score[sub][y==1].mean(0),3)}")
print(f"  cost ratio by ngen (light): ngen2 {cost[sub][y==0,0].mean():.5f}  "
      f"ngen4 {cost[sub][y==1,0].mean():.5f}")
