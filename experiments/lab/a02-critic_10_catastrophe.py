# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a02-critic #10.  Seven episodes decide the premium tier.  Who are they, is the
event predictable, and what is a veto worth?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, TIER_MULT, tier_result
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
tr, dv = load_split("train"), load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
famd = np.array([classify_family(t) for t in dv.texts])
famt = np.array([classify_family(t) for t in tr.texts])
N = len(dv)
L = dv.cost[:, 0].sum()

print("=" * 100)
print("A.  The catastrophic-cost episodes")
print("=" * 100)
ck = dv.cost[:, 2]
o = np.argsort(-ck)
print(f"  {'episode':10s} {'family':10s} {'in_tok':>7s} {'out_tok':>8s} {'ngen':>4s} "
      f"{'k1 cost':>8s} {'% of 4L':>8s} {'pred cost':>9s} {'ratio':>7s} {'k1 score':>8s} {'light':>6s}")
for i in o[:10]:
    print(f"  {dv.episode_ids[i]:10s} {famd[i]:10s} {dv.itok[i,2]:7.0f} {dv.otok[i,2]:8.0f} "
          f"{dv.ngen[i,2]:4.0f} {ck[i]:8.3f} {ck[i]/(4*L)*100:7.2f}% {P['cost_premium'][i,2]:9.3f} "
          f"{ck[i]/P['cost_premium'][i,2]:7.1f}x {dv.score[i,2]:8.2f} {dv.score[i,0]:6.2f}")

print("\n  same, on TRAIN (is the phenomenon reproducible across splits?)")
ckt = tr.cost[:, 2]
Lt = tr.cost[:, 0].sum()
ot = np.argsort(-ckt)
for i in ot[:6]:
    print(f"  {tr.episode_ids[i]:10s} {famt[i]:10s} {tr.itok[i,2]:7.0f} {tr.otok[i,2]:8.0f} "
          f"{tr.ngen[i,2]:4.0f} {ckt[i]:8.3f} {ckt[i]/(4*Lt)*100:7.2f}%")

print("\n  output-token distribution per generation (k1):")
for sp, f_ in ((tr, famt), (dv, famd)):
    x = sp.otok[:, 2] / sp.ngen[:, 2]
    print(f"    {sp.name}: median {np.median(x):.0f}  p90 {np.percentile(x,90):.0f}  "
          f"p99 {np.percentile(x,99):.0f}  max {x.max():.0f}   "
          f"n(>=16000)={int((x>=16000).sum())}  n(>=8000)={int((x>=8000).sum())}")
print("\n  which families produce >=8000 output tokens/gen on k1?")
for sp, f_ in ((tr, famt), (dv, famd)):
    x = sp.otok[:, 2] / sp.ngen[:, 2]
    d = {}
    for fm in sorted(set(f_)):
        m = f_ == fm
        d[fm] = (int((x[m] >= 8000).sum()), int(m.sum()))
    print(f"    {sp.name}: " + "  ".join(f"{k}={v[0]}/{v[1]}" for k, v in d.items() if v[0] > 0))

print()
print("=" * 100)
print("B.  What is a veto on the catastrophic tail worth?  (premium tier)")
print("=" * 100)
SG = np.arange(0.70, 1.601, 0.005)


def prem(ban_mask, label):
    C = P["cost_premium"].copy()
    C[ban_mask, 2] = 1e9
    best = None
    for sf in SG:
        r = tier_result(P["score_premium"], C, dv, "premium", float(sf))
        if r["passed"] and (best is None or r["score"] > best[0]):
            best = (r["score"], float(sf), r["ratio"], int((r["sel"] == 2).sum()))
    # also the deployed-safety number
    r0 = tier_result(P["score_premium"], C, dv, "premium", SAFE["premium"])
    rr = np.array([tier_result(P["score_premium"], C, dv, "premium", float(sf))["ratio"] for sf in SG])
    print(f"  {label:44s} banned={int(ban_mask.sum()):3d}  best={best[0]:.4f}@sf{best[1]:.3f} "
          f"r{best[2]:.3f} k1={best[3]:3d} | at deployed sf .88: {r0['score']:.4f} r{r0['ratio']:.3f}"
          f" | max ratio jump {np.abs(np.diff(rr)).max():.3f}")
    return best[0]


base = prem(np.zeros(N, bool), "no veto (deployed)")
for k in (2, 5, 7, 10, 20):
    prem(np.isin(np.arange(N), o[:k]), f"ORACLE veto of the {k} costliest k1 episodes")

print("\n  realistic vetoes that use only prompt-visible information:")
ntok_proxy = dv.itok[:, 0] / dv.ngen[:, 0]
for q in (0.98, 0.95, 0.90):
    thr = np.quantile(ntok_proxy, q)
    prem(ntok_proxy > thr, f"veto k1 when input tokens > p{q*100:.0f} ({thr:.0f})")
for fm in ("longdoc", "hrmcr", "aime"):
    prem(famd == fm, f"veto k1 for family={fm}")
prem((famd == "longdoc") | (famd == "hrmcr"), "veto k1 for longdoc+hrmcr")

print()
print("=" * 100)
print("C.  Is 'k1 blows up' predictable from the prompt?  (train -> dev, 3 cheap features)")
print("=" * 100)
from sklearn.ensemble import HistGradientBoostingClassifier


def feats(sp, fm):
    x = sp.itok[:, 0] / sp.ngen[:, 0]
    L_ = np.array([len(t) for t in sp.texts], float)
    D = np.array([sum(ch.isdigit() for ch in t) for t in sp.texts], float)
    fams = ["aime", "belebele", "code", "dmmath", "gsm8k_or_other", "hrmcr", "longdoc",
            "ruletaker", "truthfulqa"]
    F = np.stack([(fm == f).astype(float) for f in fams], 1)
    return np.hstack([np.log(x + 1)[:, None], np.log(L_ + 1)[:, None], (D / (L_ + 1))[:, None], F])


Xtr, Xdv = feats(tr, famt), feats(dv, famd)
for thr_tok in (4000, 8000, 16000):
    ytr = (tr.otok[:, 2] / tr.ngen[:, 2] >= thr_tok).astype(int)
    ydv = (dv.otok[:, 2] / dv.ngen[:, 2] >= thr_tok).astype(int)
    if ytr.sum() < 10:
        print(f"  threshold {thr_tok}: only {ytr.sum()} positives in train, skipped")
        continue
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                         min_samples_leaf=20, random_state=0).fit(Xtr, ytr)
    p = clf.predict_proba(Xdv)[:, 1]
    from sklearn.metrics import roc_auc_score, average_precision_score
    order = np.argsort(-p)
    k = int(ydv.sum())
    print(f"  out_tok/gen >= {thr_tok:5d}: train pos {int(ytr.sum()):3d}, dev pos {k:3d}, "
          f"dev AUC={roc_auc_score(ydv,p):.3f} AP={average_precision_score(ydv,p):.3f}  "
          f"precision@{k}={ydv[order[:max(k,1)]].mean():.3f}  "
          f"recall@top50={ydv[order[:50]].sum()}/{k}")
    if thr_tok == 8000:
        np.save(Path(r"C:\Users\PJ05\AppData\Local\Temp\claude"
                     r"\C--portable-skt-LLM1-LLM-ROUTE-0-7000"
                     r"\377d7fd8-9983-4bec-bb08-163cc405f7a3\scratchpad\a02_blowup_p.npy"), p)
        for topk in (20, 50, 100):
            prem(np.isin(np.arange(N), order[:topk]),
                 f"veto k1 on the top-{topk} predicted blow-ups (train-fit)")
