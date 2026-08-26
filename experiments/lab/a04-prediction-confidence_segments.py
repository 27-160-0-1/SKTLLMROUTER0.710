# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a04 / step 2 -- where does confidence collapse?

Builds the artifact's own OOD signal (kNN top-1 cosine similarity of each dev
prompt against the *train* index) and segments the deployed E43 dev prediction
error by it, and by family / length / language / code.

Also asks the decision-relevant question: does low top-1 similarity predict a
*worse allocation*, not just a worse RMSE?
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result, MODEL_IDS
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import similarity

CACHE = Path(__file__).resolve().parents[0] / "a04_knn_cache.npz"

tr, dv = load_split("train"), load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv)
IDX = np.arange(n)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}

# ---------------------------------------------------------------- kNN top-1
if CACHE.exists():
    Z = np.load(CACHE)
    top1, knn_row, top1_gap = Z["top1"], Z["knn_row"], Z["top1_gap"]
else:
    t0 = time.perf_counter()
    freqs, total = similarity.document_frequencies(tr.texts)
    idf = similarity.idf_table(freqs, total)
    vecs = [similarity.tfidf_vector(t, idf, top_components=similarity.TOP_COMPONENTS) for t in tr.texts]
    targets = np.hstack([tr.score, np.log(tr.cost)])
    index = similarity.KnnIndex(vecs, targets.tolist())
    top1 = np.zeros(n); top1_gap = np.zeros(n); knn_row = np.zeros((n, 6))
    for i, t in enumerate(dv.texts):
        q = similarity.tfidf_vector(t, idf)
        if not q:
            continue
        scores = {}
        for g, v in q.items():
            for d, s in index.postings.get(g, ()):
                scores[d] = scores.get(d, 0.0) + v * s
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:similarity.NEIGHBORS]
        tot = sum(s for _d, s in ranked)
        row = np.zeros(6)
        for d, s in ranked:
            row += (s / tot) * targets[d]
        knn_row[i] = row
        top1[i] = ranked[0][1]
        top1_gap[i] = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
    np.savez_compressed(CACHE, top1=top1, knn_row=knn_row, top1_gap=top1_gap)
    print(f"[knn] built in {time.perf_counter()-t0:.0f}s -> {CACHE}")

# ---------------------------------------------------------------- covariates
fam = np.array([similarity.classify_family(t) for t in dv.texts])
L = np.array([len(t) for t in dv.texts], dtype=float)
hangul = np.array([sum(1 for c in t if "가" <= c <= "힣") / max(len(t), 1) for t in dv.texts])
has_code = np.array([("def " in t) or ("assert " in t) or ("```" in t) for t in dv.texts])

err = np.abs(P["score_fast"] - dv.score)                # (n,3)
sq = (P["score_fast"] - dv.score) ** 2
gain_err = np.abs((P["score_fast"][:, 1] - P["score_fast"][:, 0]) - (dv.score[:, 1] - dv.score[:, 0]))
gain2_err = np.abs((P["score_fast"][:, 2] - P["score_fast"][:, 1]) - (dv.score[:, 2] - dv.score[:, 1]))
logc_err = np.abs(np.log(P["cost_premium"]) - np.log(dv.cost))

print("=" * 104)
print("STEP 2a  kNN top-1 similarity (dev vs train index) -- the artifact's own OOD signal")
print("=" * 104)
print(f"  top1 percentiles 1/5/10/25/50/75/90/99: "
      + " ".join(f"{q:.3f}" for q in np.percentile(top1, [1, 5, 10, 25, 50, 75, 90, 99])))
qs = np.quantile(top1, np.linspace(0, 1, 6))
qs[0] -= 1e-9
bins = np.digitize(top1, qs[1:-1])
print(f"\n{'quintile':10s} {'n':>4s} {'top1 range':>16s} | "
      f"{'|err| L/M/K':>22s} | {'RMSE L/M/K':>22s} | {'|gain err| md/km':>17s} | {'|logc err| K':>12s}")
for b in range(5):
    m = bins == b
    print(f"  Q{b+1:<7d} {m.sum():4d} {top1[m].min():7.3f}-{top1[m].max():.3f} | "
          + " ".join(f"{err[m, j].mean():7.4f}" for j in range(3)) + " | "
          + " ".join(f"{np.sqrt(sq[m, j].mean()):7.4f}" for j in range(3)) + " | "
          f"{gain_err[m].mean():8.4f} {gain2_err[m].mean():8.4f} | {logc_err[m,2].mean():12.3f}")

print("\n  correlations with top1 (Spearman-ish via rank):")
def sp(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]
for lbl, v in (("|err| light", err[:, 0]), ("|err| mid", err[:, 1]), ("|err| k1", err[:, 2]),
               ("|gain err| mid-light", gain_err), ("|gain err| k1-mid", gain2_err),
               ("|logcost err| k1", logc_err[:, 2]), ("|logcost err| light", logc_err[:, 0])):
    print(f"    {lbl:22s} rho(top1) = {sp(top1, v):+.3f}   rho(len) = {sp(L, v):+.3f}")

print()
print("=" * 104)
print("STEP 2b  segments: family / length / language / code")
print("=" * 104)
def seg_table(name, groups):
    print(f"\n  -- by {name}")
    print(f"    {'group':18s} {'n':>4s} {'top1':>6s} | {'|err|L':>7s} {'|err|M':>7s} {'|err|K':>7s} | "
          f"{'corrL':>6s} {'corrM':>6s} {'corrK':>6s} | {'bias L':>7s} {'bias M':>7s} {'bias K':>7s}")
    for g, m in groups:
        if m.sum() < 8:
            continue
        cors = []
        for j in range(3):
            a, b = P["score_fast"][m, j], dv.score[m, j]
            cors.append(np.corrcoef(a, b)[0, 1] if a.std() > 1e-9 and b.std() > 1e-9 else np.nan)
        bias = (P["score_fast"][m] - dv.score[m]).mean(0)
        print(f"    {g:18s} {m.sum():4d} {top1[m].mean():6.3f} | "
              + " ".join(f"{err[m, j].mean():7.4f}" for j in range(3)) + " | "
              + " ".join(f"{c:6.3f}" for c in cors) + " | "
              + " ".join(f"{b:+7.4f}" for b in bias))

seg_table("family", [(f, fam == f) for f in sorted(set(fam))])
lq = np.quantile(L, [0, .2, .4, .6, .8, 1.0]); lq[0] -= 1
lb = np.digitize(L, lq[1:-1])
seg_table("length quintile", [(f"Q{b+1} <={int(lq[b+1])}", lb == b) for b in range(5)])
seg_table("language", [("korean>20%", hangul > 0.2), ("korean<=20%", hangul <= 0.2)])
seg_table("code", [("has code", has_code), ("no code", ~has_code)])

print()
print("=" * 104)
print("STEP 2c  does low top-1 similarity hurt the DECISION (not just the RMSE)?")
print("  regret_i = true score of the ex-post best affordable choice minus realised, per quintile")
print("=" * 104)
for t in TIERS:
    ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
    r = tier_result(ps, pc, dv, t, SAFE[t])
    sel = r["sel"]
    # ex-post: what the allocator would have picked with true scores at the same lambda
    rt = tier_result(dv.score, pc, dv, t, SAFE[t])
    sel_t = rt["sel"]
    reg = dv.score[IDX, sel_t] - dv.score[IDX, sel]
    print(f"  {t:9s} overall regret {reg.mean():+.4f}  (score {r['score']:.4f} -> {rt['score']:.4f})")
    for b in range(5):
        m = bins == b
        print(f"      top1 Q{b+1}: n={m.sum():4d} regret={reg[m].mean():+.4f} "
              f"mismatch={np.mean(sel[m]!=sel_t[m]):.3f} sel_mean={sel[m].mean():.2f}/{sel_t[m].mean():.2f}")

print()
print("=" * 104)
print("STEP 2d  abstention test: force the LOWEST-confidence items to light (or to the")
print("  family-modal choice) and see whether the tier score improves")
print("=" * 104)
for t in TIERS:
    ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
    base = tier_result(ps, pc, dv, t, SAFE[t])
    print(f"  {t:9s} base score={base['score']:.4f} ratio={base['ratio']:.4f}")
    for frac in (0.05, 0.10, 0.20, 0.40):
        k = int(frac * n)
        low = np.argsort(top1)[:k]
        ps2 = ps.copy()
        ps2[low] = ps[low].mean(1, keepdims=True)      # no opinion -> flat score, allocator falls to cheapest
        r = tier_result(ps2, pc, dv, t, SAFE[t])
        print(f"      abstain lowest-{frac:.0%} top1 (n={k:3d}): score={r['score']:.4f} "
              f"({r['score']-base['score']:+.4f}) ratio={r['ratio']:.4f} pass={r['passed']}")
