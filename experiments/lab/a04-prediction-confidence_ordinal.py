# SPDX-License-Identifier: Apache-2.0
"""a04 / step 3 -- rebuild the 12 ordinal heads honestly (train -> dev) and
measure the predicted score DISTRIBUTION, its calibration and its monotonicity.

Pipeline (a faithful but self-contained copy of the deployed meta stack):
  features = dense30 + family9 + legacy6 + ridge-linear6 + knn7   (58 columns)
  ridge alpha = 10 (E43 adopted), fit on train, inner 5-fold OOF for the train rows
  kNN: char tf-idf index over train, leave-one-out for train rows
  12 HistGradientBoostingClassifier: P(s >= .25/.5/.75/1) per model
  E[s] = 0.25 * sum(P)

Outputs: per-model reliability tables, ECE, monotonicity violations, the
implied per-item variance, and whether that variance is usable.
Cache: a04_ordinal_cache.npz
"""
from __future__ import annotations
import sys, math, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import learned_router, legacy_hash_regex, similarity
from ossp_router.protocol import load_input

from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

CACHE = Path(__file__).resolve().parents[0] / "a04_ordinal_cache.npz"
THRESH = (0.25, 0.5, 0.75, 1.0)
GBM = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=30,
           l2_regularization=3.0, early_stopping=True, validation_fraction=0.15, random_state=11)

tr, dv = load_split("train"), load_split("dev")
ntr, ndv = len(tr), len(dv)

if CACHE.exists():
    Z = np.load(CACHE)
    Pdev, Ptr, Edev = Z["Pdev"], Z["Ptr"], Z["Edev"]
    print(f"[cache] loaded {CACHE}")
else:
    t0 = time.perf_counter()
    ep_tr = list(load_input(ROOT / "data/materialized/train/inputs.json").episodes)
    ep_dv = list(load_input(ROOT / "data/materialized/dev/inputs.json").episodes)
    artifact = learned_router.load_artifact(ROOT / "src/ossp_router/resources/learned-router.v1.json")
    legacy_art = legacy_hash_regex.load_artifact(ROOT / "src/ossp_router/resources/hash-regex-public.v1.json")

    def build(eps, texts):
        dense, legacy, fams = [], [], []
        rows, cols, vals = [], [], []
        for ri, e in enumerate(eps):
            d = learned_router.raw_dense_features(e)
            dense.append(d)
            it = learned_router.feature_items(
                e, word_hash_bins=artifact.word_hash_bins, char_hash_bins=artifact.char_hash_bins,
                dense_mean=artifact.dense_mean, dense_scale=artifact.dense_scale, raw_dense=d)
            for c, v in it.items():
                rows.append(ri); cols.append(c); vals.append(v)
            ls, lc = legacy_hash_regex.predict_episode(e, legacy_art)
            legacy.append([ls[m] for m in MODEL_IDS] + [math.log(lc[m]) for m in MODEL_IDS])
            fams.append(similarity.classify_family(texts[ri]))
        dim = len(learned_router.DENSE_FEATURE_NAMES) + artifact.word_hash_bins + artifact.char_hash_bins
        X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(eps), dim))
        return np.asarray(dense), np.asarray(legacy), np.array(fams), X

    D_tr, L_tr, F_tr, Xs_tr = build(ep_tr, tr.texts)
    D_dv, L_dv, F_dv, Xs_dv = build(ep_dv, dv.texts)
    print(f"[feat] {time.perf_counter()-t0:.0f}s", flush=True)

    FAMS = list(similarity.FAMILY_NAMES)
    OH_tr = np.stack([(F_tr == f).astype(float) for f in FAMS], 1)
    OH_dv = np.stack([(F_dv == f).astype(float) for f in FAMS], 1)

    tgt_tr = np.hstack([tr.score, np.log(tr.cost)])

    # ---- ridge linear head (alpha 10, E43) : dev from full train, train from inner 5-fold OOF
    ridge = Ridge(alpha=10.0, solver="sparse_cg").fit(Xs_tr, tgt_tr)
    lin_dv = ridge.predict(Xs_dv); lin_dv[:, :3] = np.clip(lin_dv[:, :3], 0, 1)
    rng = np.random.default_rng(123)
    inner = rng.integers(0, 5, ntr)
    lin_tr = np.zeros((ntr, 6))
    for k in range(5):
        m = inner == k
        r = Ridge(alpha=10.0, solver="sparse_cg").fit(Xs_tr[~m], tgt_tr[~m])
        lin_tr[m] = r.predict(Xs_tr[m])
    lin_tr[:, :3] = np.clip(lin_tr[:, :3], 0, 1)
    print(f"[ridge] {time.perf_counter()-t0:.0f}s", flush=True)

    # ---- kNN rows (7) : LOO for train, full index for dev
    freqs, total = similarity.document_frequencies(tr.texts)
    idf = similarity.idf_table(freqs, total)
    vecs = [similarity.tfidf_vector(t, idf, top_components=similarity.TOP_COMPONENTS) for t in tr.texts]
    knn = similarity.KnnIndex(vecs, tgt_tr.tolist())
    gmean = tgt_tr.mean(0)

    def kq(text, exclude=None):
        q = similarity.tfidf_vector(text, idf)
        if not q:
            return np.concatenate([gmean, [0.0]])
        sc = {}
        for g, v in q.items():
            for d, s in knn.postings.get(g, ()):
                if exclude is not None and d == exclude:
                    continue
                sc[d] = sc.get(d, 0.0) + v * s
        if not sc:
            return np.concatenate([gmean, [0.0]])
        rk = sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:similarity.NEIGHBORS]
        tot = sum(s for _d, s in rk)
        row = np.zeros(6)
        for d, s in rk:
            row += (s / tot) * tgt_tr[d]
        return np.concatenate([row, [rk[0][1]]])

    K_tr = np.array([kq(t, exclude=i) for i, t in enumerate(tr.texts)])
    K_dv = np.array([kq(t) for t in dv.texts])
    print(f"[knn] {time.perf_counter()-t0:.0f}s", flush=True)

    Xtr = np.hstack([D_tr, OH_tr, L_tr, lin_tr, K_tr])
    Xdv = np.hstack([D_dv, OH_dv, L_dv, lin_dv, K_dv])
    print("[X] shapes", Xtr.shape, Xdv.shape, flush=True)

    # ---- 12 ordinal heads ; dev from full train fit ; train from 5-fold cross-fit
    outer = np.random.default_rng(7).integers(0, 5, ntr)
    Pdev = np.zeros((ndv, 3, 4)); Ptr = np.zeros((ntr, 3, 4))
    for j in range(3):
        for ti, th in enumerate(THRESH):
            y = (tr.score[:, j] >= th - 1e-9).astype(int)
            if y.min() == y.max():
                Pdev[:, j, ti] = float(y.mean()); Ptr[:, j, ti] = float(y.mean()); continue
            m = HistGradientBoostingClassifier(**GBM).fit(Xtr, y)
            Pdev[:, j, ti] = m.predict_proba(Xdv)[:, 1]
            for k in range(5):
                hold = outer == k
                if y[~hold].min() == y[~hold].max():
                    Ptr[hold, j, ti] = float(y[~hold].mean()); continue
                mm = HistGradientBoostingClassifier(**GBM).fit(Xtr[~hold], y[~hold])
                Ptr[hold, j, ti] = mm.predict_proba(Xtr[hold])[:, 1]
        print(f"[gbm] model {MODEL_IDS[j]} done {time.perf_counter()-t0:.0f}s", flush=True)
    Edev = 0.25 * Pdev.sum(2)
    np.savez_compressed(CACHE, Pdev=Pdev, Ptr=Ptr, Edev=Edev)
    print(f"[done] {time.perf_counter()-t0:.0f}s -> {CACHE}")

Etr = 0.25 * Ptr.sum(2)
DEP = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
fam = np.array([similarity.classify_family(t) for t in dv.texts])

print("=" * 100)
print("STEP 3a  sanity: my rebuilt ordinal E[s] vs the deployed E43 predictions (dev)")
print("=" * 100)
for j, m in enumerate(MODEL_IDS):
    a, b, ts = Edev[:, j], DEP["score_fast"][:, j], dv.score[:, j]
    print(f"  {m:11s} mine corr={np.corrcoef(a,ts)[0,1]:.3f} rmse={np.sqrt(((a-ts)**2).mean()):.4f} "
          f"mean={a.mean():.4f} | deployed corr={np.corrcoef(b,ts)[0,1]:.3f} "
          f"rmse={np.sqrt(((b-ts)**2).mean()):.4f} mean={b.mean():.4f} | corr(mine,dep)={np.corrcoef(a,b)[0,1]:.3f}")

print()
print("=" * 100)
print("STEP 3b  monotonicity of the 4 heads in the threshold (should be non-increasing)")
print("=" * 100)
for j, m in enumerate(MODEL_IDS):
    p = Pdev[:, j, :]
    v = np.diff(p, axis=1) > 0
    print(f"  {m:11s} mean P = " + " ".join(f"{p[:,t].mean():.3f}" for t in range(4))
          + f" | rows with >=1 inversion: {np.mean(v.any(1))*100:5.1f}%"
          + f" | max inversion {np.max(np.diff(p,axis=1)):.3f}"
          + f" | mean total inversion mass {np.clip(np.diff(p,axis=1),0,None).sum(1).mean():.4f}")

print()
print("=" * 100)
print("STEP 3c  calibration of each head (dev).  ECE = 10-bin equal-count expected calib. error")
print("=" * 100)
def ece(p, y, nb=10):
    qs = np.quantile(p, np.linspace(0, 1, nb + 1)); qs[0] -= 1e-9
    b = np.digitize(p, qs[1:-1])
    e = 0.0; rows = []
    for k in range(nb):
        m = b == k
        if m.sum() == 0:
            continue
        e += m.mean() * abs(p[m].mean() - y[m].mean())
        rows.append((m.sum(), p[m].mean(), y[m].mean()))
    return e, rows

for j, m in enumerate(MODEL_IDS):
    print(f"\n  -- {m}")
    for ti, th in enumerate(THRESH):
        y = (dv.score[:, j] >= th - 1e-9).astype(float)
        p = Pdev[:, j, ti]
        e, rows = ece(p, y)
        br = ((p - y) ** 2).mean()
        base = y.mean() * (1 - y.mean())
        print(f"     P(s>={th:.2f}) base_rate={y.mean():.3f} pred_mean={p.mean():.3f} "
              f"ECE={e:.4f} Brier={br:.4f} skill={1-br/max(base,1e-9):+.3f} "
              f"AUC-ish corr={np.corrcoef(p,y)[0,1]:.3f}")
        print("        deciles pred->obs: " + " ".join(f"{a:.2f}->{b:.2f}" for _n, a, b in rows))

print()
print("=" * 100)
print("STEP 3d  predicted distribution -> per-item variance; is it a usable confidence?")
print("=" * 100)
# distribution on the grid {0,.25,.5,.75,1} from the (monotonised) survival probs
Pm = np.minimum.accumulate(np.clip(Pdev, 0, 1), axis=2)
surv = np.concatenate([np.ones((ndv, 3, 1)), Pm], axis=2)          # P(s>=0), ..., P(s>=1)
mass = -np.diff(surv, axis=2)                                       # P(s in [0,.25)), ...
mass = np.concatenate([mass, Pm[:, :, 3:4]], axis=2)                # + P(s=1)
grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
mass = np.clip(mass, 0, None); mass /= mass.sum(2, keepdims=True)
mu = (mass * grid).sum(2)
var = (mass * (grid ** 2)).sum(2) - mu ** 2
print(f"  E[s] from mass vs 0.25*sumP: max abs diff {np.abs(mu-Edev).max():.4f}")
for j, m in enumerate(MODEL_IDS):
    print(f"  {m:11s} var mean={var[:,j].mean():.4f} p10={np.percentile(var[:,j],10):.4f} "
          f"p90={np.percentile(var[:,j],90):.4f}")
    qs = np.quantile(var[:, j], np.linspace(0, 1, 6)); qs[0] -= 1e-9
    b = np.digitize(var[:, j], qs[1:-1])
    for k in range(5):
        mm = b == k
        e = np.abs(Edev[mm, j] - dv.score[mm, j])
        print(f"      var Q{k+1}: n={mm.sum():4d} var={var[mm,j].mean():.4f} |err|={e.mean():.4f} "
              f"rmse={np.sqrt(((Edev[mm,j]-dv.score[mm,j])**2).mean()):.4f} "
              f"corr={np.corrcoef(Edev[mm,j],dv.score[mm,j])[0,1]:+.3f}")

print()
print("=" * 100)
print("STEP 3e  selection bias check conditioned on VARIANCE (E30 was conditioned on the")
print("  predicted gain only).  bias = mean(true delta) - mean(pred delta) per (dhat decile x var half)")
print("=" * 100)
for (a_, b_, lab) in ((0, 1, "mid-light"), (1, 2, "k1-mid")):
    dhat = Edev[:, b_] - Edev[:, a_]
    dtru = dv.score[:, b_] - dv.score[:, a_]
    vv = var[:, a_] + var[:, b_]
    hi = vv > np.median(vv)
    print(f"\n  {lab}: corr(dhat,dtrue)={np.corrcoef(dhat,dtru)[0,1]:.3f}  "
          f"lo-var corr={np.corrcoef(dhat[~hi],dtru[~hi])[0,1]:.3f}  hi-var corr={np.corrcoef(dhat[hi],dtru[hi])[0,1]:.3f}")
    for name, msk in (("lo-var", ~hi), ("hi-var", hi)):
        qs = np.quantile(dhat[msk], np.linspace(0, 1, 6)); qs[0] -= 1e-9
        bb = np.digitize(dhat, qs[1:-1])
        out = []
        for k in range(5):
            mm = msk & (bb == k)
            if mm.sum() < 5:
                continue
            out.append(f"Q{k+1}(n{mm.sum()}) dhat={dhat[mm].mean():+.3f} dtrue={dtru[mm].mean():+.3f} "
                       f"bias={dtru[mm].mean()-dhat[mm].mean():+.3f}")
        print(f"    {name}: " + " | ".join(out))

print()
print("=" * 100)
print("STEP 3f  what does my rebuilt E[s] score if plugged into the deployed cost model?")
print("=" * 100)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
for label, S in (("deployed score", None), ("mine (ordinal only)", Edev),
                 ("0.5*dep + 0.5*mine", None)):
    tot = 0.0; parts = []
    for t in TIERS:
        ss = DEP[f"score_{t}"] if S is None and label.startswith("deployed") else (
            Edev if S is not None else 0.5 * DEP[f"score_{t}"] + 0.5 * Edev)
        r = tier_result(ss, DEP[f"cost_{t}"], dv, t, SAFE[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else ' BUST'}")
    print(f"  {label:22s} final={tot:.4f}  " + "  ".join(parts))
