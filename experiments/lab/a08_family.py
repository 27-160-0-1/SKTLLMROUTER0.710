# SPDX-License-Identifier: Apache-2.0
"""a08: full-stack test of the REPAIRED family classifier.

The sparse ridge path, the legacy rows and the kNN rows do not depend on the
family label, so they are computed ONCE and cached; only the meta GBM (which
consumes the 9-dim family one-hot) and the family-mean blend are re-run per
variant.  Harness semantics follow experiments/e43_joint_sweep.py (5-fold,
fold seed 123, 880xNBOOT bootstrap EV, E43 cand0 constants), minus the rank
heads (dropped for runtime; identical for every variant).

Usage: python a08_family.py [NBOOT] [SEEDS...]
"""
from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labdata import load_all  # noqa: E402
from ossp_router import learned_router, legacy_hash_regex, similarity  # noqa: E402
from ossp_router.protocol import MODEL_IDS, load_input  # noqa: E402
import re  # noqa: E402

# --- repaired family classifier (audit in a08_probe9.py) -------------------
_RT_Q = re.compile(r"\nQuestion: ")
_RT_FACT = re.compile(r"\b\w+ is (?:not )?\w+\.")
_DM_EXTRA = re.compile(
    r"^(?:Work out|Which is|What is|Add|Subtract|Total of|Product of|Divide|Multiply|"
    r"Calculate|Simplify|Solve|Evaluate|Round|Sort|Put|Let |Suppose|Differentiate|"
    r"Factor|Expand|In base|Convert|How many|What comes next|List the prime|"
    r"Is \d|Find |Give |Print |Sum |Take |Subtract|\-?\d[\d ,.eE/*+-]* (?:divided by|times|plus|minus))")
_DM_OP = re.compile(r"^-?[\d./]+ (?:divided by|times|plus|minus) -?[\d./]+\.?$")
_LATEX = re.compile(r"\\(?:frac|sqrt|sum|int|cdot|left|right|text|angle|triangle|overline|log|pi|"
                    r"binom|mathbb|dfrac|le|ge|neq|equiv|pmod)")


def classify3(text: str) -> str:
    """classify2 + the aime/$money fix (LaTeX test instead of any two '$')."""
    head = text[:600]
    if similarity._CODE.search(head):
        return "code"
    if similarity._HRMCR_AGE.search(head) or similarity._HRMCR_CAL.search(head[:200]):
        return "hrmcr"
    if similarity._TRUTHFULQA.match(text):
        return "truthfulqa"
    if _RT_Q.search(text) and len(_RT_FACT.findall(text)) >= 3:
        return "ruletaker"
    if similarity._RULETAKER.search(head) and " is " in head:
        return "ruletaker"
    if sum("가" <= ch <= "힣" for ch in head) > 40:
        return "belebele"
    if len(text) > 6_000:
        return "longdoc"
    if _LATEX.search(text) and len(text) < 2_000:
        return "aime"
    body = text.strip()
    if len(body) < 400 and (_DM_EXTRA.match(body) or _DM_OP.match(body)):
        return "dmmath"
    if similarity._DMMATH.match(head) and len(text) < 400:
        return "dmmath"
    return "gsm8k_or_other"


def classify2(text: str) -> str:
    head = text[:600]
    if similarity._CODE.search(head):
        return "code"
    if similarity._HRMCR_AGE.search(head) or similarity._HRMCR_CAL.search(head[:200]):
        return "hrmcr"
    if similarity._TRUTHFULQA.match(text):
        return "truthfulqa"
    if _RT_Q.search(text) and len(_RT_FACT.findall(text)) >= 3:
        return "ruletaker"
    if similarity._RULETAKER.search(head) and " is " in head:
        return "ruletaker"
    if sum("가" <= ch <= "힣" for ch in head) > 40:
        return "belebele"
    if len(text) > 6_000:
        return "longdoc"
    if similarity._AIME.search(head) and len(text) < 2_000:
        return "aime"
    body = text.strip()
    if len(body) < 400 and (_DM_EXTRA.match(body) or _DM_OP.match(body)):
        return "dmmath"
    if similarity._DMMATH.match(head) and len(text) < 400:
        return "dmmath"
    return "gsm8k_or_other"

CACHE = ROOT / "reports/lab/a08_cache.npz"
RIDGE_ALPHA = 10.0
GBM = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=30,
           l2_regularization=3.0, early_stopping=True, validation_fraction=0.15,
           random_state=11)
CFG = dict(legacy_w=0.9, fam_w=0.15, conf_scale=0.25, gain_alpha=0.5,
           blend_fast=0.6, blend_balanced=0.45, blend_premium=0.3)
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
WEIGHT = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
GRIDS = {"fast": np.arange(0.92, 1.001, 0.01),
         "balanced": np.arange(0.82, 0.941, 0.01),
         "premium": np.arange(0.78, 0.931, 0.01)}
FAMILIES = list(similarity.FAMILY_NAMES)

tr, dv = load_all()
texts = tr.texts + dv.texts
n = len(texts)
ntr = len(tr)
true_s = np.vstack([tr.score, dv.score])
true_c = np.vstack([tr.cost, dv.cost])
targets = np.hstack([true_s, np.log(true_c)])
full_t = np.hstack([targets,
                    np.column_stack([targets[:, 1] - targets[:, 0],
                                     targets[:, 2] - targets[:, 1]])])
rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)


def build_cache():
    t0 = time.perf_counter()
    inputs = load_input(ROOT / "data/combined/inputs.json")
    episodes = list(inputs.episodes)
    assert [e.episode_id for e in episodes] == tr.episode_ids + dv.episode_ids
    art = learned_router.load_artifact(ROOT / "src/ossp_router/resources/learned-router.v1.json")
    leg = legacy_hash_regex.load_artifact(ROOT / "src/ossp_router/resources/hash-regex-public.v1.json")
    dense, legacy = [], []
    r_, c_, v_ = [], [], []
    for i, ep in enumerate(episodes):
        d = learned_router.raw_dense_features(ep)
        dense.append(d)
        for c, v in learned_router.feature_items(
                ep, word_hash_bins=art.word_hash_bins, char_hash_bins=art.char_hash_bins,
                dense_mean=art.dense_mean, dense_scale=art.dense_scale, raw_dense=d).items():
            r_.append(i); c_.append(c); v_.append(v)
        ls, lc = legacy_hash_regex.predict_episode(ep, leg)
        legacy.append([ls[m] for m in MODEL_IDS] + [math.log(lc[m]) for m in MODEL_IDS])
    dim = len(learned_router.DENSE_FEATURE_NAMES) + art.word_hash_bins + art.char_hash_bins
    X = sparse.csr_matrix((v_, (r_, c_)), shape=(n, dim))
    dense = np.asarray(dense)
    legacy = np.asarray(legacy)
    print(f"[a08] features {time.perf_counter()-t0:.0f}s", flush=True)

    linear = np.zeros((n, 6))
    inner_all = np.zeros((n, 6))
    for fold in range(5):
        hold = fold_of == fold
        fit = np.where(~hold)[0]; hi = np.where(hold)[0]
        r = Ridge(alpha=RIDGE_ALPHA, solver="sparse_cg").fit(X[fit], targets[fit])
        lh = r.predict(X[hi]); lh[:, :3] = np.clip(lh[:, :3], 0, 1)
        linear[hi] = lh
        inner = np.random.default_rng(fold).integers(0, 5, size=len(fit))
        for k in range(5):
            a = fit[inner != k]; b = inner == k
            m = Ridge(alpha=RIDGE_ALPHA, solver="sparse_cg").fit(X[a], targets[a])
            inner_all[fit[b]] = m.predict(X[fit[b]])
        print(f"[a08]  ridge fold {fold} {time.perf_counter()-t0:.0f}s", flush=True)
    inner_all[:, :3] = np.clip(inner_all[:, :3], 0, 1)

    # kNN rows (text-only, family-independent)
    knn_hold = np.zeros((n, 7))
    knn_fit = {}
    for fold in range(5):
        hold = fold_of == fold
        fit = np.where(~hold)[0]; hi = np.where(hold)[0]
        ktexts = [texts[i] for i in fit]
        freqs, tot = similarity.document_frequencies(ktexts)
        idf = similarity.idf_table(freqs, tot)
        vecs = [similarity.tfidf_vector(t, idf, top_components=similarity.TOP_COMPONENTS)
                for t in ktexts]
        post = defaultdict(list)
        for d, vec in enumerate(vecs):
            for gidx, val in vec.items():
                post[gidx].append((d, val))
        gmean = targets[fit].mean(axis=0)

        def kq(text, exclude=None):
            q = similarity.tfidf_vector(text, idf)
            sc = {}
            for gidx, val in q.items():
                for d, st in post.get(gidx, ()):
                    if exclude is not None and d == exclude:
                        continue
                    sc[d] = sc.get(d, 0.0) + val * st
            if not sc:
                return np.concatenate([gmean, [0.0]])
            rk = sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:similarity.NEIGHBORS]
            s = sum(v for _d, v in rk)
            row = np.zeros(6)
            for d, v in rk:
                row += (v / s) * targets[fit[d]]
            return np.concatenate([row, [rk[0][1]]])

        knn_fit[fold] = np.array([kq(texts[i], exclude=k) for k, i in enumerate(fit)])
        knn_hold[hi] = np.array([kq(texts[i]) for i in hi])
        print(f"[a08]  knn fold {fold} {time.perf_counter()-t0:.0f}s", flush=True)
    np.savez_compressed(CACHE, dense=dense, legacy=legacy, linear=linear,
                        inner=inner_all, knn_hold=knn_hold,
                        **{f"knn_fit{k}": v for k, v in knn_fit.items()})
    print(f"[a08] cache written {time.perf_counter()-t0:.0f}s", flush=True)


if not CACHE.exists():
    build_cache()
Z = np.load(CACHE)
dense, legacy, linear, inner = Z["dense"], Z["legacy"], Z["linear"], Z["inner"]
knn_hold = Z["knn_hold"]
knn_fit = {k: Z[f"knn_fit{k}"] for k in range(5)}


def one_hot(names):
    M = np.zeros((n, len(FAMILIES)))
    for i, name in enumerate(names):
        M[i, FAMILIES.index(name)] = 1.0
    return M


def fam_means(names):
    out = np.zeros((n, 6))
    for fold in range(5):
        hold = fold_of == fold
        fit = np.where(~hold)[0]; hi = np.where(hold)[0]
        by = defaultdict(list)
        for i in fit:
            by[names[i]].append(targets[i])
        gl = targets[fit].mean(axis=0)
        mm = {f: (np.mean(by[f], axis=0) if len(by.get(f, [])) >= 8 else gl) for f in FAMILIES}
        out[hi] = np.array([mm[names[i]] for i in hi])
    return out


def run_meta(f1h):
    meta = np.zeros((n, 8))
    for fold in range(5):
        hold = fold_of == fold
        fit = np.where(~hold)[0]; hi = np.where(hold)[0]
        Xf = np.hstack([dense[fit], f1h[fit], legacy[fit], inner[fit], knn_fit[fold]])
        Xh = np.hstack([dense[hi], f1h[hi], legacy[hi], linear[hi], knn_hold[hi]])
        for h in range(8):
            g = HistGradientBoostingRegressor(**GBM).fit(Xf, full_t[fit, h])
            meta[hi, h] = g.predict(Xh)
    return meta


def alloc(ps, pc, mult, safety):
    lt = pc[:, 0].sum(); cap = lt * max(1.0, mult * safety)
    tb = np.array([2e-12, 1e-12, 0.0])

    def choose(pen):
        pick = np.argmax(ps - pen * pc / lt + tb, axis=1)
        return pick, pc[np.arange(len(pick)), pick].sum()

    pick, tot = choose(0.0)
    if tot > cap:
        lo, hi = 0.0, 1.0
        pick, tot = choose(hi)
        while tot > cap and hi < 2 ** 60:
            lo, hi = hi, hi * 2
            pick, tot = choose(hi)
        for _ in range(40):
            mid = (lo + hi) / 2
            c2, t2 = choose(mid)
            if t2 <= cap:
                hi, pick, tot = mid, c2, t2
            else:
                lo = mid
    if tot > cap:
        pick = np.zeros(len(ps), dtype=int)
    return pick


def evaluate(meta, fm, seed, nboot):
    prod = CFG["legacy_w"] * legacy + (1 - CFG["legacy_w"]) * linear
    prod = (1 - CFG["fam_w"]) * prod + CFG["fam_w"] * fm
    conf = np.clip(knn_hold[:, 6], 0, 1)[:, None] * CFG["conf_scale"]
    prod = (1 - conf) * prod + conf * knn_hold[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0, 1)
    m = meta[:, :6].copy()
    recon = np.column_stack([m[:, 0], m[:, 0] + meta[:, 6], m[:, 0] + meta[:, 6] + meta[:, 7]])
    m[:, :3] = (1 - CFG["gain_alpha"]) * m[:, :3] + CFG["gain_alpha"] * recon
    r = np.random.default_rng(seed)
    samples = [r.integers(0, n, size=880) for _ in range(nboot)]
    tot = 0.0
    det = {}
    for tier in ("fast", "balanced", "premium"):
        b = CFG[f"blend_{tier}"]
        st = (1 - b) * prod + b * m
        ps = np.clip(st[:, :3], 0, 1)
        pc = np.exp(np.clip(st[:, 3:], -50, 50))
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        best = None
        for s in GRIDS[tier]:
            evs = []
            for smp in samples:
                p = alloc(ps[smp], pc[smp], MULTS[tier], float(s))
                ar = np.arange(len(smp))
                ratio = true_c[smp][ar, p].sum() / true_c[smp][:, 0].sum()
                evs.append(0.0 if ratio > MULTS[tier] else true_s[smp][ar, p].mean())
            ev = float(np.mean(evs))
            if best is None or ev > best[0]:
                best = (ev, float(s))
        tot += WEIGHT[tier] * best[0]
        det[tier] = best
    return tot, det


def main():
    nboot = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    allv = {
        "old_regex": lambda: [similarity.classify_family(t) for t in texts],
        "rt_dm": lambda: [classify2(t) for t in texts],
        "rt_dm_aime": lambda: [classify3(t) for t in texts],
    }
    pick = [v for v in sys.argv[2:] if v in allv] if any(v in allv for v in sys.argv[2:]) else list(allv)
    seeds = [int(x) for x in sys.argv[2:] if x.isdigit()] or [7, 17, 23]
    variants = {k: allv[k]() for k in pick}
    metas = {}
    for name, names in variants.items():
        t0 = time.perf_counter()
        f1h = one_hot(names)
        fm = fam_means(names)
        meta = run_meta(f1h)
        metas[name] = (meta, fm)
        rs = [np.sqrt(((meta[:, j] - true_s[:, j]) ** 2).mean()) for j in range(3)]
        rc = [np.sqrt(((meta[:, 3 + j] - np.log(true_c[:, j])) ** 2).mean()) for j in range(3)]
        print(f"[a08] {name:10s} meta OOF sRMSE={np.round(rs,4)} cRMSE={np.round(rc,4)} "
              f"[{time.perf_counter()-t0:.0f}s]", flush=True)
    for name, (meta, fm) in metas.items():
        evs = []
        for sd in seeds:
            ev, det = evaluate(meta, fm, sd, nboot)
            evs.append(ev)
            print(f"[a08] {name:10s} seed{sd:3d} EV={ev:.4f}  " +
                  " ".join(f"{t[:4]}={v[0]:.4f}@{v[1]:.2f}" for t, v in det.items()), flush=True)
        print(f"[a08] {name:10s} MEAN EV={np.mean(evs):.4f}", flush=True)


if __name__ == "__main__":
    main()
