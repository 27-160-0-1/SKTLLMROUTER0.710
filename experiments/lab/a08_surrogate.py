# SPDX-License-Identifier: Apache-2.0
"""a08 cheap surrogate for testing PREPROCESSING variants.

Structure (a deliberately reduced copy of the deployed stack -- no legacy
artifact, no kNN -- so that changes to the text pipeline are not diluted):

    text --(variant)--> [dense30 | word-hash 8192 | char-hash 8192]
        -> Ridge(alpha) -> 6 outputs (3 score, 3 log cost)
        -> HistGB meta on [dense30, fam9, linear6] -> 6 regression + 2 gain heads
        -> gain reconstruction (alpha .5) -> blend with linear path per tier
        -> Lagrangian allocation (labdata) -> bootstrap EV over 880-row resamples

Everything is 5-fold OOF over the combined 2,640 rows (fold seed 123, same as
experiments/e43_joint_sweep.py).  Variants only change the hashed blocks.

Usage:  python a08_surrogate.py VARIANT [VARIANT ...]
"""
from __future__ import annotations

import math
import re
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labdata import load_all, MODEL_IDS  # noqa: E402
from ossp_router import learned_router, similarity  # noqa: E402
from ossp_router.protocol import load_input  # noqa: E402

WORD_BINS = 8192
CHAR_BINS = 8192
RIDGE_ALPHA = 10.0
GBM = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=30,
           l2_regularization=3.0, early_stopping=True, validation_fraction=0.15,
           random_state=11)
GAIN_ALPHA = 0.5
BLEND = {"fast": 0.6, "balanced": 0.45, "premium": 0.3}
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
WEIGHT = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
GRIDS = {"fast": np.arange(0.92, 1.041, 0.01),
         "balanced": np.arange(0.80, 1.001, 0.01),
         "premium": np.arange(0.78, 1.001, 0.01)}

_TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")

_H = {}


def _hb(s: str, bins: int):
    v = _H.get(s)
    if v is None:
        d = zlib.crc32(s.encode("utf-8"))
        v = (d & 0x7FFFFFFF, -1.0 if d & 0x80000000 else 1.0)
        _H[s] = v
    return v[0] & (bins - 1), v[1]


def _norm_tokens(text):
    out = []
    for t in _TOKEN.findall(text):
        t = t.casefold()
        out.append("<number>" if t.isdecimal() else t)
    return out


def _char_text(text, limit=6000):
    n = _SPACE.sub(" ", _DIGITS.sub("0", text.casefold())).strip()
    if len(n) > limit:
        h = limit // 2
        n = n[:h] + " … " + n[-h:]
    return n


def _block(values, bins):
    """signed hashed COUNT row (unnormalised); returns dict bin -> signed count."""
    counts = {}
    for value, cnt in Counter(values).items():
        idx, sgn = _hb(value, bins)
        counts[idx] = counts.get(idx, 0.0) + sgn * float(cnt)
    return {k: v for k, v in counts.items() if v}


def build_blocks(texts, word_limit=None, char_limit=6000, char_stride=3):
    """Return (word_counts_csr, char_counts_csr) of SIGNED counts."""
    wr, wc, wv, cr, cc, cv = [], [], [], [], [], []
    for i, text in enumerate(texts):
        wtext = text
        if word_limit is not None and len(text) > word_limit:
            h = word_limit // 2
            wtext = text[:h] + " … " + text[-h:]
        toks = _norm_tokens(wtext)
        vals = [f"w1:{t}" for t in toks]
        vals += [f"w2:{a}\x1f{b}" for a, b in zip(toks, toks[1:])]
        for k, v in _block(vals, WORD_BINS).items():
            wr.append(i); wc.append(k); wv.append(v)
        ct = _char_text(text, char_limit)
        cvals = [f"c{s}:{ct[j:j+s]}"
                 for s in (3, 4, 5)
                 for j in range(0, max(0, len(ct) - s + 1), char_stride)]
        for k, v in _block(cvals, CHAR_BINS).items():
            cr.append(i); cc.append(k); cv.append(v)
    n = len(texts)
    W = sparse.csr_matrix((wv, (wr, wc)), shape=(n, WORD_BINS))
    C = sparse.csr_matrix((cv, (cr, cc)), shape=(n, CHAR_BINS))
    return W, C


def weight_block(M, mode, fit_idx):
    """Apply a term weighting + row normalisation.  Fold-pure: df from fit_idx."""
    M = M.copy().astype(np.float64)
    d = M.data
    if mode in ("sublinear", "sublinear_idf"):
        d[:] = np.sign(d) * (1.0 + np.log(np.abs(d)))
    if mode in ("idf", "sublinear_idf"):
        sub = M[fit_idx]
        df = np.asarray((sub != 0).sum(axis=0)).ravel()
        idf = np.log((1.0 + len(fit_idx)) / (1.0 + df)) + 1.0
        M = M.multiply(sparse.csr_matrix(idf[None, :])).tocsr()
        d = M.data
    if mode == "sqrtlen":
        pass
    # row normalisation
    if mode == "sqrtlen":
        s = np.asarray(np.abs(M).sum(axis=1)).ravel()
        scale = 1.0 / np.sqrt(np.maximum(s, 1e-12))
    else:
        s = np.asarray(M.multiply(M).sum(axis=1)).ravel()
        scale = 1.0 / np.sqrt(np.maximum(s, 1e-12))
    scale[s <= 0] = 0.0
    return sparse.diags(scale) @ M


# ------------------------------------------------------------------ data
def load_everything():
    tr, dv = load_all()
    inputs = load_input(ROOT / "data/combined/inputs.json")
    episodes = list(inputs.episodes)
    ids = tr.episode_ids + dv.episode_ids
    assert [e.episode_id for e in episodes] == ids, "combined order mismatch"
    texts = tr.texts + dv.texts
    true_s = np.vstack([tr.score, dv.score])
    true_c = np.vstack([tr.cost, dv.cost])
    dense = np.asarray([learned_router.raw_dense_features(e) for e in episodes], dtype=np.float64)
    fam = [similarity.classify_family(t) for t in texts]
    FAM = list(similarity.FAMILY_NAMES)
    fam1h = np.zeros((len(texts), len(FAM)))
    for i, f in enumerate(fam):
        fam1h[i, FAM.index(f)] = 1.0
    return texts, dense, fam, fam1h, true_s, true_c, len(tr)


def run_variant(name, texts, dense, fam1h, true_s, true_c, blocks, wmode):
    n = len(texts)
    targets = np.hstack([true_s, np.log(true_c)])
    dmean, dstd = dense.mean(0), dense.std(0)
    dstd = np.where(dstd > 1e-6, dstd, 1.0)
    Z = (dense - dmean) / dstd
    W, C = blocks
    rng = np.random.default_rng(123)
    fold_of = rng.integers(0, 5, size=n)
    linear = np.zeros((n, 6))
    meta = np.zeros((n, 8))
    gain_t = np.column_stack([targets[:, 1] - targets[:, 0], targets[:, 2] - targets[:, 1]])
    full_t = np.hstack([targets, gain_t])
    for fold in range(5):
        hold = fold_of == fold
        fit = np.where(~hold)[0]
        hi = np.where(hold)[0]
        Wf = weight_block(W, wmode, fit)
        Cf = weight_block(C, wmode, fit)
        X = sparse.hstack([sparse.csr_matrix(Z), Wf, Cf], format="csr")
        r = Ridge(alpha=RIDGE_ALPHA, solver="sparse_cg").fit(X[fit], targets[fit])
        lh = r.predict(X[hi])
        lh[:, :3] = np.clip(lh[:, :3], 0, 1)
        linear[hi] = lh
        inner = np.random.default_rng(fold).integers(0, 5, size=len(fit))
        ioof = np.zeros((len(fit), 6))
        for k in range(5):
            a = fit[inner != k]
            b = inner == k
            m = Ridge(alpha=RIDGE_ALPHA, solver="sparse_cg").fit(X[a], targets[a])
            ioof[b] = m.predict(X[fit[b]])
        ioof[:, :3] = np.clip(ioof[:, :3], 0, 1)
        Xf = np.hstack([Z[fit], fam1h[fit], ioof])
        Xh = np.hstack([Z[hi], fam1h[hi], lh])
        for h in range(8):
            g = HistGradientBoostingRegressor(**GBM).fit(Xf, full_t[fit, h])
            meta[hi, h] = g.predict(Xh)
    return linear, meta


def evaluate(linear, meta, true_s, true_c, seed=7, nboot=200):
    m = meta[:, :6].copy()
    recon = np.column_stack([m[:, 0], m[:, 0] + meta[:, 6], m[:, 0] + meta[:, 6] + meta[:, 7]])
    m[:, :3] = (1 - GAIN_ALPHA) * m[:, :3] + GAIN_ALPHA * recon
    n = len(true_s)
    r = np.random.default_rng(seed)
    samples = [r.integers(0, n, size=880) for _ in range(nboot)]
    total, detail = 0.0, {}
    for tier in ("fast", "balanced", "premium"):
        b = BLEND[tier]
        st = (1 - b) * linear + b * m
        ps = np.clip(st[:, :3], 0, 1)
        pc = np.exp(np.clip(st[:, 3:], -50, 50))
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        best = None
        for s in GRIDS[tier]:
            evs = []
            for smp in samples:
                p = _alloc(ps[smp], pc[smp], MULTS[tier], float(s))
                ar = np.arange(len(smp))
                ratio = true_c[smp][ar, p].sum() / true_c[smp][:, 0].sum()
                evs.append(0.0 if ratio > MULTS[tier] else true_s[smp][ar, p].mean())
            ev = float(np.mean(evs))
            if best is None or ev > best[0]:
                best = (ev, float(s))
        total += WEIGHT[tier] * best[0]
        detail[tier] = best
    return total, detail


def _alloc(ps, pc, mult, safety):
    lt = pc[:, 0].sum()
    cap = lt * max(1.0, mult * safety)
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


VARIANTS = {
    #  name           word_limit  char_limit  stride  weighting
    "v0_base":        (None, 6000, 3, "count"),
    "v1_sublinear":   (None, 6000, 3, "sublinear"),
    "v2_idf":         (None, 6000, 3, "idf"),
    "v3_sub_idf":     (None, 6000, 3, "sublinear_idf"),
    "v4_wordlimit6k": (6000, 6000, 3, "count"),
    "v5_wordlimit2k": (2000, 6000, 3, "count"),
    "v6_sqrtlen":     (None, 6000, 3, "sqrtlen"),
    "v7_charlimit24k": (None, 24000, 12, "count"),
}


def main():
    want = sys.argv[1:] or list(VARIANTS)
    t0 = time.perf_counter()
    texts, dense, fam, fam1h, true_s, true_c, ntr = load_everything()
    print(f"[a08] loaded {len(texts)} rows in {time.perf_counter()-t0:.0f}s", flush=True)
    cache = {}
    for name in want:
        wl, cl, st, wm = VARIANTS[name]
        key = (wl, cl, st)
        if key not in cache:
            tb = time.perf_counter()
            cache[key] = build_blocks(texts, wl, cl, st)
            print(f"[a08] blocks {key} in {time.perf_counter()-tb:.0f}s", flush=True)
        tb = time.perf_counter()
        lin, meta = run_variant(name, texts, dense, fam1h, true_s, true_c, cache[key], wm)
        ev, det = evaluate(lin, meta, true_s, true_c)
        rm = [np.sqrt(((lin[:, j] - true_s[:, j]) ** 2).mean()) for j in range(3)]
        rc = [np.sqrt(((lin[:, 3 + j] - np.log(true_c[:, j])) ** 2).mean()) for j in range(3)]
        cs = [np.corrcoef(meta[:, j], true_s[:, j])[0, 1] for j in range(3)]
        print(f"[a08] {name:16s} EV={ev:.4f}  " +
              " ".join(f"{t[:4]}={v[0]:.4f}@{v[1]:.2f}" for t, v in det.items()) +
              f"  ridge sRMSE={np.round(rm,4)} cRMSE={np.round(rc,3)}"
              f"  meta corr={np.round(cs,3)}  [{time.perf_counter()-tb:.0f}s]", flush=True)
        np.savez(ROOT / f"reports/lab/a08_{name}.npz", linear=lin, meta=meta)


if __name__ == "__main__":
    main()
