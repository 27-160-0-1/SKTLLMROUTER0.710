# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b04 memory-heavy library.

Fast, exact numpy replica of `harness.Lab._knn_family` plus the memory-heavy
extensions it makes affordable:

  * exact (untruncated) index vectors instead of the shipped top-256 truncation
  * arbitrary k (the shipped runtime is k=16)
  * several similarity views (char 3/4/5-gram, word 1/2-gram, structural/dense)
  * per-family restricted indexes
  * seed-averaged meta heads

Nothing here writes to the repo outside `reports/lab/b04-*` caches.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from harness import Lab, DEPLOYED_EXP, _gbm_params, ORDINAL_THRESHOLDS, LUT_NODES, RANK_FLOOR_Q  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from ossp_router import similarity  # noqa: E402

CACHE = ROOT / "reports/lab/b04_counts.npz"


# ---------------------------------------------------------------- count blocks
def _fnv(gram: str) -> int:
    d = 14_695_981_039_346_656_037
    for b in gram.encode("utf-8"):
        d ^= b
        d = (d * 1_099_511_628_211) & ((1 << 64) - 1)
    return d


def char_counts(texts, bins=similarity.HASH_BINS, sizes=(3, 4, 5), limit=similarity.TEXT_LIMIT):
    """Exactly `similarity.hashed_counts` for every text, as a CSR matrix."""
    rows, cols, vals = [], [], []
    for ri, t in enumerate(texts):
        c = similarity.hashed_counts(t) if (bins == similarity.HASH_BINS and
                                            sizes == (3, 4, 5) and
                                            limit == similarity.TEXT_LIMIT) else _generic(t, bins, sizes, limit)
        rows.extend([ri] * len(c)); cols.extend(c.keys()); vals.extend(c.values())
    return sparse.csr_matrix((np.asarray(vals, float), (rows, cols)), shape=(len(texts), bins))


def _generic(text, bins, sizes, limit):
    norm = similarity._SPACE.sub(" ", text[:limit].casefold()).strip()
    p = f" {norm} "
    out = {}
    for s in sizes:
        for i in range(0, max(0, len(p) - s + 1)):
            k = _fnv(p[i:i + s]) & (bins - 1)
            out[k] = out.get(k, 0) + 1
    return out


_WORD = None


def word_counts(texts, bins=1 << 15, ngrams=(1, 2), limit=8_000):
    import re
    global _WORD
    if _WORD is None:
        _WORD = re.compile(r"[0-9a-zÀ-ɏ가-힣]+")
    rows, cols, vals = [], [], []
    for ri, t in enumerate(texts):
        toks = _WORD.findall(t[:limit].casefold())
        out = {}
        for n in ngrams:
            for i in range(0, max(0, len(toks) - n + 1)):
                k = _fnv(" ".join(toks[i:i + n])) & (bins - 1)
                out[k] = out.get(k, 0) + 1
        rows.extend([ri] * len(out)); cols.extend(out.keys()); vals.extend(out.values())
    return sparse.csr_matrix((np.asarray(vals, float), (rows, cols)), shape=(len(texts), bins))


def load_counts(lab):
    """Cached char + word count matrices for all 2,640 episodes."""
    if CACHE.exists():
        z = np.load(CACHE)
        if int(z["n"]) == lab.n:
            C = sparse.csr_matrix((z["cd"], z["ci"], z["cp"]), shape=tuple(z["cs"]))
            Wd = sparse.csr_matrix((z["wd"], z["wi"], z["wp"]), shape=tuple(z["ws"]))
            return C, Wd
    t0 = time.perf_counter()
    C = char_counts(lab.texts)
    Wd = word_counts(lab.texts)
    np.savez_compressed(CACHE, n=lab.n, cd=C.data, ci=C.indices, cp=C.indptr,
                        cs=np.array(C.shape), wd=Wd.data, wi=Wd.indices, wp=Wd.indptr,
                        ws=np.array(Wd.shape))
    print(f"[b04] counts built in {time.perf_counter()-t0:.0f}s", flush=True)
    return C, Wd


# ------------------------------------------------------------------- tf-idf view
def _rownorm(M):
    n = np.sqrt(M.multiply(M).sum(axis=1)).A.ravel()
    n[n == 0] = 1.0
    return sparse.diags(1.0 / n) @ M


def tfidf_view(C, fit_idx, top_components=256):
    """Return (Q, V) - query vectors for every row, index vectors for fit rows.

    Exactly reproduces similarity.tfidf_vector: idf from the fit rows only,
    sublinear tf, index vectors truncated to `top_components` *before* the L2
    normalisation, query vectors untruncated.
    """
    F = C[fit_idx]
    df = np.asarray((F > 0).sum(axis=0)).ravel()
    N = len(fit_idx)
    idf = np.zeros(C.shape[1])
    m = df > 0
    idf[m] = np.log((1.0 + N) / (1.0 + df[m])) + 1.0
    TF = C.copy()
    TF.data = 1.0 + np.log(TF.data)
    W = (TF @ sparse.diags(idf)).tocsr()
    W.eliminate_zeros()
    W.sort_indices()                      # tie-break in tfidf_vector is (-value, index)
    Q = _rownorm(W)
    Vi = W[fit_idx].tocsr()
    Vi.sort_indices()
    if top_components:
        Vi = _truncate(Vi, top_components)
    V = _rownorm(Vi)
    return Q.tocsr(), V.tocsr()


def _truncate(M, k):
    M = M.tocsr()
    keep_d, keep_i, keep_p = [], [], [0]
    for r in range(M.shape[0]):
        a, b = M.indptr[r], M.indptr[r + 1]
        d = M.data[a:b]; ix = M.indices[a:b]
        if len(d) > k:
            # sorted by (-value, index): stable argsort on -d over index-ascending data
            o = np.argsort(-d, kind="stable")[:k]
            d = d[o]; ix = ix[o]
        keep_d.append(d); keep_i.append(ix); keep_p.append(keep_p[-1] + len(d))
    return sparse.csr_matrix((np.concatenate(keep_d), np.concatenate(keep_i),
                              np.asarray(keep_p)), shape=M.shape)


def knn_rows(Q, V, fit_idx, rows, targets, k, self_pos=None, extra_stats=False):
    """kNN prediction for `rows` against index `V` (built over `fit_idx`).

    self_pos: for rows that are themselves in fit_idx, their position in fit_idx
    (excluded from the neighbour list), else -1.
    Returns (m, T+1) = weighted target row + top-1 similarity, matching harness.
    """
    S = (Q[rows] @ V.T).toarray()
    if self_pos is not None:
        sp = np.asarray(self_pos)
        ok = sp >= 0
        S[np.arange(len(rows))[ok], sp[ok]] = -np.inf
    tg = targets[fit_idx]
    gmean = tg.mean(axis=0)
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]
    sim = np.take_along_axis(S, order, axis=1)
    sim = np.where(np.isfinite(sim) & (sim > 0), sim, 0.0)
    tot = sim.sum(axis=1)
    wgt = np.where(tot[:, None] > 0, sim / np.maximum(tot[:, None], 1e-300), 0.0)
    out = np.einsum("ij,ijk->ik", wgt, tg[order])
    dead = tot <= 1e-9
    out[dead] = gmean
    top1 = np.where(dead, 0.0, sim[:, 0])
    res = [out, top1[:, None]]
    if extra_stats:
        res.append(np.column_stack([sim.mean(axis=1), sim[:, min(k, sim.shape[1]) - 1],
                                    tg[order].std(axis=1).mean(axis=1)]))
    return np.hstack(res)


# --------------------------------------------------------------- the Lab subclass
class MemLab(Lab):
    """Lab with a vectorised kNN stage and pluggable memory-heavy extensions.

    knn_spec keys (all optional):
      k            : neighbours for the primary (deployed-shape) view, default 16
      top_comp     : index truncation, default 256 (0 = exact vectors)
      views        : list of extra view dicts appended as meta features only, each
                     {'kind': 'char'|'word', 'k': int, 'top_comp': int,
                      'per_family': bool, 'stats': bool}
      seeds        : tuple of GBM random_states to average over (default (11,))
    """

    def __init__(self, knn_spec=None, **kw):
        super().__init__(**kw)
        self.C, self.Wd = load_counts(self)
        self.spec = dict(k=16, top_comp=256, views=(), seeds=(11,))
        if knn_spec:
            self.spec.update(knn_spec)
        self._pos = None

    def _knn_family(self, fit_idx, hold_idx, targets=None):
        tg = self.targets if targets is None else targets
        fit_idx = np.asarray(fit_idx); hold_idx = np.asarray(hold_idx)
        pos = -np.ones(self.n, dtype=int)
        pos[fit_idx] = np.arange(len(fit_idx))
        sp_fit = pos[fit_idx]
        sp_hold = pos[hold_idx]  # -1 for a genuine hold-out row

        Q, V = tfidf_view(self.C, fit_idx, self.spec["top_comp"])
        k = self.spec["k"]
        kf = knn_rows(Q, V, fit_idx, fit_idx, tg, k, sp_fit)
        kh = knn_rows(Q, V, fit_idx, hold_idx, tg, k, sp_hold)

        for v in self.spec["views"]:
            base = self.C if v.get("kind", "char") == "char" else self.Wd
            Qv, Vv = tfidf_view(base, fit_idx, v.get("top_comp", 256))
            kk = v.get("k", 16)
            st = v.get("stats", False)
            if v.get("per_family", False):
                af = self._per_family(Qv, Vv, fit_idx, fit_idx, tg, kk, sp_fit)
                ah = self._per_family(Qv, Vv, fit_idx, hold_idx, tg, kk, sp_hold)
            else:
                af = knn_rows(Qv, Vv, fit_idx, fit_idx, tg, kk, sp_fit, extra_stats=st)
                ah = knn_rows(Qv, Vv, fit_idx, hold_idx, tg, kk, sp_hold, extra_stats=st)
            kf = np.hstack([kf, af]); kh = np.hstack([kh, ah])

        from collections import defaultdict
        by = defaultdict(list)
        for i in fit_idx:
            by[self.fam_names[i]].append(tg[i])
        fg = tg[fit_idx].mean(axis=0)
        fam_mean = {nm: (np.mean(by[nm], axis=0) if len(by.get(nm, [])) >= 8 else fg)
                    for nm in self.FAMILIES}
        fam_hold = np.array([fam_mean[self.fam_names[i]] for i in hold_idx])
        fam_fit = np.array([fam_mean[self.fam_names[i]] for i in fit_idx])
        return kf, kh, fam_fit, fam_hold

    def _per_family(self, Q, V, fit_idx, rows, tg, k, self_pos):
        """Neighbours restricted to the query's own regex family."""
        out = np.zeros((len(rows), tg.shape[1] + 1))
        famfit = self.fam_arr[fit_idx]
        for nm in self.FAMILIES:
            sel = np.where(famfit == nm)[0]
            qr = np.where(self.fam_arr[rows] == nm)[0]
            if len(qr) == 0:
                continue
            if len(sel) < 4:
                out[qr] = np.concatenate([tg[fit_idx].mean(axis=0), [0.0]])
                continue
            sp = np.asarray(self_pos)[qr]
            remap = -np.ones(len(fit_idx), dtype=int)
            remap[sel] = np.arange(len(sel))
            sp2 = np.where(sp >= 0, remap[np.maximum(sp, 0)], -1)
            out[qr] = knn_rows(Q, V[sel], fit_idx[sel], np.asarray(rows)[qr], tg,
                               min(k, len(sel) - 1), sp2)
        return out

    # ------------------------------------------------------------ seed ensemble
    def fit_predict(self, fit_idx, hold_idx, exp=None, targets=None, sample_weight=None):
        seeds = self.spec.get("seeds", (11,))
        if len(seeds) == 1 and seeds[0] == 11:
            return super().fit_predict(fit_idx, hold_idx, exp, targets, sample_weight)
        return self._fit_predict_ens(fit_idx, hold_idx, exp, targets, sample_weight, seeds)

    def _fit_predict_ens(self, fit_idx, hold_idx, exp, targets, sample_weight, seeds):
        """Identical to Lab.fit_predict except every GBM head is averaged over
        `seeds` random_states (the shared stages - ridge, legacy, kNN - are fitted
        once, so the only thing that changes is the tree-ensemble fitting noise)."""
        exp = dict(DEPLOYED_EXP, **(exp or {}))
        tg = self.targets if targets is None else targets
        dt = np.column_stack([tg[:, 1] - tg[:, 0], tg[:, 2] - tg[:, 1]])
        knn_fit, knn_hold, fam_fit, fam_hold = self._knn_family(fit_idx, hold_idx, tg)
        if exp.get("legacy_refit", True):
            head = self.fit_legacy(fit_idx, exp.get("legacy_alpha", 100.0))
            leg_fit = self.predict_legacy(head, fit_idx)
            leg_hold = self.predict_legacy(head, hold_idx)
            if exp.get("legacy_oof_meta", False):
                inner_l = np.random.default_rng(7).integers(0, 5, size=len(fit_idx))
                leg_fit = np.zeros_like(leg_fit)
                for kk in range(5):
                    h = self.fit_legacy(fit_idx[inner_l != kk], exp.get("legacy_alpha", 100.0))
                    leg_fit[inner_l == kk] = self.predict_legacy(h, fit_idx[inner_l == kk])
        else:
            leg_fit, leg_hold = self.legacy[fit_idx], self.legacy[hold_idx]
        ridge = Ridge(alpha=exp["ridge_alpha"], solver="sparse_cg")
        ridge.fit(self.X[fit_idx], tg[fit_idx])
        lin_hold = ridge.predict(self.X[hold_idx])
        lin_hold[:, :3] = np.clip(lin_hold[:, :3], 0.0, 1.0)
        inner = np.random.default_rng(0).integers(0, 5, size=len(fit_idx))
        oof = np.zeros((len(fit_idx), tg.shape[1]))
        for kk in range(5):
            mm = Ridge(alpha=exp["ridge_alpha"], solver="sparse_cg")
            mm.fit(self.X[fit_idx[inner != kk]], tg[fit_idx[inner != kk]])
            oof[inner == kk] = mm.predict(self.X[fit_idx[inner == kk]])
        oof[:, :3] = np.clip(oof[:, :3], 0.0, 1.0)
        blocks_fit = [self.dense[fit_idx], self.fam_onehot[fit_idx], leg_fit, oof, knn_fit]
        blocks_hold = [self.dense[hold_idx], self.fam_onehot[hold_idx], leg_hold, lin_hold, knn_hold]
        if self._extra is not None:
            blocks_fit.append(self._extra[fit_idx]); blocks_hold.append(self._extra[hold_idx])
        Xf = np.hstack(blocks_fit); Xh = np.hstack(blocks_hold)
        sw = None if sample_weight is None else np.asarray(sample_weight)[fit_idx]

        def gp(seed):
            p = _gbm_params(exp); p["random_state"] = seed; return p

        nh = len(hold_idx)
        meta = np.zeros((nh, 6)); gain = np.zeros((nh, 2))
        for s in seeds:
            for kk in range(6):
                meta[:, kk] += HistGradientBoostingRegressor(**gp(s)).fit(
                    Xf, tg[fit_idx, kk], sample_weight=sw).predict(Xh) / len(seeds)
            for kk in range(2):
                gain[:, kk] += HistGradientBoostingRegressor(**gp(s)).fit(
                    Xf, dt[fit_idx, kk], sample_weight=sw).predict(Xh) / len(seeds)
        if exp.get("ordinal", True):
            ordsc = np.zeros((nh, 3))
            for mi in range(3):
                cum = np.zeros(nh)
                for th in ORDINAL_THRESHOLDS:
                    y = (tg[fit_idx, mi] >= th).astype(int)
                    if y.min() == y.max():
                        cum += float(y.min()); continue
                    for s in seeds:
                        raw = HistGradientBoostingClassifier(**gp(s)).fit(
                            Xf, y, sample_weight=sw).decision_function(Xh)
                        cum += (1.0 / (1.0 + np.exp(-np.clip(raw, -50, 50)))) / len(seeds)
                ordsc[:, mi] = cum / len(ORDINAL_THRESHOLDS)
            meta[:, :3] = ordsc
        rank_eff = np.zeros((nh, 2)); floors = np.zeros(2)
        if exp.get("rank", True):
            grid = np.linspace(0.0, 1.0, LUT_NODES)
            tc = np.exp(tg[:, 3:6])
            for g, (a, b) in enumerate([(0, 1), (1, 2)]):
                ds = tg[:, b] - tg[:, a]; dc = tc[:, b] - tc[:, a]
                fl = max(float(np.quantile(dc[fit_idx], RANK_FLOOR_Q)), 1e-9)
                eff = ds / np.maximum(dc, fl)
                r = rankdata(eff[fit_idx], method="average") / max(len(fit_idx) - 1, 1)
                q = np.quantile(eff[fit_idx], grid)
                pr = np.zeros(nh)
                for s in seeds:
                    pr += HistGradientBoostingRegressor(**gp(s)).fit(
                        Xf, r, sample_weight=sw).predict(Xh) / len(seeds)
                rank_eff[:, g] = np.interp(np.clip(pr, 0.0, 1.0), grid, q)
                floors[g] = fl
        return dict(idx=np.asarray(hold_idx), lin=lin_hold, legacy=leg_hold, fam=fam_hold,
                    knn=knn_hold, meta=meta, gain=gain, rank_eff=rank_eff, floors=floors)
