# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Fast in-memory replica of the deployed training/prediction chain.

Reproduces `train_learned_router_gpu.py` -> `build_router_augmentation.py` ->
`build_meta_gbm.py` -> `learned_router.predict_episode_augmented` ->
`select_models` closely enough to iterate in ~60 s instead of ~225 s, and
without touching any artifact on disk.

Two evaluation modes:
  * ``holdout``  fit on train (1,760), score dev (880) exactly -> the honest metric
  * ``cv``       5-fold nested CV over any index set + 880-item bootstrap EV

Everything is keyed off a ``cfg`` dict of the post-hoc constants and an ``exp``
dict of the expensive constants, so experiments can override any stage.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ossp_router import learned_router, legacy_hash_regex, similarity  # noqa: E402
from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes  # noqa: E402

TIERS = ("fast", "balanced", "premium")
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
ORDINAL_THRESHOLDS = (0.25, 0.5, 0.75, 1.0)
LUT_NODES = 65
RANK_FLOOR_Q = 0.05

DEPLOYED_CFG = dict(legacy_w=0.9, fam_w=0.15, conf_scale=0.25, gain_alpha=0.5, rank_beta=0.4,
                    blend_fast=0.6, blend_balanced=0.45, blend_premium=0.3)
DEPLOYED_EXP = dict(ridge_alpha=10.0, gbm_lr=0.06, gbm_leaves=15, gbm_min_leaf=30, gbm_l2=3.0,
                    gbm_iter=300, ordinal=True, rank=True, legacy_refit=True, legacy_alpha=100.0)
DEPLOYED_SAFETY = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}

CACHE = ROOT / "reports/lab/features.npz"


def _gbm_params(exp, seed=None):
    return dict(max_iter=exp["gbm_iter"], learning_rate=exp["gbm_lr"],
                max_leaf_nodes=exp["gbm_leaves"], min_samples_leaf=exp["gbm_min_leaf"],
                l2_regularization=exp["gbm_l2"], early_stopping=True,
                validation_fraction=0.15,
                random_state=exp.get("gbm_seed", 11) if seed is None else seed)


def _fit_avg(cls, exp, Xf, y, Xh, sw, seeds, proba=False):
    """Average `seeds` refits of one head.

    b04 measured that a single meta fit is a large variance source: the fast
    tier's realised cost ratio has sd 0.0183 across refits and 2 of 10 single
    fits bust it, versus 0 of 5 seed-averaged ensembles.  Averaging is free at
    build time and does not change the runtime path (the exported trees simply
    become an average of ensembles)."""
    acc = None
    for s in seeds:
        m = cls(**_gbm_params(exp, s)).fit(Xf, y, sample_weight=sw)
        v = m.decision_function(Xh) if proba else m.predict(Xh)
        acc = v if acc is None else acc + v
    return acc / len(seeds)


class Lab:
    """Loads train(0..1759) + dev(1760..2639) once and caches the feature blocks."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        t0 = time.perf_counter()
        self.policy = load_bundled_policy()
        tr_in = load_input(ROOT / "data/materialized/train/inputs.json")
        dv_in = load_input(ROOT / "data/materialized/dev/inputs.json")
        tr_out = load_outcomes(ROOT / "data/train/outcomes.json")
        dv_out = load_outcomes(ROOT / "data/dev/outcomes.json")
        self.episodes = list(tr_in.episodes) + list(dv_in.episodes)
        self.n_train = len(tr_in.episodes)
        self.n = len(self.episodes)
        self.train_idx = np.arange(self.n_train)
        self.dev_idx = np.arange(self.n_train, self.n)
        index = {(o.episode_id, o.model_id): o for o in list(tr_out.outcomes) + list(dv_out.outcomes)}
        from decimal import Decimal
        unit = Decimal(self.policy.token_unit)

        def cost(eid, mid):
            o = index[(eid, mid)]
            r = self.policy.models[mid]
            return float(r.fixed_cost + Decimal(o.input_tokens) * r.input_token_rate / unit
                         + Decimal(o.output_tokens) * r.output_token_rate / unit)

        self.true_s = np.array([[float(index[(e.episode_id, m)].score) for m in MODEL_IDS]
                                for e in self.episodes])
        self.true_c = np.array([[cost(e.episode_id, m) for m in MODEL_IDS] for e in self.episodes])
        self.ngen = np.array([[index[(e.episode_id, m)].num_generations for m in MODEL_IDS]
                              for e in self.episodes], dtype=float)
        self.itok = np.array([[index[(e.episode_id, m)].input_tokens for m in MODEL_IDS]
                              for e in self.episodes], dtype=float)
        self.otok = np.array([[index[(e.episode_id, m)].output_tokens for m in MODEL_IDS]
                              for e in self.episodes], dtype=float)
        self.targets = np.hstack([self.true_s, np.log(self.true_c)])
        self.delta_targets = np.column_stack([self.targets[:, 1] - self.targets[:, 0],
                                              self.targets[:, 2] - self.targets[:, 1]])
        self.texts = [episode_text(e) for e in self.episodes]
        self._build_features()
        self._extra = None
        if verbose:
            print(f"[lab] ready n={self.n} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- features
    def _build_features(self):
        artifact = learned_router.load_artifact(ROOT / "src/ossp_router/resources/learned-router.v1.json")
        legacy_art = legacy_hash_regex.load_artifact(ROOT / "src/ossp_router/resources/hash-regex-public.v1.json")
        self.word_bins = artifact.word_hash_bins
        self.char_bins = artifact.char_hash_bins
        similarity.NEIGHBORS = 16
        if CACHE.exists():
            z = np.load(CACHE, allow_pickle=True)
            if int(z["n"]) == self.n:
                self.dense = z["dense"]; self.legacy = z["legacy"]
                self.fam_names = list(z["fam_names"])
                self.X = sparse.csr_matrix((z["sd"], z["si"], z["sp"]), shape=tuple(z["shape"]))
                self._finish_features()
                return
        dense_rows, legacy_rows, fam_names = [], [], []
        srows, scols, svals = [], [], []
        # dense standardisation must not depend on the fit set (the artifact's own
        # constants are used at runtime), so reuse the shipped ones like the tools do
        for ri, ep in enumerate(self.episodes):
            d = learned_router.raw_dense_features(ep)
            dense_rows.append(d)
            items = learned_router.feature_items(
                ep, word_hash_bins=self.word_bins, char_hash_bins=self.char_bins,
                dense_mean=artifact.dense_mean, dense_scale=artifact.dense_scale, raw_dense=d)
            for c, v in items.items():
                srows.append(ri); scols.append(c); svals.append(v)
            ls, lc = legacy_hash_regex.predict_episode(ep, legacy_art)
            legacy_rows.append([ls[m] for m in MODEL_IDS] + [math.log(lc[m]) for m in MODEL_IDS])
            fam_names.append(similarity.classify_family(self.texts[ri]))
        self.dense = np.asarray(dense_rows)
        self.legacy = np.asarray(legacy_rows)
        self.fam_names = fam_names
        dim = len(learned_router.DENSE_FEATURE_NAMES) + self.word_bins + self.char_bins
        self.X = sparse.csr_matrix((svals, (srows, scols)), shape=(self.n, dim))
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, n=self.n, dense=self.dense, legacy=self.legacy,
                            fam_names=np.array(fam_names), sd=self.X.data, si=self.X.indices,
                            sp=self.X.indptr, shape=np.array(self.X.shape))
        self._finish_features()

    def _legacy_raw_matrix(self):
        cache = ROOT / "reports/lab/legacy_raw.npy"
        if cache.exists():
            m = np.load(cache)
            if m.shape[0] == self.n:
                return m
        m = np.asarray([legacy_hash_regex.raw_feature_vector(e, 256) for e in self.episodes],
                       dtype=np.float64)
        np.save(cache, m)
        return m

    def fit_legacy(self, fit_idx, alpha=100.0):
        """Refit the official 256-bin hash-regex heads on `fit_idx` only.

        The shipped artifact was fitted on all 1,760 Train episodes, so using it
        as a meta feature makes every Train row in-sample and every Dev row
        out-of-sample.  That asymmetry makes CV optimistic and is what pushes the
        CV-chosen safety ratio past the point where Dev survives.  Refitting per
        fold removes it.  Reproduces baselines/train_hash_regex._fit_ridge.
        """
        M = self.legacy_raw[fit_idx]
        mean = M.mean(axis=0)
        scale = M.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        Z = (M - mean) / scale
        y = self.targets[fit_idx]
        inter = y.mean(axis=0)
        cen = y - inter
        rows, cols = Z.shape
        if rows <= cols:
            coef = Z.T @ np.linalg.solve(Z @ Z.T + alpha * np.eye(rows), cen)
        else:
            coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(cols), Z.T @ cen)
        return mean, scale, inter, coef

    def predict_legacy(self, head, idx):
        mean, scale, inter, coef = head
        row = (self.legacy_raw[idx] - mean) / scale @ coef + inter
        # same post-processing as legacy_hash_regex.predict_episode
        row[:, :3] = np.clip(row[:, :3], 0.0, 1.0)
        c = np.exp(np.clip(row[:, 3:6], -50.0, 50.0))
        c[:, 1] = np.maximum(c[:, 1], c[:, 0] * (1 + 1e-12))
        c[:, 2] = np.maximum(c[:, 2], c[:, 1] * (1 + 1e-12))
        row[:, 3:6] = np.log(c)
        return row

    def _finish_features(self):
        self.legacy_raw = self._legacy_raw_matrix()
        self.FAMILIES = list(similarity.FAMILY_NAMES)
        self.fam_onehot = np.zeros((self.n, len(self.FAMILIES)))
        for i, name in enumerate(self.fam_names):
            self.fam_onehot[i, self.FAMILIES.index(name)] = 1.0
        self.fam_arr = np.array(self.fam_names)

    def set_family(self, labels):
        """Override the source-family partition (see famrepair.classify_v3)."""
        self.fam_names = list(labels)
        self.fam_onehot = np.zeros((self.n, len(self.FAMILIES)))
        for i, name in enumerate(self.fam_names):
            self.fam_onehot[i, self.FAMILIES.index(name)] = 1.0
        self.fam_arr = np.array(self.fam_names)

    def set_extra_features(self, extra):
        """extra: (n, k) float array appended to the meta-GBM feature block, or None."""
        self._extra = None if extra is None else np.asarray(extra, dtype=float)

    # ------------------------------------------------------------ kNN / family
    def _knn_family(self, fit_idx, hold_idx, targets=None):
        tg = self.targets if targets is None else targets
        texts_fit = [self.texts[i] for i in fit_idx]
        freqs, total = similarity.document_frequencies(texts_fit)
        idf = similarity.idf_table(freqs, total)
        vecs = [similarity.tfidf_vector(t, idf, top_components=similarity.TOP_COMPONENTS) for t in texts_fit]
        index = similarity.KnnIndex(vecs, tg[fit_idx].tolist())
        gmean = tg[fit_idx].mean(axis=0)

        def query(text, exclude=None):
            q = similarity.tfidf_vector(text, idf)
            if not q:
                return np.concatenate([gmean, [0.0]])
            sc = {}
            for g, v in q.items():
                for d, s in index.postings.get(g, ()):
                    if exclude is not None and d == exclude:
                        continue
                    sc[d] = sc.get(d, 0.0) + v * s
            if not sc:
                return np.concatenate([gmean, [0.0]])
            ranked = sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:similarity.NEIGHBORS]
            tot = sum(s for _d, s in ranked)
            row = np.zeros(tg.shape[1])
            for d, s in ranked:
                row += (s / tot) * tg[fit_idx[d]]
            return np.concatenate([row, [ranked[0][1]]])

        knn_fit = np.array([query(self.texts[i], exclude=k) for k, i in enumerate(fit_idx)])
        knn_hold = np.array([query(self.texts[i]) for i in hold_idx])
        by_fam = defaultdict(list)
        for i in fit_idx:
            by_fam[self.fam_names[i]].append(tg[i])
        fglob = tg[fit_idx].mean(axis=0)
        fam_mean = {name: (np.mean(by_fam[name], axis=0) if len(by_fam.get(name, [])) >= 8 else fglob)
                    for name in self.FAMILIES}
        fam_hold = np.array([fam_mean[self.fam_names[i]] for i in hold_idx])
        fam_fit = np.array([fam_mean[self.fam_names[i]] for i in fit_idx])
        return knn_fit, knn_hold, fam_fit, fam_hold

    # ------------------------------------------------------------------- stage
    def fit_predict(self, fit_idx, hold_idx, exp=None, targets=None, sample_weight=None):
        """Train the whole chain on fit_idx and return everything needed to compose
        predictions for hold_idx."""
        exp = dict(DEPLOYED_EXP, **(exp or {}))
        tg = self.targets if targets is None else targets
        dt = np.column_stack([tg[:, 1] - tg[:, 0], tg[:, 2] - tg[:, 1]])
        gp = _gbm_params(exp)
        knn_fit, knn_hold, fam_fit, fam_hold = self._knn_family(fit_idx, hold_idx, tg)
        if exp.get("legacy_refit", True):
            head = self.fit_legacy(fit_idx, exp.get("legacy_alpha", 100.0))
            leg_fit = self.predict_legacy(head, fit_idx)
            leg_hold = self.predict_legacy(head, hold_idx)
            if exp.get("legacy_oof_meta", False):
                # The meta GBM must see the legacy feature at the quality it will
                # have at prediction time.  Fitted on all of fit_idx it is
                # in-sample (score corr 0.60 vs 0.39 out-of-sample), so the tree
                # ensemble learns to over-trust it.  Refit out-of-fold, exactly
                # like the ridge block already is.
                inner_l = np.random.default_rng(7).integers(0, 5, size=len(fit_idx))
                leg_fit = np.zeros_like(leg_fit)
                for k in range(5):
                    h = self.fit_legacy(fit_idx[inner_l != k], exp.get("legacy_alpha", 100.0))
                    leg_fit[inner_l == k] = self.predict_legacy(h, fit_idx[inner_l == k])
        else:
            leg_fit, leg_hold = self.legacy[fit_idx], self.legacy[hold_idx]

        ridge = Ridge(alpha=exp["ridge_alpha"], solver="sparse_cg")
        ridge.fit(self.X[fit_idx], tg[fit_idx])
        lin_hold = ridge.predict(self.X[hold_idx])
        lin_hold[:, :3] = np.clip(lin_hold[:, :3], 0.0, 1.0)

        inner = np.random.default_rng(0).integers(0, 5, size=len(fit_idx))
        oof = np.zeros((len(fit_idx), tg.shape[1]))
        for k in range(5):
            m = Ridge(alpha=exp["ridge_alpha"], solver="sparse_cg")
            m.fit(self.X[fit_idx[inner != k]], tg[fit_idx[inner != k]])
            oof[inner == k] = m.predict(self.X[fit_idx[inner == k]])
        oof[:, :3] = np.clip(oof[:, :3], 0.0, 1.0)

        blocks_fit = [self.dense[fit_idx], self.fam_onehot[fit_idx], leg_fit, oof, knn_fit]
        blocks_hold = [self.dense[hold_idx], self.fam_onehot[hold_idx], leg_hold,
                       lin_hold, knn_hold]
        if self._extra is not None:
            blocks_fit.append(self._extra[fit_idx]); blocks_hold.append(self._extra[hold_idx])
        Xf = np.hstack(blocks_fit)
        Xh = np.hstack(blocks_hold)

        sw = None if sample_weight is None else np.asarray(sample_weight)[fit_idx]
        seeds = exp.get("meta_seeds", (11,))
        meta = np.zeros((len(hold_idx), 6))
        for k in range(6):
            meta[:, k] = _fit_avg(HistGradientBoostingRegressor, exp, Xf, tg[fit_idx, k], Xh, sw, seeds)
        gain = np.zeros((len(hold_idx), 2))
        for k in range(2):
            gain[:, k] = _fit_avg(HistGradientBoostingRegressor, exp, Xf, dt[fit_idx, k], Xh, sw, seeds)
        if exp.get("ordinal", True):
            ord_scores = np.zeros((len(hold_idx), 3))
            for mi in range(3):
                cum = np.zeros(len(hold_idx))
                for th in ORDINAL_THRESHOLDS:
                    y = (tg[fit_idx, mi] >= th).astype(int)
                    if y.min() == y.max():
                        cum += float(y.min())
                        continue
                    raw = _fit_avg(HistGradientBoostingClassifier, exp, Xf, y, Xh, sw, seeds,
                                   proba=True)
                    cum += 1.0 / (1.0 + np.exp(-np.clip(raw, -50, 50)))
                ord_scores[:, mi] = cum / len(ORDINAL_THRESHOLDS)
            meta[:, :3] = ord_scores
        rank_eff = np.zeros((len(hold_idx), 2))
        floors = np.zeros(2)
        if exp.get("rank", True):
            grid = np.linspace(0.0, 1.0, LUT_NODES)
            tc = np.exp(tg[:, 3:6])
            for g, (a, b) in enumerate([(0, 1), (1, 2)]):
                ds = tg[:, b] - tg[:, a]
                dc = tc[:, b] - tc[:, a]
                fl = max(float(np.quantile(dc[fit_idx], RANK_FLOOR_Q)), 1e-9)
                eff = ds / np.maximum(dc, fl)
                r = rankdata(eff[fit_idx], method="average") / max(len(fit_idx) - 1, 1)
                pr = _fit_avg(HistGradientBoostingRegressor, exp, Xf, r, Xh, sw, seeds)
                q = np.quantile(eff[fit_idx], grid)
                rank_eff[:, g] = np.interp(np.clip(pr, 0.0, 1.0), grid, q)
                floors[g] = fl
        return dict(idx=np.asarray(hold_idx), lin=lin_hold, legacy=leg_hold,
                    fam=fam_hold, knn=knn_hold, meta=meta, gain=gain, rank_eff=rank_eff,
                    floors=floors)

    # ----------------------------------------------------------------- compose
    def compose(self, arr, cfg, tier):
        cfg = dict(DEPLOYED_CFG, **(cfg or {}))
        prod = cfg["legacy_w"] * arr["legacy"] + (1 - cfg["legacy_w"]) * arr["lin"]
        prod = (1 - cfg["fam_w"]) * prod + cfg["fam_w"] * arr["fam"]
        conf = np.clip(arr["knn"][:, 6], 0.0, 1.0)[:, None] * cfg["conf_scale"]
        prod = (1 - conf) * prod + conf * arr["knn"][:, :6]
        prod = prod.copy()
        prod[:, :3] = np.clip(prod[:, :3], 0.0, 1.0)
        meta = arr["meta"].copy()
        pc_meta = np.exp(np.clip(meta[:, 3:6], -50, 50))
        d = arr["gain"].copy()
        if cfg["rank_beta"] > 0:
            dch = np.column_stack([np.maximum(pc_meta[:, 1] - pc_meta[:, 0], arr["floors"][0]),
                                   np.maximum(pc_meta[:, 2] - pc_meta[:, 1], arr["floors"][1])])
            d = (1 - cfg["rank_beta"]) * d + cfg["rank_beta"] * arr["rank_eff"] * dch
        if cfg["gain_alpha"] > 0:
            recon = np.column_stack([meta[:, 0], meta[:, 0] + d[:, 0], meta[:, 0] + d[:, 0] + d[:, 1]])
            meta[:, :3] = (1 - cfg["gain_alpha"]) * meta[:, :3] + cfg["gain_alpha"] * recon
        b = cfg[f"blend_{tier}"]
        row = (1 - b) * prod + b * meta
        ps = np.clip(row[:, :3], 0.0, 1.0)
        pc = np.exp(np.clip(row[:, 3:6], -50, 50))
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        return ps, pc

    # -------------------------------------------------------------- allocation
    @staticmethod
    def allocate(ps, pc, mult, safety):
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

    def score(self, arr, cfg, safety, rows=None, verbose=False):
        """Exact final score of the composed predictions on arr['idx']."""
        idx = arr["idx"] if rows is None else rows
        out = {"tiers": {}}
        total = 0.0
        for t in TIERS:
            ps, pc = self.compose(arr, cfg, t)
            pick = self.allocate(ps, pc, MULTS[t], safety[t])
            r = np.arange(len(idx))
            ratio = self.true_c[idx][r, pick].sum() / self.true_c[idx][:, 0].sum()
            sc = self.true_s[idx][r, pick].mean()
            passed = ratio <= MULTS[t] + 1e-15
            out["tiers"][t] = dict(score=float(sc), ratio=float(ratio), passed=bool(passed),
                                   pick=pick, n_by_model=np.bincount(pick, minlength=3).tolist())
            total += W[t] * (sc if passed else 0.0)
            if verbose:
                print(f"  {t:9s} score={sc:.6f} ratio={ratio:.4f} passed={passed} "
                      f"counts={out['tiers'][t]['n_by_model']}")
        out["final"] = float(total)
        return out

    # --------------------------------------------------------- bootstrap / EV
    _samples = {}

    def samples_for(self, m, seed, nboot, size=880):
        key = (m, seed, nboot, size)
        if key not in Lab._samples:
            r = np.random.default_rng(seed)
            Lab._samples[key] = [r.integers(0, m, size=size) for _ in range(nboot)]
        return Lab._samples[key]

    def bootstrap_ev(self, arr, cfg, grids=None, seed=7, nboot=200, size=880):
        """Pick the per-tier safety ratio maximising E[score * budget_passed]."""
        grids = grids or {"fast": np.arange(0.90, 1.021, 0.01),
                          "balanced": np.arange(0.80, 0.981, 0.01),
                          "premium": np.arange(0.78, 0.961, 0.01)}
        idx = arr["idx"]
        m = len(idx)
        samples = self.samples_for(m, seed, nboot, size)
        ts = self.true_s[idx]; tc = self.true_c[idx]
        out = {}
        for t in TIERS:
            ps, pc = self.compose(arr, cfg, t)
            best = None
            for s in grids[t]:
                evs = []
                for smp in samples:
                    pick = self.allocate(ps[smp], pc[smp], MULTS[t], float(s))
                    r = np.arange(len(smp))
                    ratio = tc[smp][r, pick].sum() / tc[smp][:, 0].sum()
                    evs.append(0.0 if ratio > MULTS[t] else ts[smp][r, pick].mean())
                ev = float(np.mean(evs))
                bust = float(np.mean([e == 0.0 for e in evs]))
                if best is None or ev > best[0]:
                    best = (ev, float(s), bust)
            out[t] = best
        ev = sum(W[t] * out[t][0] for t in TIERS)
        return ev, out


def holdout(lab, cfg=None, exp=None, safety=None, extra=None, targets=None,
            sample_weight=None, verbose=True):
    """Train on train(1,760), score dev(880).  Returns the report dict."""
    if extra is not None:
        lab.set_extra_features(extra)
    arr = lab.fit_predict(lab.train_idx, lab.dev_idx, exp=exp, targets=targets,
                          sample_weight=sample_weight)
    rep = lab.score(arr, cfg, safety or DEPLOYED_SAFETY, verbose=verbose)
    if verbose:
        print(f"  FINAL = {rep['final']:.6f}")
    return rep, arr


if __name__ == "__main__":
    lab = Lab()
    t0 = time.perf_counter()
    rep, arr = holdout(lab)
    print(f"[lab] holdout in {time.perf_counter()-t0:.0f}s")
