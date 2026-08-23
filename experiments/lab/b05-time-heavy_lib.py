# SPDX-License-Identifier: Apache-2.0
"""b05 library: an instrumented copy of Lab.fit_predict that also returns the
58-feature meta matrices, so head-design experiments can be run WITHOUT redoing
the expensive ridge/kNN/legacy stage.

Nothing here modifies the repo; it subclasses harness.Lab.
"""
from __future__ import annotations

import pickle, sys, time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import (Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP,  # noqa: E402
                     ORDINAL_THRESHOLDS, LUT_NODES, RANK_FLOOR_Q)
import protocol as P  # noqa: E402
import bench2 as B  # noqa: E402

CACHE_DIR = Path("reports/lab")


def gbm_params(exp, seed=11):
    return dict(max_iter=exp["gbm_iter"], learning_rate=exp["gbm_lr"],
                max_leaf_nodes=exp["gbm_leaves"], min_samples_leaf=exp["gbm_min_leaf"],
                l2_regularization=exp["gbm_l2"], early_stopping=True,
                validation_fraction=0.15, random_state=seed)


class XLab(Lab):
    """Lab whose fit_predict also emits the meta feature matrices."""

    def fit_predict_x(self, fit_idx, hold_idx, exp=None):
        exp = dict(DEPLOYED_EXP, **(exp or {}))
        tg = self.targets
        dt = np.column_stack([tg[:, 1] - tg[:, 0], tg[:, 2] - tg[:, 1]])
        seeds = tuple(exp.get("gbm_seeds", (11,)))
        knn_fit, knn_hold, fam_fit, fam_hold = self._knn_family(fit_idx, hold_idx, tg)
        head = self.fit_legacy(fit_idx, exp.get("legacy_alpha", 100.0))
        leg_hold = self.predict_legacy(head, hold_idx)
        if exp.get("legacy_oof_meta", True):
            inner_l = np.random.default_rng(7).integers(0, 5, size=len(fit_idx))
            leg_fit = np.zeros((len(fit_idx), 6))
            for k in range(5):
                h = self.fit_legacy(fit_idx[inner_l != k], exp.get("legacy_alpha", 100.0))
                leg_fit[inner_l == k] = self.predict_legacy(h, fit_idx[inner_l == k])
        else:
            leg_fit = self.predict_legacy(head, fit_idx)

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

        Xf = np.hstack([self.dense[fit_idx], self.fam_onehot[fit_idx], leg_fit, oof, knn_fit])
        Xh = np.hstack([self.dense[hold_idx], self.fam_onehot[hold_idx], leg_hold,
                        lin_hold, knn_hold])

        def avg_reg(y, Xf_, Xh_, mono=None, **kw):
            acc = np.zeros(len(Xh_))
            for s in seeds:
                p = dict(gbm_params(exp, s), **kw)
                if mono is not None:
                    p["monotonic_cst"] = mono
                acc += HistGradientBoostingRegressor(**p).fit(Xf_, y).predict(Xh_)
            return acc / len(seeds)

        meta = np.zeros((len(hold_idx), 6))
        for k in range(6):
            meta[:, k] = avg_reg(tg[fit_idx, k], Xf, Xh)
        gain = np.zeros((len(hold_idx), 2))
        for k in range(2):
            gain[:, k] = avg_reg(dt[fit_idx, k], Xf, Xh)
        if exp.get("ordinal", True):
            ord_scores = np.zeros((len(hold_idx), 3))
            for mi in range(3):
                cum = np.zeros(len(hold_idx))
                for th in ORDINAL_THRESHOLDS:
                    y = (tg[fit_idx, mi] >= th).astype(int)
                    if y.min() == y.max():
                        cum += float(y.min()); continue
                    acc = np.zeros(len(hold_idx))
                    for s in seeds:
                        clf = HistGradientBoostingClassifier(**gbm_params(exp, s)).fit(Xf, y)
                        raw = clf.decision_function(Xh)
                        acc += 1.0 / (1.0 + np.exp(-np.clip(raw, -50, 50)))
                    cum += acc / len(seeds)
                ord_scores[:, mi] = cum / len(ORDINAL_THRESHOLDS)
            meta[:, :3] = ord_scores
        rank_eff = np.zeros((len(hold_idx), 2))
        floors = np.zeros(2)
        grid = np.linspace(0.0, 1.0, LUT_NODES)
        tc = np.exp(tg[:, 3:6])
        for g, (a, b) in enumerate([(0, 1), (1, 2)]):
            ds = tg[:, b] - tg[:, a]
            dc = tc[:, b] - tc[:, a]
            fl = max(float(np.quantile(dc[fit_idx], RANK_FLOOR_Q)), 1e-9)
            eff = ds / np.maximum(dc, fl)
            r = rankdata(eff[fit_idx], method="average") / max(len(fit_idx) - 1, 1)
            pr = np.clip(avg_reg(r, Xf, Xh), 0.0, 1.0)
            q = np.quantile(eff[fit_idx], grid)
            rank_eff[:, g] = np.interp(pr, grid, q)
            floors[g] = fl
        return dict(idx=np.asarray(hold_idx), lin=lin_hold, legacy=leg_hold, fam=fam_hold,
                    knn=knn_hold, meta=meta, gain=gain, rank_eff=rank_eff, floors=floors,
                    Xf=Xf, Xh=Xh, fit_idx=np.asarray(fit_idx))


def build_stage(lab, exp=None, tag="b05base", folds=10, seed=123, n_jobs=4, force=False):
    """cv arrays (train OOF, 10 folds) + dev arrays, both with X matrices kept.

    threading backend: the class lives in a hyphenated file that a loky worker
    cannot import by name, and HistGradientBoosting releases the GIL in its
    OpenMP kernels, so threads are the practical choice here.
    """
    f = CACHE_DIR / f"stage_{tag}.pkl"
    if f.exists() and not force:
        b = pickle.loads(f.read_bytes())
        return b["cv"], b["arr"], b["parts"]
    t0 = time.perf_counter()
    idx = lab.train_idx
    fold_of = np.random.default_rng(seed).integers(0, folds, size=len(idx))
    parts = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(lab.fit_predict_x)(idx[fold_of != k], idx[fold_of == k], exp)
        for k in range(folds))
    cv = {"idx": np.concatenate([p["idx"] for p in parts])}
    for k in ("lin", "legacy", "fam", "knn", "meta", "gain", "rank_eff"):
        cv[k] = np.vstack([p[k] for p in parts])
    cv["floors"] = parts[0]["floors"]
    arr = lab.fit_predict_x(lab.train_idx, lab.dev_idx, exp)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps({"cv": cv, "arr": arr, "parts": parts}))
    print(f"[b05] stage '{tag}' built in {time.perf_counter()-t0:.0f}s", flush=True)
    return cv, arr, parts


# ------------------------------------------------------------------ diagnostics
def wf_auc(pred, truth, fam):
    """Pooled within-family AUC of pred against sign(truth)."""
    num = den = 0.0
    for f in np.unique(fam):
        m = fam == f
        pos = pred[m][truth[m] > 0]; neg = pred[m][truth[m] < 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        w = len(pos) * len(neg)
        M = pos[:, None] - neg[None, :]
        a = float((M > 0).mean() + 0.5 * (M == 0).mean())
        num += a * w; den += w
    return num / den if den else float("nan")


def gain_axis(lab, idx, g1, g2, prefix=""):
    ts = lab.true_s[idx]
    d1 = ts[:, 1] - ts[:, 0]; d2 = ts[:, 2] - ts[:, 1]
    fam = lab.fam_arr[idx]
    out = dict(corr1=float(np.corrcoef(g1, d1)[0, 1]), corr2=float(np.corrcoef(g2, d2)[0, 1]),
               auc1=wf_auc(g1, d1, fam), auc2=wf_auc(g2, d2, fam))
    if prefix:
        print(f"{prefix:34s} corr d1={out['corr1']:+.4f} d2={out['corr2']:+.4f}  "
              f"wfAUC1={out['auc1']:.4f} wfAUC2={out['auc2']:.4f}", flush=True)
    return out


def inject_gain(new_cv_gain, new_arr_gain):
    """A bench2 transform is not enough (gain feeds compose before the blend), so
    instead we clone the stage dicts with the gain block replaced."""
    def apply(cv, arr):
        cv2 = dict(cv); arr2 = dict(arr)
        cv2["gain"] = new_cv_gain; arr2["gain"] = new_arr_gain
        return cv2, arr2
    return apply
