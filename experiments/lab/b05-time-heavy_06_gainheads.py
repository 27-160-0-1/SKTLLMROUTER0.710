# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b05 step 6: heads trained DIRECTLY on the decision axis.

Every configuration replaces only arr['gain']; the rest of the b05base stage is
untouched.  Judged by the bench2 honest protocol (safety chosen by 3-seed x 400
bootstrap EV on the train OOF rows, dev scored once) AND by the gain-axis
diagnostics on the same OOF rows.

Usage:  python b05-time-heavy_06_gainheads.py <group> [<group> ...]
groups: ridge blend enc rank resid mono dfw seedlong all
"""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b05lib", HERE / "b05-time-heavy_lib.py")
lib = importlib.util.module_from_spec(spec); sys.modules["b05lib"] = lib
spec.loader.exec_module(lib)
from harness import DEPLOYED_EXP, DEPLOYED_CFG, MULTS, TIERS  # noqa: E402
import bench2 as B  # noqa: E402

EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)
lab = lib.XLab()
cv, arr, parts = lib.build_stage(lab, EXP, tag="b05base")
POS = {int(v): k for k, v in enumerate(cv["idx"])}
D = lab.delta_targets
FAM = lab.fam_arr
Z = np.load("reports/lab/b05_embed.npz")
EMB = {"static": Z["static"], "frozen": Z["frozen"]}
OUT = Path("reports/lab/b05_headsB.json")
RESULTS = json.loads(OUT.read_text()) if OUT.exists() else []

GP = dict(max_iter=EXP["gbm_iter"], learning_rate=EXP["gbm_lr"],
          max_leaf_nodes=EXP["gbm_leaves"], min_samples_leaf=EXP["gbm_min_leaf"],
          l2_regularization=EXP["gbm_l2"], early_stopping=True,
          validation_fraction=0.15, random_state=11)


def assemble(fit_fn):
    t0 = time.perf_counter()
    cvg = np.zeros((len(cv["idx"]), 2))
    for p in parts:
        g = fit_fn(p["Xf"], p["Xh"], p["fit_idx"], p["idx"])
        for j, i in enumerate(p["idx"]):
            cvg[POS[int(i)]] = g[j]
    devg = fit_fn(arr["Xf"], arr["Xh"], arr["fit_idx"], arr["idx"])
    return cvg, devg, time.perf_counter() - t0


def evaluate(name, fit_fn, note=""):
    cvg, devg, secs = assemble(fit_fn)
    dg = lib.gain_axis(lab, cv["idx"], cvg[:, 0], cvg[:, 1])
    r = B.run(lab, dict(cv, gain=cvg), dict(arr, gain=devg), DEPLOYED_CFG, label=name, verbose=False)
    row = dict(name=name, secs=round(secs, 1), note=note, **{k: round(v, 4) for k, v in dg.items()},
               EV=round(r["EV"], 6), dev=round(r["dev"], 6),
               raw=round(sum({"fast": .4, "balanced": .3, "premium": .3}[t] * r["det"][t]["raw"]
                             for t in TIERS), 6),
               safety=[round(r["safety"][t], 3) for t in TIERS],
               bust=[round(r["det"][t]["bust"] * 100, 1) for t in TIERS],
               tier=[round(r["dev_tiers"][t]["score"], 4) for t in TIERS],
               ratio=[round(r["dev_tiers"][t]["ratio"], 3) for t in TIERS],
               passed=[r["dev_tiers"][t]["passed"] for t in TIERS])
    RESULTS.append(row)
    OUT.write_text(json.dumps(RESULTS, indent=1), encoding="utf-8")
    print(f"{name:34s}{secs:7.1f}s d1={dg['corr1']:+.4f} d2={dg['corr2']:+.4f} "
          f"A1={dg['auc1']:.4f} A2={dg['auc2']:.4f} EV={r['EV']:.6f} raw={row['raw']:.6f} "
          f"dev={r['dev']:.6f} sf={row['safety']} bust={row['bust']} r={row['ratio']}", flush=True)
    return cvg, devg


# --------------------------------------------------------------- head recipes
def h_gbm(seeds=(11,), **over):
    p = dict(GP, **over)

    def f(Xf, Xh, fi, hi):
        out = np.zeros((len(hi), 2))
        for k in range(2):
            acc = np.zeros(len(hi))
            for s in seeds:
                acc += HistGradientBoostingRegressor(**dict(p, random_state=s)).fit(Xf, D[fi, k]).predict(Xh)
            out[:, k] = acc / len(seeds)
        return out
    return f


def h_ridge(alpha=30.0, block=None):
    def f(Xf, Xh, fi, hi):
        A, Bm = (Xf, Xh) if block is None else (np.hstack([Xf, EMB[block][fi]]),
                                                np.hstack([Xh, EMB[block][hi]]))
        sc = StandardScaler().fit(A)
        return Ridge(alpha=alpha).fit(sc.transform(A), D[fi]).predict(sc.transform(Bm))
    return f


def h_blend(fa, fb, w):
    def f(Xf, Xh, fi, hi):
        return (1 - w) * fa(Xf, Xh, fi, hi) + w * fb(Xf, Xh, fi, hi)
    return f


def h_rank_target(within_family=True, **over):
    """GBM on the within-family rank of the gain (a listwise ranking target)."""
    p = dict(GP, **over)

    def f(Xf, Xh, fi, hi):
        out = np.zeros((len(hi), 2))
        for k in range(2):
            y = np.zeros(len(fi))
            if within_family:
                for fam in np.unique(FAM[fi]):
                    m = FAM[fi] == fam
                    y[m] = (rankdata(D[fi[m], k]) - 0.5) / m.sum()
            else:
                y = (rankdata(D[fi, k]) - 0.5) / len(fi)
            pr = np.clip(HistGradientBoostingRegressor(**p).fit(Xf, y).predict(Xh), 0, 1)
            q = np.quantile(D[fi, k], np.linspace(0, 1, 65))
            out[:, k] = np.interp(pr, np.linspace(0, 1, 65), q)
        return out
    return f


def h_resid(**over):
    """GBM on the family-residualised gain, family mean added back."""
    p = dict(GP, **over)

    def f(Xf, Xh, fi, hi):
        out = np.zeros((len(hi), 2))
        mu = {fam: D[fi[FAM[fi] == fam]].mean(axis=0) for fam in np.unique(FAM[fi])}
        gm = D[fi].mean(axis=0)
        base_f = np.array([mu.get(FAM[i], gm) for i in fi])
        base_h = np.array([mu.get(FAM[i], gm) for i in hi])
        for k in range(2):
            r = HistGradientBoostingRegressor(**p).fit(Xf, D[fi, k] - base_f[:, k]).predict(Xh)
            out[:, k] = base_h[:, k] + r
        return out
    return f


def h_mono():
    """Monotone-increasing constraints on the four hand-built gain features.

    Column layout of the 58-feature block: dense30 | fam9 | legacy6 | lin6 | knn7.
    legacy/lin/knn each carry (s_light, s_mid, s_k1, ...), so the differences
    are the direct gain evidence.  They are appended and constrained +1.
    """
    def gains(X):
        leg = X[:, 39:42]; lin = X[:, 45:48]; knn = X[:, 51:54]
        return np.column_stack([leg[:, 1] - leg[:, 0], leg[:, 2] - leg[:, 1],
                                lin[:, 1] - lin[:, 0], lin[:, 2] - lin[:, 1],
                                knn[:, 1] - knn[:, 0], knn[:, 2] - knn[:, 1]])

    def f(Xf, Xh, fi, hi):
        Gf, Gh = gains(Xf), gains(Xh)
        A = np.hstack([Xf, Gf]); Bm = np.hstack([Xh, Gh])
        out = np.zeros((len(hi), 2))
        for k in range(2):
            cst = np.zeros(A.shape[1]); cst[58 + 2 * k] = 1; cst[58 + 2 * k + 1] = 1
            out[:, k] = HistGradientBoostingRegressor(**dict(GP, monotonic_cst=cst)).fit(
                A, D[fi, k]).predict(Bm)
        return out
    return f


def h_decision_weight(tier="fast", power=1.0, **over):
    """Decision-focused sample weighting: weight each training row by how close
    its slope Delta_s/Delta_c is to the Lagrangian threshold that the allocator
    actually uses at that tier, i.e. by how likely a change in the prediction is
    to change a decision.  Rows far inside or far outside the budget get ~0."""
    p = dict(GP, **over)
    thr = {}
    for t in TIERS:
        ps, pc = lab.compose(cv, DEPLOYED_CFG, t)
        dc = np.column_stack([pc[:, 1] - pc[:, 0], pc[:, 2] - pc[:, 1]])
        sl = np.column_stack([(ps[:, 1] - ps[:, 0]) / np.maximum(dc[:, 0], 1e-12),
                              (ps[:, 2] - ps[:, 1]) / np.maximum(dc[:, 1], 1e-12)])
        cap = pc[:, 0].sum() * max(1.0, MULTS[t] * 0.9)
        flat_sl = sl.ravel(); flat_dc = dc.ravel()
        o = np.argsort(-flat_sl)
        cum = pc[:, 0].sum() + np.cumsum(flat_dc[o])
        kk = int(np.searchsorted(cum, cap))
        pi = float(flat_sl[o][min(kk, len(o) - 1)])
        thr[t] = (pi, sl, dc)

    def f(Xf, Xh, fi, hi):
        pi, sl, dc = thr[tier]
        w = np.ones((len(fi), 2))
        rows = np.array([POS.get(int(i), -1) for i in fi])
        ok = rows >= 0
        if ok.any():
            z = np.log(np.maximum(sl[rows[ok]], 1e-12)) - np.log(max(pi, 1e-12))
            w[ok] = np.exp(-np.abs(z) / 2.0) ** power
        w = np.maximum(w, 0.05)
        out = np.zeros((len(hi), 2))
        for k in range(2):
            out[:, k] = HistGradientBoostingRegressor(**p).fit(
                Xf, D[fi, k], sample_weight=w[:, k]).predict(Xh)
        return out
    return f


GROUPS = {}


def group(name):
    def deco(fn):
        GROUPS[name] = fn
        return fn
    return deco


@group("ridge")
def _ridge():
    evaluate("B0 deployed GBM gain (ref)", h_gbm())
    for a in (3.0, 10.0, 30.0, 100.0, 300.0):
        evaluate(f"B1 ridge gain a={a:g}", h_ridge(a))


@group("blend")
def _blend():
    for w in (0.25, 0.5, 0.75):
        evaluate(f"B2 GBM*{1-w:.2f}+ridge30*{w:.2f}", h_blend(h_gbm(), h_ridge(30.0), w))


@group("enc")
def _enc():
    for blk in ("static", "frozen"):
        evaluate(f"B3 ridge a=300 58+{blk}", h_ridge(300.0, blk))
        evaluate(f"B3 ridge a=1000 58+{blk}", h_ridge(1000.0, blk))
        evaluate(f"B4 GBM.5+{blk}ridge1000.5", h_blend(h_gbm(), h_ridge(1000.0, blk), 0.5))


@group("rank")
def _rank():
    evaluate("B5 within-family rank target", h_rank_target(True))
    evaluate("B5 global rank target", h_rank_target(False))


@group("resid")
def _resid():
    evaluate("B6 family-residualised target", h_resid())


@group("mono")
def _mono():
    evaluate("B7 monotone gain features", h_mono())


@group("dfw")
def _dfw():
    for t in ("fast", "premium"):
        evaluate(f"B8 decision-weighted ({t})", h_decision_weight(t))


@group("seedlong")
def _seedlong():
    evaluate("B9 lr.02 x1500 8 seeds", h_gbm(seeds=tuple(range(11, 19)), learning_rate=0.02,
                                             max_iter=1500))


if __name__ == "__main__":
    want = sys.argv[1:] or ["ridge"]
    if want == ["all"]:
        want = list(GROUPS)
    for g in want:
        print(f"===== group {g} =====", flush=True)
        GROUPS[g]()
