# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b05 step 5: does an encoder representation beat the hashed features ON THE
GAIN AXIS?  This is the gate for the whole encoder direction (task item 4).

Same 10 folds as the b05base stage.  Everything is fitted inside the fold.
Reported: corr with the true d1/d2 and pooled within-family AUC of the sign,
on the honest train OOF rows.  Reference row = the deployed gain head.
"""
import importlib.util, json, os, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "5")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b05lib", HERE / "b05-time-heavy_lib.py")
lib = importlib.util.module_from_spec(spec); sys.modules["b05lib"] = lib
spec.loader.exec_module(lib)
from harness import DEPLOYED_EXP  # noqa: E402

EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)
lab = lib.XLab()
cv, arr, parts = lib.build_stage(lab, EXP, tag="b05base")
POS = {int(v): k for k, v in enumerate(cv["idx"])}
D = lab.delta_targets
Z = np.load("reports/lab/b05_embed.npz")
EMB = {"static": Z["static"], "frozen": Z["frozen"]}

GP = dict(max_iter=EXP["gbm_iter"], learning_rate=EXP["gbm_lr"],
          max_leaf_nodes=EXP["gbm_leaves"], min_samples_leaf=EXP["gbm_min_leaf"],
          l2_regularization=EXP["gbm_l2"], early_stopping=True,
          validation_fraction=0.15, random_state=11)

ROWS = []


def collect(name, maker):
    """maker(Xf58, Xh58, fit_idx, hold_idx) -> (n_hold, 2) gain predictions."""
    t0 = time.perf_counter()
    g = np.zeros((len(cv["idx"]), 2))
    for p in parts:
        out = maker(p["Xf"], p["Xh"], p["fit_idx"], p["idx"])
        for j, i in enumerate(p["idx"]):
            g[POS[int(i)]] = out[j]
    d = lib.gain_axis(lab, cv["idx"], g[:, 0], g[:, 1])
    d["name"] = name; d["secs"] = time.perf_counter() - t0
    ROWS.append(d)
    print(f"{name:40s} {d['secs']:6.1f}s  corr d1={d['corr1']:+.4f} d2={d['corr2']:+.4f}  "
          f"wfAUC1={d['auc1']:.4f} wfAUC2={d['auc2']:.4f}", flush=True)
    return g


def gbm_on(getX):
    def f(Xf, Xh, fi, hi):
        A, Bm = getX(Xf, Xh, fi, hi)
        out = np.zeros((len(hi), 2))
        for k in range(2):
            out[:, k] = HistGradientBoostingRegressor(**GP).fit(A, D[fi, k]).predict(Bm)
        return out
    return f


def ridge_on(getX, alpha=100.0):
    def f(Xf, Xh, fi, hi):
        A, Bm = getX(Xf, Xh, fi, hi)
        sc = StandardScaler().fit(A)
        m = Ridge(alpha=alpha).fit(sc.transform(A), D[fi])
        return m.predict(sc.transform(Bm))
    return f


def emb_only(key, ncomp=None):
    E = EMB[key]

    def g(Xf, Xh, fi, hi):
        if ncomp is None:
            return E[fi], E[hi]
        p = PCA(n_components=ncomp, random_state=0).fit(E[fi])
        return p.transform(E[fi]), p.transform(E[hi])
    return g


def emb_plus(key, ncomp=64):
    E = EMB[key]

    def g(Xf, Xh, fi, hi):
        p = PCA(n_components=ncomp, random_state=0).fit(E[fi])
        return np.hstack([Xf, p.transform(E[fi])]), np.hstack([Xh, p.transform(E[hi])])
    return g


if __name__ == "__main__":
    print("--- reference (the deployed gain head on the 58 features) ---", flush=True)
    d = lib.gain_axis(lab, cv["idx"], cv["gain"][:, 0], cv["gain"][:, 1])
    d["name"] = "REF deployed gain head"; d["secs"] = 0.0; ROWS.append(d)
    print(f"{d['name']:40s} {0.0:6.1f}s  corr d1={d['corr1']:+.4f} d2={d['corr2']:+.4f}  "
          f"wfAUC1={d['auc1']:.4f} wfAUC2={d['auc2']:.4f}", flush=True)
    collect("58feat GBM (rebuilt reference)", gbm_on(lambda Xf, Xh, fi, hi: (Xf, Xh)))
    collect("58feat ridge", ridge_on(lambda Xf, Xh, fi, hi: (Xf, Xh), alpha=30.0))

    print("--- encoder representations alone ---", flush=True)
    for key in ("static", "frozen"):
        collect(f"{key} 384d ridge a=300", ridge_on(emb_only(key), alpha=300.0))
        collect(f"{key} 384d ridge a=3000", ridge_on(emb_only(key), alpha=3000.0))
        collect(f"{key} 384d GBM", gbm_on(emb_only(key)))
        collect(f"{key} PCA64 GBM", gbm_on(emb_only(key, 64)))

    print("--- 58 features + encoder ---", flush=True)
    for key in ("static", "frozen"):
        collect(f"58feat + {key} PCA32 GBM", gbm_on(emb_plus(key, 32)))
        collect(f"58feat + {key} PCA64 GBM", gbm_on(emb_plus(key, 64)))

    Path("reports/lab/b05_encprobe.json").write_text(json.dumps(ROWS, indent=1), encoding="utf-8")
