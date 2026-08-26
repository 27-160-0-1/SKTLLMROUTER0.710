# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b05 step 13: the same gain heads, but with the blend constants that let the
gain head actually reach the allocator (gain_alpha 1.0, rank_beta 0.0).

At the deployed constants (gain_alpha .5, rank_beta .4, tier blend .6/.45/.3) a
gain-head improvement arrives at the fast-tier decision with weight
0.6 x 0.5 x 0.6 = 0.18, so the measured exchange rate between gain-axis quality
and honest EV is ~0.  This script re-measures the same heads undiluted, which
is the control that decides whether the flat exchange rate is a property of the
signal or of the blend.
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b05lib", HERE / "b05-time-heavy_lib.py")
lib = importlib.util.module_from_spec(spec); sys.modules["b05lib"] = lib
spec.loader.exec_module(lib)
from harness import DEPLOYED_EXP, DEPLOYED_CFG, TIERS, W  # noqa: E402
import bench2 as B  # noqa: E402

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
OUT = Path("reports/lab/b05_undiluted.json")
ROWS = json.loads(OUT.read_text()) if OUT.exists() else []
CFG = dict(DEPLOYED_CFG, gain_alpha=1.0, rank_beta=0.0)


def assemble(fit_fn):
    t0 = time.perf_counter()
    cvg = np.zeros((len(cv["idx"]), 2))
    for p in parts:
        g = fit_fn(p["Xf"], p["Xh"], p["fit_idx"], p["idx"])
        for j, i in enumerate(p["idx"]):
            cvg[POS[int(i)]] = g[j]
    return cvg, fit_fn(arr["Xf"], arr["Xh"], arr["fit_idx"], arr["idx"]), time.perf_counter() - t0


def evaluate(name, fit_fn):
    cvg, devg, secs = assemble(fit_fn)
    dg = lib.gain_axis(lab, cv["idx"], cvg[:, 0], cvg[:, 1])
    r = B.run(lab, dict(cv, gain=cvg), dict(arr, gain=devg), CFG, label=name, verbose=False)
    row = dict(name=name, secs=round(secs, 1), **{k: round(v, 4) for k, v in dg.items()},
               EV=round(r["EV"], 6), raw=round(sum(W[t] * r["det"][t]["raw"] for t in TIERS), 6),
               dev=round(r["dev"], 6), safety=[round(r["safety"][t], 3) for t in TIERS],
               ratio=[round(r["dev_tiers"][t]["ratio"], 3) for t in TIERS])
    ROWS.append(row); OUT.write_text(json.dumps(ROWS, indent=1), encoding="utf-8")
    print(f"{name:34s}{secs:7.1f}s d1={dg['corr1']:+.4f} d2={dg['corr2']:+.4f} "
          f"A1={dg['auc1']:.4f} A2={dg['auc2']:.4f} EV={row['EV']:.6f} raw={row['raw']:.6f} "
          f"dev={row['dev']:.6f} sf={row['safety']} r={row['ratio']}", flush=True)


def h_gbm(seeds=(11,), **over):
    p = dict(GP, **over)

    def f(Xf, Xh, fi, hi):
        out = np.zeros((len(hi), 2))
        for k in range(2):
            acc = np.zeros(len(hi))
            for s in seeds:
                acc += HistGradientBoostingRegressor(**dict(p, random_state=s)).fit(
                    Xf, D[fi, k]).predict(Xh)
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


if __name__ == "__main__":
    evaluate("U0 deployed GBM gain", h_gbm())
    evaluate("U1 ridge a=30", h_ridge(30.0))
    evaluate("U1 ridge a=300", h_ridge(300.0))
    evaluate("U2 GBM.5+ridge30.5", h_blend(h_gbm(), h_ridge(30.0), 0.5))
    evaluate("U3 ridge a=1000 58+frozen", h_ridge(1000.0, "frozen"))
    evaluate("U3 ridge a=300 58+static", h_ridge(300.0, "static"))
    evaluate("U4 GBM.5+frozenridge.5", h_blend(h_gbm(), h_ridge(1000.0, "frozen"), 0.5))
    evaluate("U5 20-seed GBM", h_gbm(seeds=tuple(range(11, 31))))
