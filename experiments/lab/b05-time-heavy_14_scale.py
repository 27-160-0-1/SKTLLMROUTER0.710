# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b05 step 14: is the EV difference between gain heads an ORDERING effect or a
SCALE effect?

The allocator is invariant to multiplying BOTH gains by the same positive
constant (a05 I2), but NOT to the ratio sd(g2)/sd(g1), nor to the offset: the
Lagrangian threshold is a slope, so a head that shrinks d1 more than d2 buys a
different mix of mid- and k1-upgrades at the same budget.  Here each head is
(a) measured as trained, and (b) affinely rescaled per column to the deployed
GBM head's mean and sd, which leaves its within-family ordering EXACTLY intact
and removes the scale difference.
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
CFG = dict(DEPLOYED_CFG, gain_alpha=1.0, rank_beta=0.0)
OUT = Path("reports/lab/b05_scale.json")
ROWS = []


def assemble(fit_fn):
    cvg = np.zeros((len(cv["idx"]), 2))
    for p in parts:
        g = fit_fn(p["Xf"], p["Xh"], p["fit_idx"], p["idx"])
        for j, i in enumerate(p["idx"]):
            cvg[POS[int(i)]] = g[j]
    return cvg, fit_fn(arr["Xf"], arr["Xh"], arr["fit_idx"], arr["idx"])


def ev(name, cvg, devg, cfg=CFG):
    dg = lib.gain_axis(lab, cv["idx"], cvg[:, 0], cvg[:, 1])
    r = B.run(lab, dict(cv, gain=cvg), dict(arr, gain=devg), cfg, label=name, verbose=False)
    row = dict(name=name, **{k: round(v, 4) for k, v in dg.items()},
               sd1=round(float(cvg[:, 0].std()), 4), sd2=round(float(cvg[:, 1].std()), 4),
               mu1=round(float(cvg[:, 0].mean()), 4), mu2=round(float(cvg[:, 1].mean()), 4),
               EV=round(r["EV"], 6), raw=round(sum(W[t] * r["det"][t]["raw"] for t in TIERS), 6),
               dev=round(r["dev"], 6), safety=[round(r["safety"][t], 3) for t in TIERS],
               ratio=[round(r["dev_tiers"][t]["ratio"], 3) for t in TIERS])
    ROWS.append(row); OUT.write_text(json.dumps(ROWS, indent=1), encoding="utf-8")
    print(f"{name:34s} mu=({row['mu1']:+.4f},{row['mu2']:+.4f}) sd=({row['sd1']:.4f},"
          f"{row['sd2']:.4f}) r21={row['sd2']/max(row['sd1'],1e-9):5.2f} "
          f"A1={dg['auc1']:.4f} A2={dg['auc2']:.4f} EV={row['EV']:.6f} raw={row['raw']:.6f} "
          f"dev={row['dev']:.6f} sf={row['safety']}", flush=True)


def h_gbm(Xf, Xh, fi, hi):
    return np.column_stack([HistGradientBoostingRegressor(**GP).fit(Xf, D[fi, k]).predict(Xh)
                            for k in range(2)])


def h_ridge(alpha=30.0, block=None):
    def f(Xf, Xh, fi, hi):
        A, Bm = (Xf, Xh) if block is None else (np.hstack([Xf, EMB[block][fi]]),
                                                np.hstack([Xh, EMB[block][hi]]))
        sc = StandardScaler().fit(A)
        return Ridge(alpha=alpha).fit(sc.transform(A), D[fi]).predict(sc.transform(Bm))
    return f


def match(x, ref):
    s = x.std(axis=0); s = np.where(s > 1e-12, s, 1.0)
    return (x - x.mean(axis=0)) / s * ref.std(axis=0) + ref.mean(axis=0)


if __name__ == "__main__":
    base_cv, base_dev = assemble(h_gbm)
    ev("V0 GBM (reference)", base_cv, base_dev)
    for nm, fn in (("ridge30", h_ridge(30.0)), ("ridge300", h_ridge(300.0)),
                   ("58+frozen a1000", h_ridge(1000.0, "frozen")),
                   ("58+static a300", h_ridge(300.0, "static"))):
        g, dv = assemble(fn)
        ev(f"V1 {nm} as trained", g, dv)
        ev(f"V2 {nm} scale-matched", match(g, base_cv), match(dv, base_dev))
    # and the converse: the GBM head rescaled to the ridge's scale
    g, dv = assemble(h_ridge(30.0))
    ev("V3 GBM rescaled to ridge30", match(base_cv, g), match(base_dev, dv))
