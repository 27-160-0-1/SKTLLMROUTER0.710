# SPDX-License-Identifier: Apache-2.0
"""b05 step 9: how much of a better gain head actually reaches the allocator?

`compose` dilutes the gain head by rank_beta (0.4), gain_alpha (0.5) and the
per-tier blend (0.6/0.45/0.3), so a head improvement arrives at the decision
with weight ~0.18 in the fast tier.  a03 P1b asked for this sweep; here it is
run under the honest bench2 protocol with BOTH the deployed and an improved
gain head, so the interaction can be read off.
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
from harness import DEPLOYED_EXP, DEPLOYED_CFG, TIERS  # noqa: E402
import bench2 as B  # noqa: E402

EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)
lab = lib.XLab()
cv, arr, parts = lib.build_stage(lab, EXP, tag="b05base")
POS = {int(v): k for k, v in enumerate(cv["idx"])}
D = lab.delta_targets
GP = dict(max_iter=EXP["gbm_iter"], learning_rate=EXP["gbm_lr"],
          max_leaf_nodes=EXP["gbm_leaves"], min_samples_leaf=EXP["gbm_min_leaf"],
          l2_regularization=EXP["gbm_l2"], early_stopping=True,
          validation_fraction=0.15, random_state=11)
OUT = Path("reports/lab/b05_blend.json")
ROWS = json.loads(OUT.read_text()) if OUT.exists() else []


def build(fit_fn):
    cvg = np.zeros((len(cv["idx"]), 2))
    for p in parts:
        g = fit_fn(p["Xf"], p["Xh"], p["fit_idx"], p["idx"])
        for j, i in enumerate(p["idx"]):
            cvg[POS[int(i)]] = g[j]
    return cvg, fit_fn(arr["Xf"], arr["Xh"], arr["fit_idx"], arr["idx"])


def h_gbm(Xf, Xh, fi, hi):
    return np.column_stack([HistGradientBoostingRegressor(**GP).fit(Xf, D[fi, k]).predict(Xh)
                            for k in range(2)])


def h_ridge(Xf, Xh, fi, hi):
    sc = StandardScaler().fit(Xf)
    return Ridge(alpha=30.0).fit(sc.transform(Xf), D[fi]).predict(sc.transform(Xh))


def h_mix(Xf, Xh, fi, hi):
    return 0.5 * h_gbm(Xf, Xh, fi, hi) + 0.5 * h_ridge(Xf, Xh, fi, hi)


def sweep(tag, fit_fn):
    cvg, devg = build(fit_fn)
    c2, a2 = dict(cv, gain=cvg), dict(arr, gain=devg)
    dg = lib.gain_axis(lab, cv["idx"], cvg[:, 0], cvg[:, 1])
    print(f"--- {tag}: OOF corr d1={dg['corr1']:+.4f} d2={dg['corr2']:+.4f} "
          f"wfAUC1={dg['auc1']:.4f} wfAUC2={dg['auc2']:.4f} ---", flush=True)
    for ga in (0.5, 0.7, 0.85, 1.0):
        for rb in (0.0, 0.2, 0.4, 0.6):
            cfg = dict(DEPLOYED_CFG, gain_alpha=ga, rank_beta=rb)
            r = B.run(lab, c2, a2, cfg, label="", verbose=False)
            row = dict(head=tag, gain_alpha=ga, rank_beta=rb, EV=round(r["EV"], 6),
                       dev=round(r["dev"], 6),
                       raw=round(sum({"fast": .4, "balanced": .3, "premium": .3}[t] *
                                     r["det"][t]["raw"] for t in TIERS), 6),
                       safety=[round(r["safety"][t], 3) for t in TIERS],
                       ratio=[round(r["dev_tiers"][t]["ratio"], 3) for t in TIERS],
                       passed=[r["dev_tiers"][t]["passed"] for t in TIERS])
            ROWS.append(row)
            OUT.write_text(json.dumps(ROWS, indent=1), encoding="utf-8")
            print(f"  ga={ga:.2f} rb={rb:.2f}  EV={row['EV']:.6f} raw={row['raw']:.6f} "
                  f"dev={row['dev']:.6f} sf={row['safety']} r={row['ratio']}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["gbm", "mix"]
    fns = {"gbm": h_gbm, "mix": h_mix, "ridge": h_ridge}
    for w in which:
        sweep(w, fns[w])
