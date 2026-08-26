# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b05 step 2: buy gain-axis accuracy with training compute.

Every configuration replaces ONLY the two gain heads (arr['gain']); the rest of
the stage (ridge / legacy-OOF / kNN / family / 12 ordinal heads / 6 regressions /
2 rank heads) is the cached b05base stage.  Reported per config:

  wall  seconds of training compute for the two heads over 11 fits
  corr/AUC  the gain-axis diagnostics on the honest 10-fold train OOF rows
  EV/dev    the bench2 honest protocol (safety chosen on OOF, dev scored once)
"""
import importlib.util, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b05lib", HERE / "b05-time-heavy_lib.py")
lib = importlib.util.module_from_spec(spec); sys.modules["b05lib"] = lib
spec.loader.exec_module(lib)

from harness import DEPLOYED_EXP, DEPLOYED_CFG
import bench2 as B

EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)
lab = lib.XLab()
cv, arr, parts = lib.build_stage(lab, EXP, tag="b05base")
POS = {int(v): k for k, v in enumerate(cv["idx"])}
FAM = lab.fam_arr
D = lab.delta_targets                      # (n,2) true d1,d2
S = lab.true_s

RESULTS = []


def assemble(fit_fn):
    """Run fit_fn over the 10 OOF folds + the dev fit; return (cv_gain, dev_gain, secs)."""
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
    cv2 = dict(cv, gain=cvg); arr2 = dict(arr, gain=devg)
    r = B.run(lab, cv2, arr2, DEPLOYED_CFG, label=f"{name}", verbose=False)
    row = dict(name=name, secs=secs, note=note, **dg, EV=r["EV"], dev=r["dev"],
               safety=tuple(round(r["safety"][t], 3) for t in ("fast", "balanced", "premium")),
               bust=tuple(round(r["det"][t]["bust"] * 100, 1) for t in ("fast", "balanced", "premium")))
    RESULTS.append(row)
    print(f"{name:32s} {secs:6.1f}s  d1={dg['corr1']:+.4f} d2={dg['corr2']:+.4f} "
          f"AUC1={dg['auc1']:.4f} AUC2={dg['auc2']:.4f}  EV={r['EV']:.6f} dev={r['dev']:.6f} "
          f"safety={row['safety']} bust={row['bust']}", flush=True)
    return row


# ---------------------------------------------------------------- head recipes
def gbm(seeds=(11,), **over):
    p = dict(max_iter=EXP["gbm_iter"], learning_rate=EXP["gbm_lr"],
             max_leaf_nodes=EXP["gbm_leaves"], min_samples_leaf=EXP["gbm_min_leaf"],
             l2_regularization=EXP["gbm_l2"], early_stopping=True, validation_fraction=0.15)
    p.update(over)

    def f(Xf, Xh, fi, hi):
        out = np.zeros((len(hi), 2))
        for k in range(2):
            acc = np.zeros(len(hi))
            for s in seeds:
                acc += HistGradientBoostingRegressor(random_state=s, **p).fit(Xf, D[fi, k]).predict(Xh)
            out[:, k] = acc / len(seeds)
        return out
    return f


def gbm_target(ymaker, seeds=(11,), inverse=None, **over):
    """Train on a transformed target; `inverse(pred, fi)` maps back to a gain scale."""
    p = dict(max_iter=EXP["gbm_iter"], learning_rate=EXP["gbm_lr"],
             max_leaf_nodes=EXP["gbm_leaves"], min_samples_leaf=EXP["gbm_min_leaf"],
             l2_regularization=EXP["gbm_l2"], early_stopping=True, validation_fraction=0.15)
    p.update(over)

    def f(Xf, Xh, fi, hi):
        out = np.zeros((len(hi), 2))
        for k in range(2):
            y = ymaker(fi, k)
            acc = np.zeros(len(hi))
            for s in seeds:
                acc += HistGradientBoostingRegressor(random_state=s, **p).fit(Xf, y).predict(Xh)
            pr = acc / len(seeds)
            out[:, k] = inverse(pr, fi, k) if inverse else pr
        return out
    return f


if __name__ == "__main__":
    print("=== A. how much does raw training compute buy on the gain axis? ===", flush=True)
    evaluate("A0 deployed gain head", gbm(), "300 iters, lr .06, 15 leaves, 1 seed")
    evaluate("A1 5-seed average", gbm(seeds=tuple(range(11, 16))))
    evaluate("A2 20-seed average", gbm(seeds=tuple(range(11, 31))))
    evaluate("A3 lr.01 x3000 iters", gbm(learning_rate=0.01, max_iter=3000))
    evaluate("A4 lr.01 x3000 x5 seeds", gbm(seeds=tuple(range(11, 16)), learning_rate=0.01, max_iter=3000))
    evaluate("A5 63 leaves min_leaf 10", gbm(max_leaf_nodes=63, min_samples_leaf=10))
    evaluate("A6 wide+long+20seed", gbm(seeds=tuple(range(11, 31)), max_leaf_nodes=31,
                                        learning_rate=0.02, max_iter=2000, min_samples_leaf=15))
    evaluate("A7 heavy L2 (l2=30)", gbm(l2_regularization=30.0))
    evaluate("A8 no early stop, 100 it", gbm(early_stopping=False, max_iter=100))

    import json
    Path("reports/lab").mkdir(parents=True, exist_ok=True)
    Path("reports/lab/b05_headsA.json").write_text(json.dumps(RESULTS, indent=1), encoding="utf-8")
