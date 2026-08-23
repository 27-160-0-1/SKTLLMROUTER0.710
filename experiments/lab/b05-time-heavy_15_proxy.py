# SPDX-License-Identifier: Apache-2.0
"""b05 step 15: find a CHEAP statistic that actually predicts the honest EV,
and re-test the E18 verdict (MLP meta-head) on the gain axis.

Steps 6/13/14 showed that within-family gain AUC does NOT predict EV (it is
uncorrelated at the deployed blend and NEGATIVELY correlated undiluted), and
that per-column rescaling explains only ~25 % of the gap.  The remaining
suspect is the GLOBAL, cost-weighted efficiency ordering, which is what the
Lagrangian actually consumes.  Two candidate statistics per head:

  eff_rho   Spearman of the predicted segment slope (g_k / dc_pred_k) against
            the realised slope (ds_true_k / dc_true_k), pooled over all 2n
            segments of the OOF rows.
  oracost   the realised score of the exact allocator run with this head's
            composed scores but the TRUE costs at safety 1.0 -- a pure
            score-side decision-quality number with the budget channel removed.
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b05lib", HERE / "b05-time-heavy_lib.py")
lib = importlib.util.module_from_spec(spec); sys.modules["b05lib"] = lib
spec.loader.exec_module(lib)
from harness import DEPLOYED_EXP, DEPLOYED_CFG, TIERS, MULTS, W  # noqa: E402
import bench2 as B  # noqa: E402
import protocol as P  # noqa: E402

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
OUT = Path("reports/lab/b05_proxy.json")
ROWS = []
IDX = cv["idx"]
TS = lab.true_s[IDX]; TC = lab.true_c[IDX]
DSt = np.column_stack([TS[:, 1] - TS[:, 0], TS[:, 2] - TS[:, 1]])
DCt = np.column_stack([TC[:, 1] - TC[:, 0], TC[:, 2] - TC[:, 1]])
TRUE_SLOPE = DSt / np.maximum(DCt, 1e-12)


def assemble(fit_fn):
    t0 = time.perf_counter()
    cvg = np.zeros((len(IDX), 2))
    for p in parts:
        g = fit_fn(p["Xf"], p["Xh"], p["fit_idx"], p["idx"])
        for j, i in enumerate(p["idx"]):
            cvg[POS[int(i)]] = g[j]
    return cvg, fit_fn(arr["Xf"], arr["Xh"], arr["fit_idx"], arr["idx"]), time.perf_counter() - t0


def diagnostics(cvg, cfg):
    ps, pc = lab.compose(dict(cv, gain=cvg), cfg, "fast")
    dcp = np.column_stack([np.maximum(pc[:, 1] - pc[:, 0], 1e-12),
                           np.maximum(pc[:, 2] - pc[:, 1], 1e-12)])
    gp = np.column_stack([ps[:, 1] - ps[:, 0], ps[:, 2] - ps[:, 1]])
    pred_slope = gp / dcp
    rho = float(spearmanr(pred_slope.ravel(), TRUE_SLOPE.ravel()).statistic)
    # score-side decision quality: exact allocator, predicted scores, TRUE costs,
    # safety 1.0 -> the budget channel is removed entirely
    oc = 0.0
    for t in TIERS:
        ps_t, _ = lab.compose(dict(cv, gain=cvg), cfg, t)
        pick = P.exact_allocate(ps_t, TC, MULTS[t], 1.0)
        oc += W[t] * float(TS[np.arange(len(IDX)), pick].mean())
    return rho, oc


def evaluate(name, fit_fn, cfg=CFG):
    cvg, devg, secs = assemble(fit_fn)
    dg = lib.gain_axis(lab, IDX, cvg[:, 0], cvg[:, 1])
    rho, oc = diagnostics(cvg, cfg)
    r = B.run(lab, dict(cv, gain=cvg), dict(arr, gain=devg), cfg, label=name, verbose=False)
    row = dict(name=name, secs=round(secs, 1), **{k: round(v, 4) for k, v in dg.items()},
               eff_rho=round(rho, 4), oracost=round(oc, 6), EV=round(r["EV"], 6),
               raw=round(sum(W[t] * r["det"][t]["raw"] for t in TIERS), 6),
               dev=round(r["dev"], 6), safety=[round(r["safety"][t], 3) for t in TIERS])
    ROWS.append(row); OUT.write_text(json.dumps(ROWS, indent=1), encoding="utf-8")
    print(f"{name:32s}{secs:6.1f}s A1={dg['auc1']:.4f} A2={dg['auc2']:.4f} "
          f"eff_rho={rho:+.4f} oracost={oc:.6f} EV={row['EV']:.6f} raw={row['raw']:.6f} "
          f"dev={row['dev']:.6f}", flush=True)


def h_gbm(seeds=(11,), **over):
    p = dict(GP, **over)

    def f(Xf, Xh, fi, hi):
        return np.column_stack([
            np.mean([HistGradientBoostingRegressor(**dict(p, random_state=s)).fit(
                Xf, D[fi, k]).predict(Xh) for s in seeds], axis=0) for k in range(2)])
    return f


def h_ridge(alpha=30.0, block=None):
    def f(Xf, Xh, fi, hi):
        A, Bm = (Xf, Xh) if block is None else (np.hstack([Xf, EMB[block][fi]]),
                                                np.hstack([Xh, EMB[block][hi]]))
        sc = StandardScaler().fit(A)
        return Ridge(alpha=alpha).fit(sc.transform(A), D[fi]).predict(sc.transform(Bm))
    return f


def h_mlp(hidden=64, epochs=400, lr=3e-3, seeds=(0, 1, 2)):
    """E18's rejected MLP meta-head, re-tested on the gain axis."""
    def f(Xf, Xh, fi, hi):
        sc = StandardScaler().fit(Xf)
        A = torch.tensor(sc.transform(Xf), dtype=torch.float32)
        Bm = torch.tensor(sc.transform(Xh), dtype=torch.float32)
        y = D[fi]; mu, sd = y.mean(0), y.std(0) + 1e-9
        T = torch.tensor((y - mu) / sd, dtype=torch.float32)
        acc = np.zeros((len(hi), 2))
        for s in seeds:
            torch.manual_seed(s)
            m = nn.Sequential(nn.Linear(A.shape[1], hidden), nn.GELU(), nn.Dropout(0.2),
                              nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            for _ in range(epochs):
                opt.zero_grad(set_to_none=True)
                nn.functional.mse_loss(m(A), T).backward()
                opt.step(); sch.step()
            with torch.no_grad():
                acc += m(Bm).numpy() * sd + mu
        return acc / len(seeds)
    return f


if __name__ == "__main__":
    evaluate("W0 GBM (deployed head)", h_gbm())
    evaluate("W1 ridge a=30", h_ridge(30.0))
    evaluate("W1 ridge a=300", h_ridge(300.0))
    evaluate("W2 58+frozen a=1000", h_ridge(1000.0, "frozen"))
    evaluate("W2 58+static a=300", h_ridge(300.0, "static"))
    evaluate("W3 MLP 64-64 (E18 recheck)", h_mlp())
    evaluate("W4 20-seed GBM", h_gbm(seeds=tuple(range(11, 31))))
    evaluate("W5 GBM 63 leaves", h_gbm(max_leaf_nodes=63, min_samples_leaf=10))
    print("\n--- does the statistic predict EV? ---")
    a1 = np.array([r["auc1"] for r in ROWS]); a2 = np.array([r["auc2"] for r in ROWS])
    rho = np.array([r["eff_rho"] for r in ROWS]); oc = np.array([r["oracost"] for r in ROWS])
    evs = np.array([r["EV"] for r in ROWS])
    for nm, x in (("wfAUC1", a1), ("wfAUC2", a2), ("eff_rho", rho), ("oracost", oc)):
        print(f"  corr({nm:8s}, EV) = {np.corrcoef(x, evs)[0,1]:+.3f}")
