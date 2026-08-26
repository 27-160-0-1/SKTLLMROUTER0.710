# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b06 forest-view: the free-parameter inventory and its effective degrees of freedom.

  * ridge:   edf = sum_i  lam_i / (lam_i + alpha)  over the eigenvalues of X X^T
             (rank <= n_fit, so the 1,760 x 1,760 Gram matrix is enough)
  * GBM:     actual leaves after early stopping, summed over all 22 heads
  * legacy:  256 x 6 ridge, refit per fold
  * plus the hand-set constants and the tables.
Also: the exact invariance check of the score LEVEL channel, and the true risk
scale of a dev difference (bust probability under an 880-item dev bootstrap).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP, _gbm_params, ORDINAL_THRESHOLDS
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from scipy.stats import rankdata
import bench2 as B

OUT = Path("reports/lab/b06_dof.json")


def main():
    lab = Lab()
    rep = {}

    # ------------------------------------------------------------ ridge edf
    tr = lab.train_idx
    Xt = lab.X[tr]
    G = (Xt @ Xt.T).toarray()
    lam = np.linalg.eigvalsh(G)
    lam = np.clip(lam, 0, None)
    edf = {}
    for a in (1.0, 10.0, 30.0, 100.0, 1000.0):
        edf[a] = float((lam / (lam + a)).sum())
    rep["ridge"] = dict(n_coef=int(lab.X.shape[1] * 6), dims=int(lab.X.shape[1]),
                        nnz_per_row=float(Xt.nnz / Xt.shape[0]),
                        edf_per_output=edf, alpha=DEPLOYED_EXP["ridge_alpha"],
                        edf_total_6outputs=6 * edf[DEPLOYED_EXP["ridge_alpha"]])
    print(f"ridge: {lab.X.shape[1]} dims x 6 outputs = {lab.X.shape[1]*6:,} raw coefficients")
    print(f"       nnz/row = {Xt.nnz/Xt.shape[0]:.1f}")
    for a, v in edf.items():
        print(f"       alpha={a:<7} edf/output = {v:8.1f}  (n_fit=1760)")
    print(f"       DEPLOYED alpha=10 -> edf 6 x {edf[10.0]:.1f} = {6*edf[10.0]:.0f} "
          f"effective parameters on 1,760 rows")

    # --------------------------------------------------------- legacy edf
    M = lab.legacy_raw[tr]
    Z = (M - M.mean(0)) / np.where(M.std(0) > 1e-12, M.std(0), 1.0)
    lam2 = np.clip(np.linalg.eigvalsh(Z.T @ Z), 0, None)
    ledf = {a: float((lam2 / (lam2 + a)).sum()) for a in (1.0, 10.0, 100.0, 1000.0)}
    rep["legacy"] = dict(n_coef=256 * 6, edf_per_output=ledf,
                         alpha=DEPLOYED_EXP["legacy_alpha"],
                         edf_total=6 * ledf[100.0])
    print(f"legacy: 256 dims x 6 = 1,536 raw; alpha=100 -> edf/output {ledf[100.0]:.1f} "
          f"-> {6*ledf[100.0]:.0f} effective")

    # ------------------------------------------------------------- GBM size
    exp = dict(DEPLOYED_EXP)
    gp = _gbm_params(exp)
    tg = lab.targets
    dt = lab.delta_targets
    knn_fit, knn_hold, fam_fit, fam_hold = lab._knn_family(tr, lab.dev_idx, tg)
    head = lab.fit_legacy(tr, exp["legacy_alpha"])
    leg_fit = lab.predict_legacy(head, tr)
    ridge = Ridge(alpha=exp["ridge_alpha"], solver="sparse_cg").fit(lab.X[tr], tg[tr])
    inner = np.random.default_rng(0).integers(0, 5, size=len(tr))
    oof = np.zeros((len(tr), 6))
    for k in range(5):
        m = Ridge(alpha=exp["ridge_alpha"], solver="sparse_cg").fit(lab.X[tr[inner != k]], tg[tr[inner != k]])
        oof[inner == k] = m.predict(lab.X[tr[inner == k]])
    oof[:, :3] = np.clip(oof[:, :3], 0, 1)
    Xf = np.hstack([lab.dense[tr], lab.fam_onehot[tr], leg_fit, oof, knn_fit])
    print(f"meta feature block: {Xf.shape[1]} columns, {Xf.shape[0]} rows")

    def leaves(model):
        return int(sum(p[0].get_n_leaf_nodes() for p in model._predictors))

    tot_leaves = 0; heads = []
    for k in range(6):
        m = HistGradientBoostingRegressor(**gp).fit(Xf, tg[tr, k])
        L = leaves(m); tot_leaves += L
        heads.append(dict(kind="reg", k=k, iters=int(m.n_iter_), leaves=L))
    for k in range(2):
        m = HistGradientBoostingRegressor(**gp).fit(Xf, dt[tr, k])
        L = leaves(m); tot_leaves += L
        heads.append(dict(kind="gain", k=k, iters=int(m.n_iter_), leaves=L))
    for mi in range(3):
        for th in ORDINAL_THRESHOLDS:
            y = (tg[tr, mi] >= th).astype(int)
            if y.min() == y.max():
                heads.append(dict(kind="ord", k=mi, th=th, iters=0, leaves=0)); continue
            m = HistGradientBoostingClassifier(**gp).fit(Xf, y)
            L = leaves(m); tot_leaves += L
            heads.append(dict(kind="ord", k=mi, th=th, iters=int(m.n_iter_), leaves=L))
    tc = np.exp(tg[:, 3:6])
    for g, (a, b) in enumerate([(0, 1), (1, 2)]):
        ds = tg[:, b] - tg[:, a]; dc = tc[:, b] - tc[:, a]
        fl = max(float(np.quantile(dc[tr], 0.05)), 1e-9)
        eff = ds / np.maximum(dc, fl)
        r = rankdata(eff[tr], method="average") / max(len(tr) - 1, 1)
        m = HistGradientBoostingRegressor(**gp).fit(Xf, r)
        L = leaves(m); tot_leaves += L
        heads.append(dict(kind="rank", k=g, iters=int(m.n_iter_), leaves=L))
    rep["gbm"] = dict(n_heads=len(heads), total_leaves=tot_leaves, heads=heads,
                      n_train=int(len(tr)))
    print(f"GBM: {len(heads)} heads, {tot_leaves:,} leaves fitted on {len(tr)} rows "
          f"= {tot_leaves/len(tr):.1f} leaves per training episode")
    for h in heads:
        print(f"    {h['kind']:5s} k={h['k']} it={h['iters']:4d} leaves={h['leaves']:6d}")

    # ----------------------------------------------------- inventory + bias
    inv = [
        ("ridge on 16,414 hashed dims", 16414 * 6, 6 * edf[10.0]),
        ("legacy 256-bin hash-regex", 256 * 6, 6 * ledf[100.0]),
        ("meta GBM, 22 heads", tot_leaves, tot_leaves),
        ("family mean table (9 x 6)", 54, 54),
        ("kNN stored targets (1,760 x 6)", 1760 * 6, 0.0),
        ("post-hoc constants", 8, 8),
        ("safety ratios", 3, 3),
    ]
    rep["inventory"] = [dict(name=a, raw=b, edf=c) for a, b, c in inv]
    print("\n| block | raw params | effective df |")
    for a, b, c in inv:
        print(f"| {a:32s} | {b:10,} | {c:12.1f} |")
    print(f"| {'TOTAL':32s} | {sum(b for _, b, _ in inv):10,} | "
          f"{sum(c for _, _, c in inv):12.1f} |   on 1,760 train episodes")

    # ------------------------------------ exact invariance of the level channel
    cv, arr = B.stage(lab, DEPLOYED_EXP, tag="base")
    print("\nlevel-channel invariance (no clipping):")

    def t_shift(lab_, a, ps, pc, tier):
        rs = np.random.default_rng(1).normal(0, 0.2, size=ps.shape[0])
        return ps + rs[:, None], pc

    r0 = B.run(lab, cv, arr, None, label="base", verbose=False)
    r1 = B.run(lab, cv, arr, None, transform=t_shift, label="+N(0,0.2) per-item level",
               verbose=False)
    print(f"    base  EV={r0['EV']:.6f} dev={r0['dev']:.6f}")
    print(f"    +lvl  EV={r1['EV']:.6f} dev={r1['dev']:.6f}   "
          f"dEV={r1['EV']-r0['EV']:+.8f} ddev={r1['dev']-r0['dev']:+.8f}")
    rep["level_invariance"] = dict(base=r0["EV"], shifted=r1["EV"],
                                   d_ev=r1["EV"] - r0["EV"], d_dev=r1["dev"] - r0["dev"])

    # ---------------------------------------- the real risk scale on dev
    print("\nbust probability of the DEPLOYED-cfg dev picks under an 880-item dev bootstrap:")
    idx = arr["idx"]; r = np.arange(len(idx))
    rng = np.random.default_rng(11); nb = 4000
    detail = {}
    picks = {}
    for t in TIERS:
        ps, pc = lab.compose(arr, DEPLOYED_CFG, t)
        picks[t] = lab.allocate(ps, pc, MULTS[t], r0["safety"][t])
    fin = np.zeros(nb); busts = {t: 0 for t in TIERS}
    for b in range(nb):
        s = rng.integers(0, len(idx), size=len(idx))
        v = 0.0
        for t in TIERS:
            p = picks[t]
            base = lab.true_c[idx][:, 0][s].sum()
            ratio = lab.true_c[idx][r, p][s].sum() / base
            if ratio > MULTS[t]:
                busts[t] += 1
            else:
                v += W[t] * lab.true_s[idx][r, p][s].mean()
        fin[b] = v
    for t in TIERS:
        detail[t] = dict(bust=busts[t] / nb, safety=r0["safety"][t])
    sf = "/".join(f"{r0['safety'][t]:.3f}" for t in TIERS)
    bb = "/".join(f"{busts[t]/nb*100:.1f}" for t in TIERS)
    print(f"    safety {sf}  dev-bootstrap bust% {bb}")
    print(f"    dev final: point {r0['dev']:.6f}  bootstrap mean {fin.mean():.6f} "
          f"sd {fin.std():.6f}  p5 {np.quantile(fin,0.05):.6f}  p95 {np.quantile(fin,0.95):.6f}")
    rep["dev_risk"] = dict(tiers=detail, mean=float(fin.mean()), sd=float(fin.std()),
                           p5=float(np.quantile(fin, 0.05)), p95=float(np.quantile(fin, 0.95)),
                           point=r0["dev"])

    OUT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
