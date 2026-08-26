# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a03: what each learned component actually achieves, and where the fit fails.

  parts    ridge-alone vs legacy-alone vs 0.9 blend vs deployed E43 preds
  decouple 2-D (alpha_score x alpha_cost) sweep -> is high alpha helping the
           score head or the cost head?
  family   per-family dev corr / RMSE for all 6 columns
  bv       bias / variance / binomial-noise decomposition of the score MSE
  bins     hash-bin capacity by exact fold-down of the 8192+8192 blocks
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
CACHE = Path(r"C:\Users\PJ05\AppData\Local\Temp\claude\C--portable-skt-LLM1-LLM-ROUTE-0-7000\377d7fd8-9983-4bec-bb08-163cc405f7a3\scratchpad")

import importlib.util  # noqa: E402

import labdata  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "a03curve", Path(__file__).with_name("a03-train-failure_curve.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)
load, grams, ridge_dual, metrics, to_alloc, best_final = (
    _c.load, _c.grams, _c.ridge_dual, _c.metrics, _c.to_alloc, _c.best_final)

MODELS = ("light", "mid", "k1")
NDENSE = 30
WB = CB = 8192


def ridge_split_alpha(G, Gd, Y, idx, a_score, a_cost):
    ps, _ = ridge_dual(G, Gd, Y, idx, a_score)
    pc, _ = ridge_dual(G, Gd, Y, idx, a_cost)
    return np.column_stack([ps[:, :3], pc[:, 3:]])


def summary(name, pred, Ydv, dv):
    mt = metrics(pred, Ydv)
    f, s, per = best_final(pred, dv)
    print(f"{name:26s} corr {mt[0][0]:.3f}/{mt[1][0]:.3f}/{mt[2][0]:.3f}  "
          f"lcRMSE {mt[3][1]:.3f}/{mt[4][1]:.3f}/{mt[5][1]:.3f}  "
          f"final {f:.4f}  (fast {per['fast']:.4f} bal {per['balanced']:.4f} prem {per['premium']:.4f})"
          f"  safety {s['fast']:.2f}/{s['balanced']:.2f}/{s['premium']:.2f}")
    return f


def parts(G, Gd, m, dv):
    Y, Ydv, Ldv = m["Ytr"], m["Ydv"], m["Ldv"]
    idx = np.arange(len(Y))
    print("== components, each scored on dev with per-tier dev-tuned safety (optimistic)")
    for a in (3.0, 10.0, 100.0, 1000.0):
        summary(f"ridge16414 alpha={a:g}", ridge_dual(G, Gd, Y, idx, a)[0], Ydv, dv)
    summary("legacy 270-bin (alpha100)", Ldv, Ydv, dv)
    for w in (0.5, 0.75, 0.9, 1.0):
        p = (1 - w) * ridge_dual(G, Gd, Y, idx, 10.0)[0] + w * Ldv
        summary(f"blend legacy w={w:g}", p, Ydv, dv)
    d = np.load(ROOT / "reports/lab/dev_preds_e43.npz")
    dep = np.column_stack([d["score_fast"], np.log(d["cost_fast"])])
    print("-- deployed E43 pipeline (fast-tier blend), same scoring:")
    summary("deployed E43 fast preds", dep, Ydv, dv)
    for t in ("fast", "balanced", "premium"):
        p = np.column_stack([d[f"score_{t}"], np.log(d[f"cost_{t}"])])
        mt = metrics(p, Ydv)
        print(f"   {t:9s} corr {mt[0][0]:.3f}/{mt[1][0]:.3f}/{mt[2][0]:.3f}  "
              f"lcRMSE {mt[3][1]:.3f}/{mt[4][1]:.3f}/{mt[5][1]:.3f}")


def decouple(G, Gd, m, dv):
    Y, Ydv = m["Ytr"], m["Ydv"]
    idx = np.arange(len(Y))
    alphas = (1.0, 3.0, 10.0, 100.0, 1000.0)
    cache = {a: ridge_dual(G, Gd, Y, idx, a)[0] for a in alphas}
    print("== final score, alpha_score (rows) x alpha_cost (cols), dev-tuned safety")
    print("             " + "".join(f"c={a:<8g}" for a in alphas))
    for a_s in alphas:
        row = []
        for a_c in alphas:
            p = np.column_stack([cache[a_s][:, :3], cache[a_c][:, 3:]])
            f, _, _ = best_final(p, dv)
            row.append(f)
        print(f"  s={a_s:<8g} " + "".join(f"{v:.4f}   " for v in row))
    print("== same grid: mean dev score corr (rows) / mean logcost RMSE (cols)")
    for a in alphas:
        mt = metrics(cache[a], Ydv)
        print(f"  alpha {a:<8g} corr {np.mean([mt[j][0] for j in range(3)]):.4f} "
              f" lcRMSE {np.mean([mt[j][1] for j in range(3,6)]):.4f}")


def family(G, Gd, m, dv):
    Y, Ydv, fdv = m["Ytr"], m["Ydv"], m["fdv"]
    ng = m["ngdv"]
    idx = np.arange(len(Y))
    pred = ridge_dual(G, Gd, Y, idx, 10.0)[0]
    d = np.load(ROOT / "reports/lab/dev_preds_e43.npz")
    dep = np.column_stack([d["score_fast"], np.log(d["cost_fast"])])
    fams = sorted(set(fdv.tolist()))
    print("== per-family dev metrics (ridge alpha10 trained on train-1760)")
    print(f"{'family':16s} {'n':>4s} | corr s0/s1/s2        | RMSE s0/s1/s2       "
          f"| noise-RMSE s0/s1/s2  | lcRMSE c0/c1/c2")
    for f in fams:
        k = fdv == f
        n = int(k.sum())
        cs, rs, ns, lc = [], [], [], []
        for j in range(3):
            p, y = pred[k, j], Ydv[k, j]
            cs.append(np.corrcoef(p, y)[0, 1] if y.std() > 1e-9 else np.nan)
            rs.append(np.sqrt(np.mean((p - y) ** 2)))
            pq = y * (1 - y) * ng[k, j] / (ng[k, j] - 1)
            ns.append(np.sqrt(max(np.mean(pq / ng[k, j]), 0.0)))
            lc.append(np.sqrt(np.mean((pred[k, 3 + j] - Ydv[k, 3 + j]) ** 2)))
        print(f"{f:16s} {n:4d} | " + "/".join(f"{v:5.2f}" for v in cs) + " | "
              + "/".join(f"{v:.3f}" for v in rs) + " | "
              + "/".join(f"{v:.3f}" for v in ns) + " | "
              + "/".join(f"{v:.3f}" for v in lc))
    print("== same, deployed E43 pipeline (fast tier)")
    for f in fams:
        k = fdv == f
        cs = [np.corrcoef(dep[k, j], Ydv[k, j])[0, 1] if Ydv[k, j].std() > 1e-9 else np.nan
              for j in range(3)]
        rs = [np.sqrt(np.mean((dep[k, j] - Ydv[k, j]) ** 2)) for j in range(3)]
        lc = [np.sqrt(np.mean((dep[k, 3 + j] - Ydv[k, 3 + j]) ** 2)) for j in range(3)]
        print(f"{f:16s} {int(k.sum()):4d} | " + "/".join(f"{v:5.2f}" for v in cs) + " | "
              + "/".join(f"{v:.3f}" for v in rs) + " |                      | "
              + "/".join(f"{v:.3f}" for v in lc))


def bv(G, Gd, m, dv, B=40, alpha=10.0):
    """Bias^2 / variance / binomial-noise split of dev score MSE."""
    Y, Ydv, ng = m["Ytr"], m["Ydv"], m["ngdv"]
    n = len(Y)
    preds = []
    rng = np.random.default_rng(7)
    for b in range(B):
        idx = rng.choice(n, size=n, replace=True)
        idx = np.unique(idx)          # dual ridge needs distinct rows
        preds.append(ridge_dual(G, Gd, Y, idx, alpha)[0])
    P = np.stack(preds)               # (B, ndev, 6)
    pbar = P.mean(0)
    var = P.var(0).mean(0)
    print(f"== bias/variance/noise on dev, {B} bootstrap refits, alpha={alpha}")
    print(" col      MSE      noise    bias^2   variance   (bias^2+var+noise)")
    for j in range(6):
        mse = float(np.mean((preds[0][:, j] - Ydv[:, j]) ** 2))
        if j < 3:
            pq = Ydv[:, j] * (1 - Ydv[:, j]) * ng[:, j] / (ng[:, j] - 1)
            noise = float(np.mean(pq / ng[:, j]))
        else:
            noise = 0.0
        b2 = float(np.mean((pbar[:, j] - Ydv[:, j]) ** 2)) - noise
        print(f" {j}     {mse:.5f}  {noise:.5f}  {b2:.5f}  {var[j]:.5f}    "
              f"{b2+var[j]+noise:.5f}")
    print(" (col 0..2 = score light/mid/k1, 3..5 = log-cost)")


def _fold(X, bins_w, bins_c):
    """Exact simulation of a smaller hash space: fold columns, renormalise blocks."""
    X = X.tocsc()
    D = X[:, :NDENSE]
    Wm = X[:, NDENSE:NDENSE + WB]
    Cm = X[:, NDENSE + WB:]
    out = [D]
    for M, full, small in ((Wm, WB, bins_w), (Cm, CB, bins_c)):
        if small == full:
            out.append(M)
            continue
        F = sparse.csr_matrix(
            (np.ones(full), (np.arange(full), np.arange(full) % small)), shape=(full, small))
        R = (M @ F).tocsr()
        nrm = np.sqrt(np.asarray(R.multiply(R).sum(axis=1)).ravel())
        nrm[nrm == 0] = 1.0
        R = sparse.diags(1.0 / nrm) @ R
        out.append(R)
    return sparse.hstack(out).tocsr()


def bins(m, dv):
    Xtr = sparse.load_npz(CACHE / "Xtr.npz").astype(np.float64)
    Xdv = sparse.load_npz(CACHE / "Xdv.npz").astype(np.float64)
    Y, Ydv = m["Ytr"], m["Ydv"]
    idx = np.arange(Xtr.shape[0])
    print("== hash-space capacity (exact fold-down, alpha chosen per config from {3,10,30})")
    print(" word/char bins   dim    corr(avg)  lcRMSE(avg)  final")
    for bw, bc in ((256, 256), (1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192),
                   (8192, 256), (256, 8192)):
        A = _fold(Xtr, bw, bc)
        Bm = _fold(Xdv, bw, bc)
        G = (A @ A.T).toarray()
        Gd = (Bm @ A.T).toarray()
        best = None
        for a in (3.0, 10.0, 30.0):
            p = ridge_dual(G, Gd, Y, idx, a)[0]
            mt = metrics(p, Ydv)
            c = np.mean([mt[j][0] for j in range(3)])
            if best is None or c > best[0]:
                best = (c, np.mean([mt[j][1] for j in range(3, 6)]), a, p)
        f, _, _ = best_final(best[3], dv)
        print(f" {bw:5d}/{bc:<5d} {NDENSE+bw+bc:7d}  {best[0]:.4f}     {best[1]:.4f} "
              f"      {f:.4f}   (alpha {best[2]:g})")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "parts"
    Xtr, Xdv, m = load()
    dv = labdata.load_split("dev")
    if what in ("bins",):
        bins(m, dv)
    else:
        G, Gd = grams(Xtr, Xdv)
        {"parts": parts, "decouple": decouple, "family": family, "bv": bv}[what](G, Gd, m, dv)
