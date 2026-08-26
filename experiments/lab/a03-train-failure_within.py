# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a03: is the learned signal anything more than family identification?

  split     between-family vs within-family decomposition of the fit
  wcurve    learning curve of the WITHIN-family correlation
  dup       nearest-train-neighbour similarity per family (aime sanity check)
  extrap    power-law extrapolation of the learning curve
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
CACHE = Path(r"C:\Users\PJ05\AppData\Local\Temp\claude\C--portable-skt-LLM1-LLM-ROUTE-0-7000\377d7fd8-9983-4bec-bb08-163cc405f7a3\scratchpad")

import labdata  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "a03curve", Path(__file__).with_name("a03-train-failure_curve.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

MODELS = ("light", "mid", "k1")


def _demean(v, fam):
    out = v.astype(float).copy()
    for f in set(fam.tolist()):
        k = fam == f
        out[k] -= out[k].mean()
    return out


def wcorr(pred, y, fam):
    a, b = _demean(pred, fam), _demean(y, fam)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def split(m, G, Gd):
    Y, Ydv, ftr, fdv, ng = m["Ytr"], m["Ydv"], m["ftr"], m["fdv"], m["ngdv"]
    idx = np.arange(len(Y))
    pred = _c.ridge_dual(G, Gd, Y, idx, 10.0)[0]
    d = np.load(ROOT / "reports/lab/dev_preds_e43.npz")
    dep = np.column_stack([d["score_fast"], np.log(d["cost_fast"])])
    # family-mean-only predictor (train means, applied to dev)
    fam_pred = np.zeros_like(Ydv)
    for f in set(ftr.tolist()):
        fam_pred[fdv == f] = Y[ftr == f].mean(axis=0)
    print("== total vs within-family correlation on dev (family = regex classifier)")
    print(f"{'column':10s} {'ridge tot':>9s} {'ridge wthn':>10s} {'E43 tot':>8s} "
          f"{'E43 wthn':>9s} {'fammean tot':>11s} | {'ceiling tot':>11s} {'ceiling wthn':>12s}")
    for j in range(6):
        y = Ydv[:, j]
        if j < 3:
            pq = y * (1 - y) * ng[:, j] / (ng[:, j] - 1)
            noise = float(np.mean(pq / ng[:, j]))
            vt = y.var()
            yw = _demean(y, fdv)
            vw = float(np.mean(yw ** 2))
            ct = np.sqrt(max(vt - noise, 0) / vt)
            cw = np.sqrt(max(vw - noise, 0) / vw)
        else:
            ct = cw = np.nan
        print(f"{j:<10d} {np.corrcoef(pred[:,j],y)[0,1]:9.3f} {wcorr(pred[:,j],y,fdv):10.3f} "
              f"{np.corrcoef(dep[:,j],y)[0,1]:8.3f} {wcorr(dep[:,j],y,fdv):9.3f} "
              f"{np.corrcoef(fam_pred[:,j],y)[0,1]:11.3f} | {ct:11.3f} {cw:12.3f}")
    print("\n== per-family latent variance of the SCORE label (dev)")
    print(f"{'family':16s} {'n':>4s} " + " ".join(
        f"{m2:>26s}" for m2 in ("light var(s)/noise/var(p)", "mid var(s)/noise/var(p)",
                                "k1  var(s)/noise/var(p)")))
    for f in sorted(set(fdv.tolist())):
        k = fdv == f
        cells = []
        for j in range(3):
            y = Ydv[k, j]
            n = ng[k, j]
            vs = float(y.var())
            noise = float(np.mean(y * (1 - y) * n / (n - 1) / n))
            cells.append(f"{vs:.4f}/{noise:.4f}/{max(vs-noise,0):.4f}      ")
        print(f"{f:16s} {int(k.sum()):4d} " + " ".join(cells))
    print("\n== pooled within-family score variance vs binomial noise (dev)")
    for j in range(3):
        y = Ydv[:, j]
        yw = _demean(y, fdv)
        vw = float(np.mean(yw ** 2))
        noise = float(np.mean(y * (1 - y) * ng[:, j] / (ng[:, j] - 1) / ng[:, j]))
        print(f"  {MODELS[j]:5s} within var(s) {vw:.5f}  binomial noise {noise:.5f}  "
              f"=> within var(p) {max(vw-noise,0):.5f}  ({100*max(vw-noise,0)/vw:.0f}% learnable), "
              f"ceiling corr {np.sqrt(max(vw-noise,0)/vw):.3f}")


def wcurve(m, G, Gd, sizes=(200, 400, 800, 1200, 1760), seeds=(0, 1, 2, 3, 4)):
    Y, Ydv, fdv = m["Ytr"], m["Ydv"], m["fdv"]
    ntr = len(Y)
    print("== learning curve of the WITHIN-family correlation (alpha 10)")
    print("   n    total corr l/m/k1        within corr l/m/k1        within lcRMSE l/m/k1")
    for n in sizes:
        ss = seeds if n < ntr else (0,)
        tc, wc, wl = [], [], []
        for sd in ss:
            rng = np.random.default_rng(1000 + sd)
            idx = rng.permutation(ntr)[:n] if n < ntr else np.arange(ntr)
            p = _c.ridge_dual(G, Gd, Y, idx, 10.0)[0]
            tc.append([np.corrcoef(p[:, j], Ydv[:, j])[0, 1] for j in range(3)])
            wc.append([wcorr(p[:, j], Ydv[:, j], fdv) for j in range(3)])
            wl.append([np.sqrt(np.mean(_demean(p[:, 3 + j] - Ydv[:, 3 + j], fdv) ** 2))
                       for j in range(3)])
        t, w, l = np.mean(tc, 0), np.mean(wc, 0), np.mean(wl, 0)
        print(f" {n:5d}   {t[0]:.3f}/{t[1]:.3f}/{t[2]:.3f}           "
              f"{w[0]:.3f}/{w[1]:.3f}/{w[2]:.3f}           {l[0]:.3f}/{l[1]:.3f}/{l[2]:.3f}")


def dup(m):
    Xtr = sparse.load_npz(CACHE / "Xtr.npz").astype(np.float64)
    Xdv = sparse.load_npz(CACHE / "Xdv.npz").astype(np.float64)
    ftr, fdv = m["ftr"], m["fdv"]
    # char block only, already L2-normalised per row -> dot product = cosine
    C = slice(30 + 8192, 30 + 16384)
    A = Xtr[:, C]
    B = Xdv[:, C]
    S = np.asarray((B @ A.T).todense())
    top = S.max(axis=1)
    print("== dev->train nearest-neighbour cosine on the char n-gram block")
    print(f"{'family':16s} {'n':>4s} {'mean top1':>10s} {'median':>8s} {'p90':>7s} "
          f"{'frac>0.9':>9s} {'frac>0.98':>10s}   train n")
    for f in sorted(set(fdv.tolist())):
        k = fdv == f
        t = top[k]
        print(f"{f:16s} {int(k.sum()):4d} {t.mean():10.3f} {np.median(t):8.3f} "
              f"{np.quantile(t,0.9):7.3f} {(t>0.9).mean():9.2f} {(t>0.98).mean():10.2f}   "
              f"{int((ftr==f).sum()):5d}")


def extrap(m, G, Gd):
    """Fit corr(n) = c_inf - a*n^-b on the measured learning curve."""
    from scipy.optimize import curve_fit
    Y, Ydv = m["Ytr"], m["Ydv"]
    ntr = len(Y)
    sizes = np.array([200, 300, 440, 600, 880, 1200, 1500, 1760])
    seeds = (0, 1, 2, 3, 4, 5, 6, 7)
    cur = []
    for n in sizes:
        ss = seeds if n < ntr else (0,)
        v = []
        for sd in ss:
            rng = np.random.default_rng(1000 + sd)
            idx = rng.permutation(ntr)[:n] if n < ntr else np.arange(ntr)
            p = _c.ridge_dual(G, Gd, Y, idx, 10.0)[0]
            v.append([np.corrcoef(p[:, j], Ydv[:, j])[0, 1] for j in range(3)])
        cur.append(np.mean(v, 0))
    cur = np.asarray(cur)
    print("== measured corr(n), 8 sizes x 8 seeds, alpha 10")
    for i, n in enumerate(sizes):
        print(f"  n={n:5d}  {cur[i,0]:.4f} {cur[i,1]:.4f} {cur[i,2]:.4f}")

    def f(n, c, a, b):
        return c - a * n ** (-b)
    print("== power-law fit corr(n) = c_inf - a n^-b  and extrapolation")
    for j in range(3):
        try:
            popt, _ = curve_fit(f, sizes, cur[:, j], p0=[0.6, 2.0, 0.5],
                                bounds=([0.2, 0.0, 0.05], [1.0, 50.0, 2.0]), maxfev=20000)
        except Exception as exc:  # noqa: BLE001
            print(" fit failed", exc)
            continue
        pred = {k: f(k, *popt) for k in (2640, 5280, 10560, 26400, 100000)}
        print(f"  {MODELS[j]:5s} c_inf={popt[0]:.3f} a={popt[1]:.2f} b={popt[2]:.2f} | "
              + "  ".join(f"n={k}:{v:.3f}" for k, v in pred.items()))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "split"
    Xtr, Xdv, m = _c.load()
    if what == "dup":
        dup(m)
    else:
        G, Gd = _c.grams(Xtr, Xdv)
        {"split": split, "wcurve": wcurve, "extrap": extrap}[what](m, G, Gd)
