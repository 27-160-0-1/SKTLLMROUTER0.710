# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a03: learning curves + noise ceiling for the linear (ridge) head.

Exact ridge in the dual (n << d): identical solution to the deployed
cupy-LSMR fit with damp=sqrt(alpha), but 200x faster and deterministic.

Usage:  python a03-train-failure_curve.py [verify|curve|alpha|ceiling|all]
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

import labdata  # noqa: E402

MODELS = ("light", "mid", "k1")


# ------------------------------------------------------------------ data
def load():
    Xtr = sparse.load_npz(CACHE / "Xtr.npz").astype(np.float64)
    Xdv = sparse.load_npz(CACHE / "Xdv.npz").astype(np.float64)
    m = np.load(CACHE / "meta.npz", allow_pickle=True)
    return Xtr, Xdv, m


def grams(Xtr, Xdv):
    f = CACHE / "grams.npz"
    if f.exists():
        d = np.load(f)
        return d["G"], d["Gd"]
    G = (Xtr @ Xtr.T).toarray()
    Gd = (Xdv @ Xtr.T).toarray()
    np.savez(f, G=G, Gd=Gd)
    return G, Gd


# ------------------------------------------------------------------ ridge
def ridge_dual(G, Gd, Y, idx, alpha):
    """Exact ridge fit on rows `idx`; returns dev predictions (n_dev, k)."""
    n = len(idx)
    Gs = G[np.ix_(idx, idx)]
    r = Gs.sum(axis=1)
    tot = r.sum()
    Kc = Gs - r[:, None] / n - r[None, :] / n + tot / n**2
    Gds = Gd[:, idx]
    rd = Gds.sum(axis=1)
    Kd = Gds - rd[:, None] / n - r[None, :] / n + tot / n**2
    Ys = Y[idx]
    yb = Ys.mean(axis=0)
    A = Kc + alpha * np.eye(n)
    a = np.linalg.solve(A, Ys - yb)
    return Kd @ a + yb, (Kc @ a + yb)  # dev pred, in-sample train pred


def metrics(pred, Y):
    """corr and rmse for 6 columns."""
    out = {}
    for j in range(6):
        p, y = pred[:, j], Y[:, j]
        c = np.corrcoef(p, y)[0, 1]
        out[j] = (float(c), float(np.sqrt(np.mean((p - y) ** 2))))
    return out


def to_alloc(pred):
    ps = np.clip(pred[:, :3], 0.0, 1.0)
    pc = np.exp(np.clip(pred[:, 3:], -50, 50))
    pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
    pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
    return ps, pc


def best_final(pred, dv):
    ps, pc = to_alloc(pred)
    best = -1.0
    bs = None
    grid = np.arange(0.60, 1.001, 0.01)
    per = {}
    for t in labdata.TIERS:
        bt, bsr = -1.0, None
        for s in grid:
            r = labdata.tier_result(ps, pc, dv, t, float(s))
            if r["tier_score"] > bt:
                bt, bsr = r["tier_score"], float(s)
        per[t] = (bt, bsr)
    best = sum(labdata.TIER_WEIGHT[t] * per[t][0] for t in labdata.TIERS)
    bs = {t: per[t][1] for t in labdata.TIERS}
    return float(best), bs, {t: per[t][0] for t in labdata.TIERS}


# ------------------------------------------------------------------ steps
def verify(Xtr, Xdv, m, G, Gd):
    """Compare the exact dual ridge against the deployed train-only GPU artifact."""
    import json
    import math
    art = json.load(open(ROOT / "reports/holdout_local/learned-router.v1.json", encoding="utf-8"))
    ids = ["ax31-light", "ax31", "axk1-think"]
    heads = [art["score_heads"][i] for i in ids] + [art["log_cost_heads"][i] for i in ids]
    W = np.asarray([h["coefficients"] for h in heads]).T
    b = np.asarray([h["intercept"] for h in heads])
    gpu = np.asarray(Xdv @ W) + b
    Y = m["Ytr"]
    idx = np.arange(Xtr.shape[0])
    mine, _ = ridge_dual(G, Gd, Y, idx, 10.0)
    print("== verify: exact dual ridge vs deployed cupy-LSMR heads (alpha=10, train-only)")
    print("col   corr(mine,gpu)   rmse(mine-gpu)   sd(gpu)")
    for j in range(6):
        c = np.corrcoef(mine[:, j], gpu[:, j])[0, 1]
        print(f"{j}     {c:.6f}        {np.sqrt(np.mean((mine[:,j]-gpu[:,j])**2)):.5f}"
              f"        {gpu[:,j].std():.5f}")
    dvsplit = labdata.load_split("dev")
    for nm, P in (("dual", mine), ("gpu", gpu)):
        mt = metrics(P, m["Ydv"])
        print(nm, "score corr", [round(mt[j][0], 4) for j in range(3)],
              "logcost rmse", [round(mt[j][1], 4) for j in range(3, 6)])
        f, s, per = best_final(P, dvsplit)
        print("   best-safety final", round(f, 4), s, {k: round(v, 4) for k, v in per.items()})


def ceiling(m):
    """Irreducible binomial noise in the score labels."""
    print("== score-label noise decomposition (dev, then train)")
    for split, Y, ng in (("train", m["Ytr"], m["ngtr"]), ("dev", m["Ydv"], m["ngdv"])):
        print(f"-- {split}")
        print(" model  var(s)    E[p(1-p)/n]=noise  var(p)   ceiling corr  ngen mean")
        for j in range(3):
            s = Y[:, j]
            n = ng[:, j]
            vs = s.var()
            # unbiased per-item p(1-p) = s(1-s)*n/(n-1)
            pq = s * (1 - s) * n / (n - 1)
            noise = float(np.mean(pq / n))
            vp = vs - noise
            print(f" {MODELS[j]:5s}  {vs:.5f}   {noise:.5f}            {vp:.5f}  "
                  f"{np.sqrt(max(vp,0)/vs):.4f}        {n.mean():.2f}")


def curve(Xtr, Xdv, m, G, Gd, alpha=10.0, sizes=(200, 400, 800, 1200, 1760), seeds=(0, 1, 2, 3, 4)):
    Y, Ydv = m["Ytr"], m["Ydv"]
    dv = labdata.load_split("dev")
    ntr = Xtr.shape[0]
    print(f"== learning curve (exact ridge alpha={alpha}, dev held out)")
    print("  n     seeds  corr_light corr_mid corr_k1 | rmse_lc_light  mid    k1  | final(best safety)")
    rows = []
    for n in sizes:
        ss = seeds if n < ntr else (0,)
        cs, rs, fs = [], [], []
        for sd in ss:
            rng = np.random.default_rng(1000 + sd)
            idx = rng.permutation(ntr)[:n] if n < ntr else np.arange(ntr)
            pred, _ = ridge_dual(G, Gd, Y, idx, alpha)
            mt = metrics(pred, Ydv)
            cs.append([mt[j][0] for j in range(3)])
            rs.append([mt[j][1] for j in range(3, 6)])
            f, _, _ = best_final(pred, dv)
            fs.append(f)
        c = np.mean(cs, 0)
        r = np.mean(rs, 0)
        print(f"  {n:5d} {len(ss):3d}    {c[0]:.4f}    {c[1]:.4f}   {c[2]:.4f}  |  "
              f"{r[0]:.4f}      {r[1]:.4f} {r[2]:.4f} |  {np.mean(fs):.4f} (sd {np.std(fs):.4f})")
        rows.append((n, c, r, np.mean(fs), np.std(fs)))
    return rows


def alpha_path(Xtr, Xdv, m, G, Gd, sizes=(440, 880, 1760)):
    Y, Ydv = m["Ytr"], m["Ydv"]
    dv = labdata.load_split("dev")
    ntr = Xtr.shape[0]
    print("== ridge alpha path (dev corr avg over 3 score cols / logcost rmse avg / final)")
    for n in sizes:
        seeds = (0, 1, 2) if n < ntr else (0,)
        print(f"-- n={n}")
        for alpha in (0.3, 1, 3, 10, 30, 100, 300, 1000, 3000):
            cs, rs, fs, tr_r = [], [], [], []
            for sd in seeds:
                rng = np.random.default_rng(1000 + sd)
                idx = rng.permutation(ntr)[:n] if n < ntr else np.arange(ntr)
                pred, ins = ridge_dual(G, Gd, Y, idx, float(alpha))
                mt = metrics(pred, Ydv)
                cs.append(np.mean([mt[j][0] for j in range(3)]))
                rs.append(np.mean([mt[j][1] for j in range(3, 6)]))
                mi = metrics(ins, Y[idx])
                tr_r.append(np.mean([mi[j][1] for j in range(3)]))
                f, _, _ = best_final(pred, dv)
                fs.append(f)
            mt2 = metrics(pred, Ydv)
            print(f"  alpha {alpha:7.1f}  devcorr {np.mean(cs):.4f}  devrmse_lc {np.mean(rs):.4f} "
                  f" trainRMSE_score {np.mean(tr_r):.4f}  final {np.mean(fs):.4f}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    Xtr, Xdv, m = load()
    G, Gd = grams(Xtr, Xdv)
    if what in ("verify", "all"):
        verify(Xtr, Xdv, m, G, Gd)
    if what in ("ceiling", "all"):
        ceiling(m)
    if what in ("curve", "all"):
        curve(Xtr, Xdv, m, G, Gd)
    if what in ("alpha", "all"):
        alpha_path(Xtr, Xdv, m, G, Gd)
