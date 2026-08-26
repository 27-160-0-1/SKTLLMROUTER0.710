# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a09: exact piecewise-Lagrangian bootstrap harness.

For a given (pred_score, pred_cost) it enumerates every breakpoint of the
Lagrangian path, so allocation for ANY safety value / ANY bootstrap batch is a
lookup instead of a bisection.  Verified against labdata.tier_result.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result


def path_segments(S: np.ndarray, C: np.ndarray):
    """Return (sel (G,n) int8, lam (G,)) covering every distinct argmax pattern."""
    n = len(S)
    bps = []
    for a in range(3):
        for b in range(a + 1, 3):
            dC = C[:, b] - C[:, a]
            with np.errstate(divide="ignore", invalid="ignore"):
                l = (S[:, b] - S[:, a]) / dC
            l = l[np.isfinite(l) & (l > 0)]
            bps.append(l)
    bp = np.unique(np.concatenate(bps))
    bp = bp[bp < 1e12]
    mids = np.concatenate(([0.0], (bp[:-1] + bp[1:]) / 2.0, [bp[-1] * 1.5 + 1.0])) if len(bp) > 1 else np.array([0.0, 1.0])
    # dedupe selections
    U = S[None, :, :] - mids[:, None, None] * C[None, :, :]
    sel = U.argmax(axis=2).astype(np.int8)          # (G,n)
    keep = np.concatenate(([True], (sel[1:] != sel[:-1]).any(axis=1)))
    return sel[keep], mids[keep]


class LPath:
    def __init__(self, S, C, true_cost, true_score):
        sel, lam = path_segments(S, C)
        idx = np.arange(S.shape[0])
        self.sel = sel
        self.pc = np.take_along_axis(C, sel.astype(np.intp).T, 1).T.astype(np.float64)   # (G,n)
        self.tc = np.take_along_axis(true_cost, sel.astype(np.intp).T, 1).T.astype(np.float64)
        self.ts = np.take_along_axis(true_score, sel.astype(np.intp).T, 1).T.astype(np.float64)
        self.C0 = C[:, 0].astype(np.float64)
        self.T0 = true_cost[:, 0].astype(np.float64)
        # enforce monotone-decreasing predicted total along the path
        self.G = len(lam)

    def batch(self, W):
        """W: (n,B) resample weights.  Returns dict of (G,B) totals + (B,) baselines."""
        return dict(pred=self.pc @ W, true=self.tc @ W, score=self.ts @ W,
                    pl=self.C0 @ W, tl=self.T0 @ W, nb=W.sum(0))


def eval_tier(pth: LPath, W, mult, safety):
    b = pth.batch(W)
    cap = b["pl"] * np.maximum(1.0, mult * safety)
    ok = b["pred"] <= cap[None, :]
    g = ok.argmax(axis=0)
    any_ok = ok.any(axis=0)
    B = W.shape[1]
    j = np.arange(B)
    ratio = b["true"][g, j] / b["tl"]
    score = b["score"][g, j] / b["nb"]
    # fallback: if no segment fits, allocator picks all-light
    ratio = np.where(any_ok, ratio, 1.0)
    score = np.where(any_ok, score, (pth.ts[0] * 0 + 1) @ W / b["nb"] * 0 + (b["score"][-1, j] / b["nb"]))
    passed = ratio <= mult + 1e-15
    return score, ratio, passed


def headroom(pth: LPath, W, mult):
    """max safety whose realised ratio still passes, per batch."""
    b = pth.batch(W)
    okt = (b["true"] / b["tl"][None, :]) <= mult + 1e-15
    g = okt.argmax(axis=0)
    j = np.arange(W.shape[1])
    return b["pred"][g, j] / (mult * b["pl"])


def make_W(n, B, seed, size=None):
    rng = np.random.default_rng(seed)
    size = size or n
    idx = rng.integers(0, n, size=(B, size))
    W = np.zeros((n, B))
    for i in range(B):
        np.add.at(W[:, i], idx[i], 1.0)
    return W


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    dv = load_split("dev"); n = len(dv)
    P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
    Wfull = np.ones((n, 1))
    print("=== harness verification vs labdata.tier_result (full dev) ===")
    for t in TIERS:
        S, C = P[f"score_{t}"], P[f"cost_{t}"]
        pth = LPath(S, C, dv.cost, dv.score)
        print(f" {t}: G={pth.G} segments")
        for sf in (0.80, 0.85, 0.87, 0.89, 0.98, 1.05):
            r = tier_result(S, C, dv, t, sf)
            sc, ra, pa = eval_tier(pth, Wfull, TIER_MULT[t], sf)
            flag = "OK " if (abs(sc[0]-r["score"]) < 1e-9 and abs(ra[0]-r["ratio"]) < 1e-9) else "MISMATCH"
            print(f"   safety={sf:.2f} labdata score={r['score']:.6f} ratio={r['ratio']:.6f} | "
                  f"path {sc[0]:.6f} {ra[0]:.6f}  {flag}")


def eval_cached(b, mult, safety):
    """b = LPath.batch(W) cached; vectorised over the batch dimension."""
    cap = b["pl"] * max(1.0, mult * safety)
    ok = b["pred"] <= cap[None, :]
    g = ok.argmax(axis=0)
    any_ok = ok.any(axis=0)
    j = np.arange(len(g))
    ratio = np.where(any_ok, b["true"][g, j] / b["tl"], 1.0)
    score = np.where(any_ok, b["score"][g, j] / b["nb"], b["score"][-1, j] / b["nb"])
    passed = ratio <= mult + 1e-15
    return score, ratio, passed


def headroom_cached(b, mult):
    okt = (b["true"] / b["tl"][None, :]) <= mult + 1e-15
    g = okt.argmax(axis=0)
    j = np.arange(len(g))
    return b["pred"][g, j] / (mult * b["pl"])
