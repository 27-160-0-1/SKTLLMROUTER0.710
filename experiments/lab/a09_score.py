# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a09: score-side post-processing - monotonicity, gain shrinkage, isotonic, OOD shrinkage."""
from __future__ import annotations
import sys, time
from pathlib import Path as FP
import numpy as np
sys.path.insert(0, str(FP(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, MODEL_IDS
from a09_harness import LPath, eval_cached, headroom_cached, make_W
ROOT = FP(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.similarity import classify_family
from sklearn.isotonic import IsotonicRegression

dv = load_split("dev"); tr = load_split("train"); n = len(dv)
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
fam = np.array([classify_family(t) for t in dv.texts]); fams = sorted(set(fam))
famid = np.array([fams.index(f) for f in fam])
half = np.random.default_rng(0).integers(0, 2, size=n)
GRID = np.round(np.arange(0.55, 1.501, 0.005), 4)
SEEDS = (7, 17, 23); Ws = {s: make_W(n, 400, s) for s in SEEDS}

# ---- OOD signal: top-1 char n-gram cosine similarity to the public train set
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=1 << 15,
                       alternate_sign=False, norm=None)
Xtr = hv.transform([t[:4000] for t in tr.texts]); Xdv = hv.transform([t[:4000] for t in dv.texts])
tf = TfidfTransformer().fit(Xtr)
A = tf.transform(Xtr); Bm = tf.transform(Xdv)
import sklearn.preprocessing as pp
A = pp.normalize(A); Bm = pp.normalize(Bm)
SIM = np.asarray((Bm @ A.T).max(axis=1).todense()).ravel()
print(f"top-1 train similarity: mean {SIM.mean():.3f} p05 {np.quantile(SIM,.05):.3f} "
      f"p50 {np.median(SIM):.3f} p95 {np.quantile(SIM,.95):.3f}")
for j, m in enumerate(MODEL_IDS):
    e = np.abs(P['score_fast'][:, j] - dv.score[:, j])
    lo, hi = SIM < np.median(SIM), SIM >= np.median(SIM)
    print(f"  {m:11s} |score err| low-sim {e[lo].mean():.4f} high-sim {e[hi].mean():.4f} ; "
          f"|log cost err| low {np.abs(np.log(P['cost_fast'][lo,j]/dv.cost[lo,j])).mean():.3f} "
          f"high {np.abs(np.log(P['cost_fast'][hi,j]/dv.cost[hi,j])).mean():.3f}")

# ---------------- score transforms
def s_base(S): return S.copy()

def s_cummax(S):
    return np.maximum.accumulate(S, axis=1)

def s_clipgain(S, floor=0.0):
    out = S.copy()
    for j in (1, 2):
        out[:, j] = out[:, j - 1] + np.maximum(S[:, j] - S[:, j - 1], floor)
    return out

def s_gain_lin(S, mode="lin"):
    """cross-fitted calibration of E[true gain | pred gain]."""
    out = S.copy()
    G = np.column_stack([S[:, 1] - S[:, 0], S[:, 2] - S[:, 1]])
    T = np.column_stack([dv.score[:, 1] - dv.score[:, 0], dv.score[:, 2] - dv.score[:, 1]])
    Gc = G.copy()
    for h in (0, 1):
        a = half != h; b = half == h
        for k in (0, 1):
            if mode == "lin":
                A = np.polyfit(G[a, k], T[a, k], 1)
                Gc[b, k] = A[0] * G[b, k] + A[1]
            else:
                iso = IsotonicRegression(out_of_bounds="clip").fit(G[a, k], T[a, k])
                Gc[b, k] = iso.predict(G[b, k])
    out[:, 1] = S[:, 0] + Gc[:, 0]; out[:, 2] = out[:, 1] + Gc[:, 1]
    return out

def s_iso_level(S):
    out = S.copy()
    for h in (0, 1):
        a = half != h; b = half == h
        for j in range(3):
            iso = IsotonicRegression(out_of_bounds="clip").fit(S[a, j], dv.score[a, j])
            out[b, j] = iso.predict(S[b, j])
    return out

def s_shrink_ood(S, w=0.5, target="fam"):
    """shrink toward the family (or global) mean when the top-1 similarity is low."""
    out = S.copy()
    for h in (0, 1):
        a = half != h; b = half == h
        for j in range(3):
            gm = S[a, j].mean()
            mu = np.array([S[a & (famid == f), j].mean() if (a & (famid == f)).sum() > 5 else gm
                           for f in range(len(fams))])
            base = mu[famid[b]] if target == "fam" else gm
            lam = w * (1.0 - SIM[b])
            out[b, j] = (1 - lam) * S[b, j] + lam * base
    return out

def s_expand(S, g=1.2):
    """variance expansion around the per-model mean (opposite of shrinkage)."""
    out = S.copy()
    for j in range(3):
        out[:, j] = S[:, j].mean() + g * (S[:, j] - S[:, j].mean())
    return np.clip(out, 0.0, 1.0)

def sig2_family(C, shrink=20.0):
    r = np.log(dv.cost) - np.log(C); V = np.zeros_like(C)
    for h in (0, 1):
        a = half != h; b = half == h
        for j in range(3):
            g = r[a, j].var()
            for f in range(len(fams)):
                aa = a & (famid == f); bb = b & (famid == f)
                if bb.sum() == 0: continue
                if aa.sum() < 8: V[bb, j] = g; continue
                w = aa.sum() / (aa.sum() + shrink)
                V[bb, j] = w * r[aa, j].var() + (1 - w) * g
    return V

COSTS = {"base": lambda C, S: C, "famvar1.0": lambda C, S: C * np.exp(1.0 * sig2_family(C))}

def stats(mk_s, mk_c):
    tot = 0.0; loso = 0.0; rows = []
    for t in TIERS:
        S = mk_s(P[f"score_{t}"]); C = mk_c(P[f"cost_{t}"].copy(), S)
        pth = LPath(S, C, dv.cost, dv.score); mult = TIER_MULT[t]
        cur = {}
        for s in SEEDS:
            b = pth.batch(Ws[s]); ev = np.zeros(len(GRID))
            for gi, sf in enumerate(GRID):
                sc, ra, pa = eval_cached(b, mult, float(sf)); ev[gi] = (sc * pa).mean()
            cur[s] = ev
        ev = np.mean([cur[s] for s in SEEDS], axis=0); bi = int(ev.argmax())
        lo = np.mean([cur[h][int(np.mean([cur[s] for s in SEEDS if s != h], axis=0).argmax())] for h in SEEDS])
        tot += TIER_WEIGHT[t]*ev[bi]; loso += TIER_WEIGHT[t]*lo
        rows.append(f"{t[:4]} {ev[bi]:.4f}@{GRID[bi]:.3f}")
    return tot, loso, rows

VAR = {
    "S0 base":          s_base,
    "S1 cummax":        s_cummax,
    "S2 clip gain>=0":  s_clipgain,
    "S3 gain linear":   lambda S: s_gain_lin(S, "lin"),
    "S4 gain isotonic": lambda S: s_gain_lin(S, "iso"),
    "S5 level isotonic": s_iso_level,
    "S6 OOD shrink fam w=.5": lambda S: s_shrink_ood(S, 0.5, "fam"),
    "S7 OOD shrink fam w=1":  lambda S: s_shrink_ood(S, 1.0, "fam"),
    "S8 expand 1.2":    lambda S: s_expand(S, 1.2),
    "S9 expand 1.5":    lambda S: s_expand(S, 1.5),
    "S10 expand 0.8":   lambda S: s_expand(S, 0.8),
}
print("\n=== score transforms x cost model ===")
for cn, cf in COSTS.items():
    print(f"-- cost = {cn}")
    for sn, sf_ in VAR.items():
        tot, lo, rows = stats(sf_, cf)
        print(f"   {sn:24s} EVbest={tot:.4f} loso={lo:.4f}  " + " | ".join(rows))
