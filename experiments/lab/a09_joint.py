# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a09: robustness of the gain-isotonic + family-variance smearing combo."""
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

dv = load_split("dev"); n = len(dv)
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
fam = np.array([classify_family(t) for t in dv.texts]); fams = sorted(set(fam))
famid = np.array([fams.index(f) for f in fam])
GRID = np.round(np.arange(0.55, 1.501, 0.005), 4)
SEEDS = (7, 17, 23); Ws = {s: make_W(n, 400, s) for s in SEEDS}
HS = {h: np.random.default_rng(h).integers(0, 2, size=n) for h in (0, 1, 2, 3, 4)}

def gain_iso(S, half, alpha=1.0, kfold=2):
    out = S.copy()
    G = np.column_stack([S[:, 1] - S[:, 0], S[:, 2] - S[:, 1]])
    T = np.column_stack([dv.score[:, 1] - dv.score[:, 0], dv.score[:, 2] - dv.score[:, 1]])
    Gc = G.copy()
    for h in (0, 1):
        a = half != h; b = half == h
        for k in (0, 1):
            iso = IsotonicRegression(out_of_bounds="clip").fit(G[a, k], T[a, k])
            Gc[b, k] = iso.predict(G[b, k])
    Gm = (1 - alpha) * G + alpha * Gc
    out[:, 1] = S[:, 0] + Gm[:, 0]; out[:, 2] = out[:, 1] + Gm[:, 1]
    return out

def sig2_family(C, half, shrink=20.0):
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

def stats(mk_s, mk_c, tag=""):
    tot = 0.0; loso = 0.0; rows = []
    for t in TIERS:
        S = mk_s(P[f"score_{t}"]); C = mk_c(P[f"cost_{t}"].copy())
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
    print(f"  {tag:34s} EVbest={tot:.4f} loso={loso:.4f}  " + " | ".join(rows))
    return tot, loso

print("=== cross-fit-split robustness (5 splits) ===")
res = {}
for h in (0, 1, 2, 3, 4):
    hh = HS[h]
    res[("base", h)] = stats(lambda S: S, lambda C: C, f"split{h} base")
    res[("iso", h)] = stats(lambda S, q=hh: gain_iso(S, q), lambda C: C, f"split{h} gain-iso")
    res[("fam", h)] = stats(lambda S: S, lambda C, q=hh: C*np.exp(sig2_family(C, q)), f"split{h} famvar")
    res[("both", h)] = stats(lambda S, q=hh: gain_iso(S, q), lambda C, q=hh: C*np.exp(sig2_family(C, q)), f"split{h} both")
print()
for k in ("base", "iso", "fam", "both"):
    b = np.array([res[(k, h)][0] for h in range(5)]); l = np.array([res[(k, h)][1] for h in range(5)])
    print(f"  {k:6s} EVbest {b.mean():.4f} +-{b.std():.4f}  loso {l.mean():.4f} +-{l.std():.4f}  "
          f"deltas vs base per split: {np.round(b - np.array([res[('base',h)][0] for h in range(5)]),4)}")

print("\n=== gain-isotonic alpha sweep (split 0) ===")
for al in (0.0, 0.25, 0.5, 0.75, 1.0):
    stats(lambda S, a=al: gain_iso(S, HS[0], a), lambda C: C*np.exp(sig2_family(C, HS[0])), f"alpha={al}")

print("\n=== what the isotonic map does (split 0, premium tier) ===")
S = P["score_premium"]
G = np.column_stack([S[:, 1]-S[:, 0], S[:, 2]-S[:, 1]])
Gc = gain_iso(S, HS[0]); Gc = np.column_stack([Gc[:, 1]-Gc[:, 0], Gc[:, 2]-Gc[:, 1]])
for k, lab in enumerate(("mid-light", "k1-mid")):
    qs = np.quantile(G[:, k], np.linspace(0, 1, 11))
    print(f"  {lab}: " + " ".join(f"[{qs[i]:+.2f}->{Gc[(G[:,k]>=qs[i])&(G[:,k]<=qs[i+1]),k].mean():+.3f}]" for i in range(10)))
    print(f"    spread: raw sd={G[:,k].std():.4f} calibrated sd={Gc[:,k].std():.4f} "
          f"raw mean={G[:,k].mean():+.4f} cal mean={Gc[:,k].mean():+.4f}")
