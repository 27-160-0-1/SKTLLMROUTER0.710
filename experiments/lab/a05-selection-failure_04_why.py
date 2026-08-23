# SPDX-License-Identifier: Apache-2.0
"""a05 step 4: WHY the wrong model is picked.
(A) headroom ceiling: is predicted gain calibrated conditional on the predicted
    light score?  (B) pairwise decision quality.  (C) marginal value of a perfect
    gain signal, pair by pair, evaluated on realised score AND on EB expected p.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result, MODEL_IDS
ROOT = Path(__file__).resolve().parents[2]

tr = load_split("train"); dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
Cc = np.load(ROOT / "experiments/lab/a05-selection-failure_cache.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
N = len(dv); IDX = np.arange(N)
FAM = Cc["fam"]; PHAT = Cc["phat"]
L = dv.cost[:, 0].sum()

# ---------------------------------------------------------------- (A) headroom ceiling
print("=== (A) is predicted gain calibrated against the headroom ceiling? ===")
print("    realised gain (mid-light) is bounded above by 1 - s_light.")
for tier in ("premium",):
    S = P[f"score_{tier}"]
    sl = S[:, 0]
    q = np.quantile(sl, np.linspace(0, 1, 11))
    q[0] -= 1e-9
    b = np.digitize(sl, q[1:-1])
    print(f"  tier={tier}  decile of predicted s_light")
    print(f"  {'dec':>3s} {'n':>4s} {'mean pred sL':>12s} {'mean true sL':>12s} {'mean EB pL':>10s} | "
          f"{'pred gM':>8s} {'true gM':>8s} {'EB gM':>7s} {'headroom':>9s} | "
          f"{'pred gK':>8s} {'true gK':>8s} {'EB gK':>7s}")
    for d in range(10):
        m = b == d
        pgm = (S[m, 1] - S[m, 0]).mean(); tgm = (dv.score[m, 1] - dv.score[m, 0]).mean()
        egm = (PHAT[m, 1] - PHAT[m, 0]).mean()
        pgk = (S[m, 2] - S[m, 1]).mean(); tgk = (dv.score[m, 2] - dv.score[m, 1]).mean()
        egk = (PHAT[m, 2] - PHAT[m, 1]).mean()
        print(f"  {d:3d} {m.sum():4d} {S[m,0].mean():12.3f} {dv.score[m,0].mean():12.3f} "
              f"{PHAT[m,0].mean():10.3f} | {pgm:+8.3f} {tgm:+8.3f} {egm:+7.3f} "
              f"{1-PHAT[m,0].mean():9.3f} | {pgk:+8.3f} {tgk:+8.3f} {egk:+7.3f}")

# same diagnostic on TRAIN would need train preds (not available) -> state as not measured
print("\n  NOTE: this table uses dev predictions (held-out w.r.t. training) -- it is a")
print("  diagnostic, not a fitted correction.  Train-side predictions were not dumped.")

# ---------------------------------------------------------------- (B) pairwise decision quality
print("\n=== (B) pairwise decision quality (dev, premium-tier predictions) ===")
S = P["score_premium"]
def auc(sc, lab):
    lab = lab.astype(bool)
    if lab.all() or (~lab).any() is False:
        return float("nan")
    r = np.argsort(np.argsort(sc)) + 1.0
    n1 = lab.sum(); n0 = (~lab).sum()
    return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

pairs = [("mid>light", 0, 1), ("k1>mid", 1, 2), ("k1>light", 0, 2)]
print(f"  {'pair':10s} {'n strict+':>9s} {'AUC realised':>13s} {'AUC EB p':>9s} "
      f"{'spear gain':>11s} {'spear EBgain':>13s} {'spear eff':>10s}")
from scipy.stats import spearmanr
for nm, a, bq in pairs:
    pg = S[:, bq] - S[:, a]
    tg = dv.score[:, bq] - dv.score[:, a]
    eg = PHAT[:, bq] - PHAT[:, a]
    pos = tg > 0
    pe = pg / np.maximum(P["cost_premium"][:, bq] - P["cost_premium"][:, a], 1e-9)
    te = tg / np.maximum(dv.cost[:, bq] - dv.cost[:, a], 1e-9)
    print(f"  {nm:10s} {pos.sum():9d} {auc(pg, pos):13.3f} {auc(pg, eg>1e-6):9.3f} "
          f"{spearmanr(pg, tg).statistic:11.3f} {spearmanr(pg, eg).statistic:13.3f} "
          f"{spearmanr(pe, te).statistic:10.3f}")

# precision of the upgrades we actually bought
print("\n  precision of the upgrades actually bought (deployed allocation):")
for tier in TIERS:
    sd = Cc[f"sel_d_{tier}"]
    for j, nm in ((1, "mid"), (2, "k1 ")):
        m = sd == j
        if m.sum() == 0:
            continue
        tg = dv.score[m, j] - dv.score[m, 0]
        eg = PHAT[m, j] - PHAT[m, 0]
        print(f"    {tier:9s} chose {nm} n={m.sum():4d}: realised gain>0 {np.mean(tg>0):5.1%}  "
              f"=0 {np.mean(tg==0):5.1%}  <0 {np.mean(tg<0):5.1%}   "
              f"mean realised {tg.mean():+.3f}  mean EB {eg.mean():+.3f}  "
              f"cost/L {(dv.cost[m,j]-dv.cost[m,0]).sum()/L:.3f}")

# ---------------------------------------------------------------- (C) marginal value of a perfect pair signal
print("\n=== (C) marginal value of a perfect signal, decomposed by decision ===")
def run(name, mk_s, mk_c, tune=False):
    tot_r = tot_e = 0.0; parts = []
    for t in TIERS:
        if tune:
            best = None
            for sf in np.arange(0.5, 1.301, 0.005):
                r = tier_result(mk_s(t), mk_c(t), dv, t, float(sf))
                if r["passed"] and (best is None or r["score"] > best["score"]):
                    best = r
            r = best
        else:
            r = tier_result(mk_s(t), mk_c(t), dv, t, SAFE[t])
        e = PHAT[IDX, r["sel"]].mean() if r["passed"] else 0.0
        tot_r += TIER_WEIGHT[t] * r["tier_score"]; tot_e += TIER_WEIGHT[t] * e
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else '!'}")
    print(f"  {name:44s} realised={tot_r:.4f} EB={tot_e:.4f}  " + " ".join(parts))
    return tot_r, tot_e

pc = lambda t: P[f"cost_{t}"]
tc = lambda t: dv.cost
for cname, cf in (("predCost", pc), ("TRUEcost", tc)):
    print(f"  --- decision costs = {cname}, safety re-tuned on dev (optimistic) ---")
    run(f"[{cname}] baseline", lambda t: P[f"score_{t}"], cf, tune=True)
    def mk_gm(t):     # perfect mid-light gain, everything else as predicted
        Sx = P[f"score_{t}"].copy()
        Sx[:, 1] = Sx[:, 0] + (dv.score[:, 1] - dv.score[:, 0])
        Sx[:, 2] = Sx[:, 1] + (P[f"score_{t}"][:, 2] - P[f"score_{t}"][:, 1])
        return Sx
    run(f"[{cname}] + perfect gain(mid-light)", mk_gm, cf, tune=True)
    def mk_gk(t):     # perfect k1-mid gain
        Sx = P[f"score_{t}"].copy()
        Sx[:, 2] = Sx[:, 1] + (dv.score[:, 2] - dv.score[:, 1])
        return Sx
    run(f"[{cname}] + perfect gain(k1-mid)", mk_gk, cf, tune=True)
    def mk_l(t):      # perfect light level only (gains kept as predicted)
        Sx = P[f"score_{t}"].copy()
        d1 = Sx[:, 1] - Sx[:, 0]; d2 = Sx[:, 2] - Sx[:, 1]
        Sx[:, 0] = dv.score[:, 0]; Sx[:, 1] = Sx[:, 0] + d1; Sx[:, 2] = Sx[:, 1] + d2
        return Sx
    run(f"[{cname}] + perfect light level (gains predicted)", mk_l, cf, tune=True)
    run(f"[{cname}] + all true scores", lambda t: dv.score, cf, tune=True)

# EB-surface versions: allocate on EB p (noise-free target), evaluate on EB p
print("\n  --- the same, but the 'perfect' signal is the EB expected p (noise stripped) ---")
for cname, cf in (("predCost", pc), ("TRUEcost", tc)):
    def mk_gm(t):
        Sx = P[f"score_{t}"].copy()
        Sx[:, 1] = Sx[:, 0] + (PHAT[:, 1] - PHAT[:, 0])
        Sx[:, 2] = Sx[:, 1] + (P[f"score_{t}"][:, 2] - P[f"score_{t}"][:, 1])
        return Sx
    def mk_gk(t):
        Sx = P[f"score_{t}"].copy()
        Sx[:, 2] = Sx[:, 1] + (PHAT[:, 2] - PHAT[:, 1])
        return Sx
    run(f"[{cname}] EB-perfect gain(mid-light)", mk_gm, cf, tune=True)
    run(f"[{cname}] EB-perfect gain(k1-mid)", mk_gk, cf, tune=True)
    run(f"[{cname}] EB-perfect all", lambda t: PHAT, cf, tune=True)
