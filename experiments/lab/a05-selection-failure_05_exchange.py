# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a05 step 5: per-decision exchange rate + a cross-fitted 2-D gain calibration test.

(1) The allocator is invariant to a per-item constant added to all three scores,
    so only the two gains matter.  Measure the exchange rate separately for the
    mid-light gain and the k1-mid gain, blending toward the EB expected p
    (noise-stripped) rather than the realised label.
(2) Cross-fitted (2-fold on dev) calibration of the predicted gain on
    (pred s_light, pred gain) -- an untried 2-D variant of E30.
(3) Oracle vs deployed: budget efficiency of mid picks and k1 picks.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
ROOT = Path(__file__).resolve().parents[2]

dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
Cc = np.load(ROOT / "experiments/lab/a05-selection-failure_cache.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
N = len(dv); IDX = np.arange(N)
FAM = Cc["fam"]; PHAT = Cc["phat"]
L = dv.cost[:, 0].sum()


def run(mk_s, mk_c, tune=True):
    tot_r = tot_e = 0.0; sfs = []
    for t in TIERS:
        if tune:
            best = None
            for sf in np.arange(0.5, 1.301, 0.005):
                r = tier_result(mk_s(t), mk_c(t), dv, t, float(sf))
                if r["passed"] and (best is None or r["score"] > best["score"]):
                    best = r; bsf = float(sf)
            r = best; sfs.append(bsf)
        else:
            r = tier_result(mk_s(t), mk_c(t), dv, t, SAFE[t]); sfs.append(SAFE[t])
        e = PHAT[IDX, r["sel"]].mean() if r["passed"] else 0.0
        tot_r += TIER_WEIGHT[t] * r["tier_score"]; tot_e += TIER_WEIGHT[t] * e
    return tot_r, tot_e, sfs


pc = lambda t: P[f"cost_{t}"]
print("=== (1) exchange rate per DECISION (blend toward EB expected p, pred costs, safety re-tuned) ===")
print(f"  {'target':16s} {'lam':>5s} {'final(realised)':>15s} {'final(EB)':>10s} "
      f"{'AUC(pair)':>9s} {'spear':>7s} {'safety f/b/p':>18s}")
for which in ("mid-light", "k1-mid", "both"):
    for lam in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        def mk(t, lam=lam, which=which):
            S = P[f"score_{t}"].copy()
            g1 = S[:, 1] - S[:, 0]; g2 = S[:, 2] - S[:, 1]
            t1 = PHAT[:, 1] - PHAT[:, 0]; t2 = PHAT[:, 2] - PHAT[:, 1]
            if which in ("mid-light", "both"):
                g1 = (1 - lam) * g1 + lam * t1
            if which in ("k1-mid", "both"):
                g2 = (1 - lam) * g2 + lam * t2
            S[:, 1] = S[:, 0] + g1; S[:, 2] = S[:, 1] + g2
            return S
        r, e, sfs = run(mk, pc)
        S = mk("premium")
        if which == "k1-mid":
            pg = S[:, 2] - S[:, 1]; tg = PHAT[:, 2] - PHAT[:, 1]; lab = dv.score[:, 2] > dv.score[:, 1]
        else:
            pg = S[:, 1] - S[:, 0]; tg = PHAT[:, 1] - PHAT[:, 0]; lab = dv.score[:, 1] > dv.score[:, 0]
        rk = np.argsort(np.argsort(pg)) + 1.0
        n1 = lab.sum(); n0 = (~lab).sum()
        a = (rk[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        print(f"  {which:16s} {lam:5.2f} {r:15.4f} {e:10.4f} {a:9.3f} "
              f"{spearmanr(pg, tg).statistic:7.3f} {'/'.join(f'{x:.3f}' for x in sfs):>18s}")

print("\n=== (2) cross-fitted 2-D gain calibration g_hat -> E[g_true | s_hat_light, g_hat] ===")
print("    (2 folds on dev, fitted on the other half; 4x4 quantile grid, shrunk to the cell mean)")
rng = np.random.default_rng(0)
fold = rng.permutation(N) % 2
def calibrate(tier, target, nb=4, kmin=20):
    S = P[f"score_{tier}"]
    out = S.copy()
    for (a, b) in ((0, 1), (1, 2)):
        gh = S[:, b] - S[:, a]
        gt = target[:, b] - target[:, a]
        newg = gh.copy()
        for f in (0, 1):
            trm = fold != f; tem = fold == f
            qx = np.quantile(S[trm, 0], np.linspace(0, 1, nb + 1)); qx[0] -= 1; qx[-1] += 1
            qy = np.quantile(gh[trm], np.linspace(0, 1, nb + 1)); qy[0] -= 1; qy[-1] += 1
            bx_tr = np.digitize(S[trm, 0], qx[1:-1]); by_tr = np.digitize(gh[trm], qy[1:-1])
            bx_te = np.digitize(S[tem, 0], qx[1:-1]); by_te = np.digitize(gh[tem], qy[1:-1])
            gl = gt[trm].mean()
            for i in range(nb):
                for j in range(nb):
                    m_tr = (bx_tr == i) & (by_tr == j)
                    m_te = (bx_te == i) & (by_te == j)
                    if m_te.sum() == 0:
                        continue
                    k = m_tr.sum()
                    v = (gt[trm][m_tr].sum() + kmin * gl) / (k + kmin) if k else gl
                    idxs = np.where(tem)[0][m_te]
                    newg[idxs] = v
        if b == 1:
            out[:, 1] = out[:, 0] + newg
        else:
            out[:, 2] = out[:, 1] + newg
    return out

for tname, target in (("realised s", dv.score), ("EB p", PHAT)):
    for alpha in (0.0, 0.25, 0.5, 1.0):
        cache = {t: calibrate(t, target) for t in TIERS}
        def mk(t, alpha=alpha, cache=cache):
            return (1 - alpha) * P[f"score_{t}"] + alpha * cache[t]
        r, e, sfs = run(mk, pc)
        print(f"  target={tname:11s} alpha={alpha:4.2f}  realised={r:.4f}  EB={e:.4f}  "
              f"safety={'/'.join(f'{x:.3f}' for x in sfs)}")

print("\n=== (3) budget efficiency of the picks: deployed vs oracle ===")
print(f"  {'tier':9s} {'who':9s} {'model':6s} {'n':>4s} {'score gain/N':>13s} {'cost/L':>8s} {'gain per L':>11s}")
for t in TIERS:
    for who, sel in (("deployed", Cc[f"sel_d_{t}"]), ("oracle", Cc[f"sel_o_{t}"])):
        for j, nm in ((1, "mid"), (2, "k1")):
            m = sel == j
            if m.sum() == 0:
                continue
            g = (dv.score[m, j] - dv.score[m, 0]).sum() / N
            c = (dv.cost[m, j] - dv.cost[m, 0]).sum() / L
            print(f"  {t:9s} {who:9s} {nm:6s} {m.sum():4d} {g:+13.4f} {c:8.3f} "
                  f"{g/max(c,1e-9):11.4f}")

print("\n=== (4) how much of the deployed budget goes to each decision ===")
for t in TIERS:
    sd = Cc[f"sel_d_{t}"]
    cm = (dv.cost[sd == 1, 1] - dv.cost[sd == 1, 0]).sum() / L
    ck = (dv.cost[sd == 2, 2] - dv.cost[sd == 2, 0]).sum() / L
    print(f"  {t:9s} extra budget: mid {cm:.3f}L ({100*cm/(cm+ck):.0f}%)  "
          f"k1 {ck:.3f}L ({100*ck/(cm+ck):.0f}%)   total {cm+ck:.3f}L of "
          f"{TIER_MULT[t]-1:.2f}L available")
