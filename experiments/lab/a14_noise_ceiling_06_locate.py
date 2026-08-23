# SPDX-License-Identifier: Apache-2.0
"""a14 step 6 -- where is the deployed system on the honest axes?

Exact identities used (no model needed for these):
    Cov(pred, p)      = Cov(pred, s)            (label noise _|_ prediction)
    corr(pred, p)     = corr(pred, s) * sd(s)/sd(p)
    Cov(predgain, gp) = Cov(predgain, gs)
    corr(predgain,gp) = corr(predgain, gs) * sd(gs)/sd(gp)
sd(p) and sd(gain_p) come from the exact variance decomposition of step 1.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_split, TIERS, MODEL_IDS                       # noqa: E402
from ossp_router.similarity import classify_family                     # noqa: E402

dv = load_split("dev")
N = len(dv)
P43 = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
CACHE = HERE / "_a14_cache"
D = np.load(CACHE / "pdraws_dev.npz")
PHAT = D["phat"].astype(np.float64)
PD = D["p"].astype(np.float64)


def cell_moments(x, nn):
    mu = float(x.mean())
    inv = float((1.0 / nn).mean())
    return mu, (float(np.mean(x ** 2)) - inv * mu) / (1.0 - inv) - mu * mu


VP = np.array([cell_moments(dv.score[:, j], dv.ngen[:, j])[1] for j in range(3)])
CS = np.cov(dv.score.T)
CP = CS.copy()
np.fill_diagonal(CP, VP)


def main():
    print("=" * 96)
    print("LEVEL AXIS -- corr(prediction, latent p) of the deployed E43 predictions (dev)")
    print("=" * 96)
    print(f"{'tier':10s} " + " ".join(f"{m:>26s}" for m in MODEL_IDS))
    print(f"{'':10s} " + " ".join(f"{'corr(s)':>8s}{'sd(s)/sd(p)':>12s}{'corr(p)':>8s}"
                                  for _ in MODEL_IDS))
    for t in TIERS:
        cells = []
        for j in range(3):
            cs = np.corrcoef(P43[f"score_{t}"][:, j], dv.score[:, j])[0, 1]
            k = dv.score[:, j].std() / np.sqrt(VP[j])
            cells.append(f"{cs:8.3f}{k:12.4f}{cs*k:8.3f}")
        print(f"{t:10s} " + " ".join(cells))

    print("\n" + "=" * 96)
    print("GAIN AXIS -- what the Lagrangian allocator actually consumes")
    print("=" * 96)
    print(f"{'tier':10s} {'pair':10s} {'corr(g_pred,g_obs)':>19s} {'sd(gs)/sd(gp)':>14s} "
          f"{'corr(g_pred,g_p)':>17s} {'sd(g_pred)':>11s} {'sd(g_p)':>9s} {'slope':>7s}")
    for t in TIERS:
        for (a, b, lab) in ((0, 1, "mid-light"), (1, 2, "k1-mid"), (0, 2, "k1-light")):
            gp = P43[f"score_{t}"][:, b] - P43[f"score_{t}"][:, a]
            gs = dv.score[:, b] - dv.score[:, a]
            vgp = VP[a] + VP[b] - 2 * CP[a, b]
            cg = np.corrcoef(gp, gs)[0, 1]
            k = gs.std() / np.sqrt(vgp)
            slope = np.cov(gp, gs)[0, 1] / gp.var()      # regression of latent gain on pred
            print(f"{t:10s} {lab:10s} {cg:19.3f} {k:14.4f} {cg*k:17.3f} "
                  f"{gp.std():11.4f} {np.sqrt(vgp):9.4f} {slope:7.3f}")

    print("\n" + "=" * 96)
    print("ERROR STRUCTURE of the deployed predictions (needed to pick the right")
    print("cross-model error correlation for the synthetic exchange-rate predictor)")
    print("=" * 96)
    for t in TIERS:
        e = P43[f"score_{t}"] - PHAT
        C = np.corrcoef(e.T)
        print(f"  {t:9s} err sd={np.round(e.std(0),4)}  err corr(0,1)={C[0,1]:.3f} "
              f"(0,2)={C[0,2]:.3f} (1,2)={C[1,2]:.3f}  mean err={np.round(e.mean(0),4)}")
    print("  (errors measured against E[p|k]; the part of the error that is label")
    print("   noise is removed, so these are close to true prediction errors)")

    print("\ncalibration: regression latent p ~ a + b*pred  (b=1 means calibrated)")
    for t in TIERS:
        bs = []
        for j in range(3):
            pr = P43[f"score_{t}"][:, j]
            bs.append(np.cov(pr, dv.score[:, j])[0, 1] / pr.var())
        print(f"  {t:9s} slope={np.round(bs,3)}  pred sd={np.round(P43[f'score_{t}'].std(0),4)}"
              f"  sd(p)={np.round(np.sqrt(VP),4)}")

    print("\n" + "=" * 96)
    print("HOW MUCH OF corr(pred,p) IS JUST THE FAMILY MEAN?")
    print("=" * 96)
    fam = np.array([classify_family(t) for t in dv.texts])
    fm = np.zeros_like(dv.score)
    for f_ in set(fam):
        m = fam == f_
        fm[m] = dv.score[m].mean(0)
    for j in range(3):
        cs = np.corrcoef(fm[:, j], dv.score[:, j])[0, 1]
        k = dv.score[:, j].std() / np.sqrt(VP[j])
        print(f"  {MODEL_IDS[j]:12s} family-mean corr(p)={cs*k:.3f} "
              f"vs deployed {np.corrcoef(P43['fast_score' if False else 'score_fast'][:, j], dv.score[:, j])[0,1]*k:.3f}")
    for (a, b, lab) in ((0, 1, "mid-light"), (1, 2, "k1-mid")):
        gfm = fm[:, b] - fm[:, a]
        gs = dv.score[:, b] - dv.score[:, a]
        vgp = VP[a] + VP[b] - 2 * CP[a, b]
        k = gs.std() / np.sqrt(vgp)
        print(f"  gain {lab:10s} family-mean corr(g_p)={np.corrcoef(gfm, gs)[0,1]*k:.3f}")

    print("\n" + "=" * 96)
    print("PER-FAMILY corr(pred, latent p) -- where the signal is and is not")
    print("=" * 96)
    print(f"{'family':16s} {'N':>4s} " + " ".join(f"{m:>13s}" for m in MODEL_IDS)
          + f" {'gain m-l':>9s} {'gain k-m':>9s}")
    for f_ in sorted(set(fam)):
        m = fam == f_
        if m.sum() < 20:
            continue
        row = []
        for j in range(3):
            x = dv.score[m, j]
            mu, vp = cell_moments(x, dv.ngen[m, j])
            if vp <= 1e-6:
                row.append(f"{'--':>13s}"); continue
            c = np.corrcoef(P43["score_fast"][m, j], x)[0, 1] * x.std() / np.sqrt(vp)
            row.append(f"{c:13.3f}")
        gs_ = []
        for (a, b) in ((0, 1), (1, 2)):
            Cm = np.cov(dv.score[m].T)
            vpa = cell_moments(dv.score[m, a], dv.ngen[m, a])[1]
            vpb = cell_moments(dv.score[m, b], dv.ngen[m, b])[1]
            vg = max(vpa + vpb - 2 * Cm[a, b], 1e-9)
            g_obs = dv.score[m, b] - dv.score[m, a]
            g_pr = P43["score_fast"][m, b] - P43["score_fast"][m, a]
            if g_obs.std() < 1e-9 or g_pr.std() < 1e-9:
                gs_.append(f"{'--':>9s}"); continue
            gs_.append(f"{np.corrcoef(g_pr, g_obs)[0,1]*g_obs.std()/np.sqrt(vg):9.3f}")
        print(f"{f_:16s} {m.sum():4d} " + " ".join(row) + " " + " ".join(gs_))


if __name__ == "__main__":
    main()
