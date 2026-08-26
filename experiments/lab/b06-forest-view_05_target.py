# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b06 forest-view: EV is not the objective if the goal is literally 0.72.

The competition pays ONE 880-item sample.  bench2 chooses the safety triple that
maximises E[final].  If the goal is P(final >= target), the optimum is a
different, higher-spend triple, because the payoff is convex at the threshold.
Measured two ways:
  * honestly, on the Train-OOF bootstrap (no dev read at any step);
  * descriptively, on a dev bootstrap, to size the sampling distribution.
Also: the in-sample vs out-of-fold R2 gap of every GBM head (how much the tree
stack memorises 1,760 rows).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP, DEPLOYED_SAFETY, _gbm_params
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import bench2 as B
import protocol as P

OUT = Path("reports/lab/b06_target.json")
GRID = {"fast": np.arange(0.86, 1.021, 0.02),
        "balanced": np.arange(0.70, 1.001, 0.02),
        "premium": np.arange(0.62, 1.001, 0.02)}


def tier_panel(lab, a, cfg, tier, samples):
    """(n_safety, nboot) realised tier contribution (0 when the tier busts)."""
    idx = a["idx"]
    ps, pc = lab.compose(a, cfg, tier)
    g = GRID[tier]
    smp = np.asarray(samples)
    PS = ps[smp]; PC = pc[smp]
    TS = lab.true_s[idx][smp]; TC = lab.true_c[idx][smp]
    out = np.zeros((len(g), len(smp)))
    bus = np.zeros((len(g), len(smp)))
    for gi, s in enumerate(g):
        pick = P.exact_allocate(PS, PC, MULTS[tier], float(s))
        r = np.arange(len(smp))[:, None]
        real = np.take_along_axis(TC, pick[:, :, None], axis=2)[:, :, 0].sum(axis=1)
        sc = np.take_along_axis(TS, pick[:, :, None], axis=2)[:, :, 0].mean(axis=1)
        base = TC[:, :, 0].sum(axis=1)
        bust = (real / base) > MULTS[tier]
        out[gi] = np.where(bust, 0.0, sc)
        bus[gi] = bust
    return g, out, bus


def analyse(lab, a, cfg, samples, targets, tag):
    panels = {}
    for t in TIERS:
        panels[t] = tier_panel(lab, a, cfg, t, samples)
    gf, Ff, Bf = panels["fast"]; gb, Fb, Bb = panels["balanced"]; gp, Fp, Bp = panels["premium"]
    nb = Ff.shape[1]
    best = {}
    # E[final] is separable -> argmax per tier
    evs = {t: panels[t][1].mean(axis=1) for t in TIERS}
    sEV = {t: float(panels[t][0][int(np.argmax(evs[t]))]) for t in TIERS}
    evEV = sum(W[t] * evs[t].max() for t in TIERS)
    best["EV"] = dict(safety=sEV, ev=float(evEV))
    # P(final >= T) needs the joint grid
    F = (0.4 * Ff[:, None, None, :] + 0.3 * Fb[None, :, None, :] + 0.3 * Fp[None, None, :, :])
    mean = F.mean(axis=3)
    for T in targets:
        pr = (F >= T).mean(axis=3)
        i, j, k = np.unravel_index(int(np.argmax(pr)), pr.shape)
        iE = int(np.argmax(evs["fast"])); jE = int(np.argmax(evs["balanced"])); kE = int(np.argmax(evs["premium"]))
        best[f"P>={T}"] = dict(
            safety={"fast": float(gf[i]), "balanced": float(gb[j]), "premium": float(gp[k])},
            p=float(pr[i, j, k]), mean_at_p_opt=float(mean[i, j, k]),
            p_at_EV_opt=float(pr[iE, jE, kE]), mean_at_EV_opt=float(mean[iE, jE, kE]))
    print(f"\n[{tag}] nboot={nb}")
    print(f"  argmax E[final]: safety {'/'.join(f'{sEV[t]:.2f}' for t in TIERS)}  E={evEV:.6f}")
    for T in targets:
        d = best[f"P>={T}"]
        sfs = "/".join(f"{d['safety'][t]:.2f}" for t in TIERS)
        print(f"  argmax P(final>={T}): safety {sfs}"
              f"  P={d['p']:.4f} (E={d['mean_at_p_opt']:.6f})  "
              f"vs at the EV optimum P={d['p_at_EV_opt']:.4f} (E={d['mean_at_EV_opt']:.6f})")
    return best


def main():
    lab = Lab()
    cv, arr = B.stage(lab, DEPLOYED_EXP, tag="base")
    rep = {}

    # ---------------- honest: Train-OOF bootstrap, 880-item resamples
    m = len(cv["idx"])
    smp = np.asarray(lab.samples_for(m, 7, 600, 880))
    rep["oof"] = analyse(lab, cv, DEPLOYED_CFG, smp, [0.66, 0.68, 0.70], "honest Train-OOF")

    # ---------------- descriptive: dev bootstrap (oracle; sizing only)
    md = len(arr["idx"])
    smpd = np.asarray(lab.samples_for(md, 7, 600, 880))
    rep["dev"] = analyse(lab, arr, DEPLOYED_CFG, smpd, [0.70, 0.71, 0.72, 0.73], "DEV bootstrap (descriptive)")

    # sampling distribution at three named operating points
    print("\n  dev sampling distribution at named safety triples:")
    named = {"bench2 EV-opt": {"fast": 0.96, "balanced": 0.84, "premium": 0.735},
             "E43 deployed": DEPLOYED_SAFETY,
             "E43 0.7017": {"fast": 0.98, "balanced": 0.89, "premium": 0.88},
             "spend-max": {"fast": 1.00, "balanced": 1.00, "premium": 1.00}}
    idx = arr["idx"]; r = np.arange(len(idx))
    rows = {}
    for nm, sf in named.items():
        picks = {}
        for t in TIERS:
            ps, pc = lab.compose(arr, DEPLOYED_CFG, t)
            picks[t] = lab.allocate(ps, pc, MULTS[t], sf[t])
        rng = np.random.default_rng(11); nb = 4000
        fin = np.zeros(nb); bu = {t: 0 for t in TIERS}
        pt = 0.0
        for t in TIERS:
            ratio = lab.true_c[idx][r, picks[t]].sum() / lab.true_c[idx][:, 0].sum()
            pt += W[t] * (lab.true_s[idx][r, picks[t]].mean() if ratio <= MULTS[t] else 0.0)
        for b in range(nb):
            s = rng.integers(0, len(idx), size=len(idx))
            v = 0.0
            for t in TIERS:
                base = lab.true_c[idx][:, 0][s].sum()
                ratio = lab.true_c[idx][r, picks[t]][s].sum() / base
                if ratio > MULTS[t]:
                    bu[t] += 1
                else:
                    v += W[t] * lab.true_s[idx][r, picks[t]][s].mean()
            fin[b] = v
        rows[nm] = dict(point=float(pt), mean=float(fin.mean()), sd=float(fin.std()),
                        p5=float(np.quantile(fin, .05)), p50=float(np.quantile(fin, .5)),
                        p95=float(np.quantile(fin, .95)),
                        p_ge_072=float((fin >= 0.72).mean()), p_ge_071=float((fin >= 0.71).mean()),
                        bust={t: bu[t] / nb for t in TIERS}, safety=dict(sf))
        print(f"    {nm:15s} sf={'/'.join(f'{sf[t]:.3f}' for t in TIERS)} point={pt:.4f} "
              f"mean={fin.mean():.4f} sd={fin.std():.4f} p5={np.quantile(fin,.05):.4f} "
              f"P(>=.72)={(fin>=0.72).mean():.3f} bust%="
              f"{'/'.join(f'{bu[t]/nb*100:.1f}' for t in TIERS)}")
    rep["named"] = rows

    # ---------------- GBM memorisation: in-sample vs out-of-fold R2
    print("\n  GBM head memorisation (fit on 1,760, 6 regression heads):")
    tr = lab.train_idx
    gp = _gbm_params(DEPLOYED_EXP)
    knn_fit, _, _, _ = lab._knn_family(tr, lab.dev_idx, lab.targets)
    head = lab.fit_legacy(tr, 100.0)
    leg = lab.predict_legacy(head, tr)
    inner = np.random.default_rng(0).integers(0, 5, size=len(tr))
    oof = np.zeros((len(tr), 6))
    for k in range(5):
        mm = Ridge(alpha=10.0, solver="sparse_cg").fit(lab.X[tr[inner != k]], lab.targets[tr[inner != k]])
        oof[inner == k] = mm.predict(lab.X[tr[inner == k]])
    oof[:, :3] = np.clip(oof[:, :3], 0, 1)
    Xf = np.hstack([lab.dense[tr], lab.fam_onehot[tr], leg, oof, knn_fit])
    fold = np.random.default_rng(5).integers(0, 5, size=len(tr))
    names = ["s_light", "s_mid", "s_k1", "logc_light", "logc_mid", "logc_k1"]
    gbmr2 = {}
    for k in range(6):
        y = lab.targets[tr, k]
        mfull = HistGradientBoostingRegressor(**gp).fit(Xf, y)
        yin = mfull.predict(Xf)
        yo = np.zeros(len(tr))
        for f in range(5):
            mo = HistGradientBoostingRegressor(**gp).fit(Xf[fold != f], y[fold != f])
            yo[fold == f] = mo.predict(Xf[fold == f])
        r2i = 1 - ((y - yin) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r2o = 1 - ((y - yo) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        gbmr2[names[k]] = dict(r2_in=float(r2i), r2_oof=float(r2o), gap=float(r2i - r2o))
        print(f"    {names[k]:11s} R2 in-sample {r2i:6.3f}  out-of-fold {r2o:6.3f}  "
              f"gap {r2i-r2o:6.3f}")
    rep["gbm_r2"] = gbmr2

    OUT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
