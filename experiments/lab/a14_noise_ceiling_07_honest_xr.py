# SPDX-License-Identifier: Apache-2.0
"""a14 step 7 -- THE honest exchange rate.

The BRIEF blends the deployed score predictions toward the REALISED score and
scores on the realised score.  That double-counts label noise: the blended
predictor sees the very noise it is then rewarded for.  The honest version
blends toward the LATENT p and evaluates the EXPECTED score:

    pred(lam) = (1-lam) * E43_prediction + lam * p          (p = posterior draw)
    value     = sum_i p[i, sel_i] / N,  budget checked on the TRUE cost

Also decomposes the improvement into
    LEVEL-only  : add lam * (pbar_true - pbar_pred) to all three arms  (gain unchanged)
    GAIN-only   : add lam * ((p_j - pbar_true) - (pred_j - pbar_pred)) (level unchanged)
which says which half of a better score head actually pays.

Safety regimes:
    S1    true cost, safety 1.0                       (pure score axis)
    Sfix  deployed cost model, deployed safety .98/.87/.85 + bootstrap bust prob
    Sdev  deployed cost model, safety re-tuned on dev  (optimistic, = BRIEF)
    Sboot deployed cost model, safety by 880-bootstrap EV (E09 protocol)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_WEIGHT, TIER_MULT, MODEL_IDS   # noqa: E402
from a14_noise_ceiling_05_realistic import Alloc, counts_full, counts_boot, SAF_GRID  # noqa: E402

dv = load_split("dev")
N = len(dv)
P43 = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
CACHE = HERE / "_a14_cache"
D = np.load(CACHE / "pdraws_dev.npz")
PD = D["p"].astype(np.float64)
PHAT = D["phat"].astype(np.float64)
SAFE43 = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
W1 = counts_full()


def cell_moments(x, nn):
    mu = float(x.mean())
    inv = float((1.0 / nn).mean())
    return mu, (float(np.mean(x ** 2)) - inv * mu) / (1.0 - inv) - mu * mu


VP = np.array([cell_moments(dv.score[:, j], dv.ngen[:, j])[1] for j in range(3)])
CP = np.cov(dv.score.T).copy()
np.fill_diagonal(CP, VP)
VG = {(0, 1): VP[0] + VP[1] - 2 * CP[0, 1], (1, 2): VP[1] + VP[2] - 2 * CP[1, 2]}


def regimes(pred_of_tier, eval_score, Wb):
    """returns dict regime -> weighted final (+ diagnostics)."""
    tot = dict(S1=0.0, Sfix=0.0, Sdev=0.0, Sboot=0.0)
    det = {}
    for t in TIERS:
        ps = pred_of_tier(t)
        At = Alloc(ps, dv.cost, dv.cost, eval_score)
        sc, tr, ok = At.run(W1, TIER_MULT[t], 1.0)
        tot["S1"] += TIER_WEIGHT[t] * (sc[0] if ok[0] else 0.0)
        Ap = Alloc(ps, P43[f"cost_{t}"], dv.cost, eval_score)
        sc, tr, ok = Ap.run(W1, TIER_MULT[t], SAFE43[t])
        scb, trb, okb = Ap.run(Wb, TIER_MULT[t], SAFE43[t])
        bust = float(np.mean(~okb))
        tot["Sfix"] += TIER_WEIGHT[t] * (sc[0] if ok[0] else 0.0)
        best_d = (-1, None)
        best_b = (-1, None, None)
        for s in SAF_GRID:
            a, b, c = Ap.run(W1, TIER_MULT[t], float(s))
            if c[0] and a[0] > best_d[0]:
                best_d = (float(a[0]), float(s))
            a2, b2, c2 = Ap.run(Wb, TIER_MULT[t], float(s))
            ev = float(np.mean(np.where(c2, a2, 0.0)))
            if ev > best_b[0]:
                best_b = (ev, float(s), float(np.mean(~c2)))
        tot["Sdev"] += TIER_WEIGHT[t] * best_d[0]
        tot["Sboot"] += TIER_WEIGHT[t] * best_b[0]
        det[t] = dict(fix_ratio=float(tr[0]), fix_pass=bool(ok[0]), fix_bust=bust,
                      s_dev=best_d[1], s_boot=best_b[1], bust_boot=best_b[2])
    return tot, det


def corrs(pred, p):
    lvl = [np.corrcoef(pred[:, j], p[:, j])[0, 1] for j in range(3)]
    g = []
    for (a, b) in ((0, 1), (1, 2)):
        gp = pred[:, b] - pred[:, a]
        gt = p[:, b] - p[:, a]
        g.append(np.corrcoef(gp, gt)[0, 1] if gp.std() > 1e-12 else np.nan)
    return lvl, g


def sweep(name, build, lams, Wb, reps=3, seed=5150):
    rng = np.random.default_rng(seed)
    print(f"\n{'lam':>5s} {'corr(pred,p) l/m/k':>21s} {'gain m-l':>9s} {'gain k-m':>9s} | "
          f"{'S1':>7s} {'Sfix':>7s} {'bust%':>6s} {'Sdev':>7s} {'Sboot':>7s} | {'safety(boot)':>16s}")
    rows = []
    for lam in lams:
        acc = {k: [] for k in ("S1", "Sfix", "Sdev", "Sboot")}
        cl, cg, bu, sb = [], [], [], []
        for r in range(reps):
            p = PD[rng.integers(PD.shape[0])]
            mk = lambda t, p=p, lam=lam: build(P43[f"score_{t}"], p, lam)
            tot, det = regimes(mk, p, Wb)
            for k in acc:
                acc[k].append(tot[k])
            a, b = corrs(mk("fast"), p)
            cl.append(a); cg.append(b)
            bu.append(np.mean([det[t]["fix_bust"] for t in TIERS]))
            sb.append([det[t]["s_boot"] for t in TIERS])
        cl = np.mean(cl, 0); cg = np.mean(cg, 0); sb = np.mean(sb, 0)
        print(f"{lam:5.2f} {'/'.join(f'{c:.3f}' for c in cl):>21s} {cg[0]:9.3f} {cg[1]:9.3f} | "
              f"{np.mean(acc['S1']):7.4f} {np.mean(acc['Sfix']):7.4f} {100*np.mean(bu):6.1f} "
              f"{np.mean(acc['Sdev']):7.4f} {np.mean(acc['Sboot']):7.4f} | "
              f"{'/'.join(f'{x:.2f}' for x in sb):>16s}")
        rows.append((lam, cl, cg, {k: np.mean(v) for k, v in acc.items()}))
    return rows


def blend_all(pred, p, lam):
    return (1 - lam) * pred + lam * p


def blend_level(pred, p, lam):
    """move only the item-level mean toward the truth; leave the gains alone."""
    d = p.mean(1) - pred.mean(1)
    return pred + lam * d[:, None]


def blend_gain(pred, p, lam):
    """move only the between-model differences toward the truth."""
    dp = p - p.mean(1)[:, None]
    dq = pred - pred.mean(1)[:, None]
    return pred + lam * (dp - dq)


def main():
    Wb = counts_boot(200, 4242)
    print("=" * 118)
    print("BASELINE: deployed E43 predictions, honest evaluation (expected score under latent p)")
    print("=" * 118)
    tot, det = regimes(lambda t: P43[f"score_{t}"], PHAT, Wb)
    print(f"  S1={tot['S1']:.4f}  Sfix={tot['Sfix']:.4f}  Sdev={tot['Sdev']:.4f} "
          f"Sboot={tot['Sboot']:.4f}")
    for t in TIERS:
        print(f"    {t:9s} ratio@fix={det[t]['fix_ratio']:.3f} pass={det[t]['fix_pass']} "
              f"bootstrap bust={100*det[t]['fix_bust']:.1f}%  s_dev={det[t]['s_dev']:.2f} "
              f"s_boot={det[t]['s_boot']:.2f} (bust {100*det[t]['bust_boot']:.1f}%)")
    a, b = corrs(P43["score_fast"], PHAT)
    print(f"  [note] corr against E[p|k] understates; the exact back-out (step 6) gives "
          f"level 0.441/0.519/0.452, gain 0.133/0.406")

    print("\n" + "=" * 118)
    print("A) HONEST EXCHANGE RATE -- blend the whole prediction toward the latent p")
    print("=" * 118)
    sweep("all", blend_all, [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00], Wb)

    print("\n" + "=" * 118)
    print("B) LEVEL-ONLY improvement (per-item mean of the 3 arms moves to the truth,")
    print("   between-arm gains untouched)")
    print("=" * 118)
    sweep("level", blend_level, [0.0, 0.25, 0.50, 0.75, 1.00], Wb)

    print("\n" + "=" * 118)
    print("C) GAIN-ONLY improvement (between-arm differences move to the truth,")
    print("   per-item level untouched)")
    print("=" * 118)
    sweep("gain", blend_gain, [0.0, 0.10, 0.25, 0.50, 0.75, 1.00], Wb)


if __name__ == "__main__":
    main()
