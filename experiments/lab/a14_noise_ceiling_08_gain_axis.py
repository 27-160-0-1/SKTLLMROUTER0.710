# SPDX-License-Identifier: Apache-2.0
"""a14 step 8 -- the honest exchange rate on the ONLY axis that can matter.

Theorem (verified numerically in the header of the report): the Lagrangian
allocator picks argmax_j (pred_score[i,j] - pen * pred_cost[i,j]/L).  Adding a
per-item constant delta_i to all three arms leaves every argmax and every
predicted-cost total unchanged, hence the selection is *identical* for every
penalty and every budget.  So the item-level component of the score prediction
is exactly worthless; only the between-arm differences (gains) can move the
score.  This script therefore sweeps gain quality only, paired across p-draws.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_WEIGHT, TIER_MULT              # noqa: E402
from a14_noise_ceiling_05_realistic import Alloc, counts_full, counts_boot, SAF_GRID  # noqa: E402

dv = load_split("dev")
N = len(dv)
P43 = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
D = np.load(HERE / "_a14_cache/pdraws_dev.npz")
PD = D["p"].astype(np.float64)
PHAT = D["phat"].astype(np.float64)
SAFE43 = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
W1 = counts_full()
# real-world gain corr of the deployed predictions (exact back-out, step 6)
REAL_G0 = np.array([0.133, 0.406])


def regimes(pred_of_tier, eval_score, Wb):
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
        tot["Sfix"] += TIER_WEIGHT[t] * (sc[0] if ok[0] else 0.0)
        best_d, best_b = (-1, None), (-1, None, None)
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
        det[t] = dict(bust_fix=float(np.mean(~okb)), s_boot=best_b[1], bust_boot=best_b[2],
                      s_dev=best_d[1])
    return tot, det


def blend_gain(pred, p, lam):
    dp = p - p.mean(1)[:, None]
    dq = pred - pred.mean(1)[:, None]
    return pred + lam * (dp - dq)


def gcorr(pred, p):
    out = []
    for (a, b) in ((0, 1), (1, 2)):
        gp = pred[:, b] - pred[:, a]
        gt = p[:, b] - p[:, a]
        out.append(np.corrcoef(gp, gt)[0, 1])
    return np.array(out)


def main():
    Wb = counts_boot(300, 4242)
    reps = 6
    draws = [PD[i] for i in np.linspace(0, PD.shape[0] - 1, reps).astype(int)]
    lams = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
    print("=" * 122)
    print("HONEST EXCHANGE RATE ON THE GAIN AXIS (paired p-draws, expected-score evaluation)")
    print("scale: the deployed predictor sits at simulated gain corr 0.106/0.379; the exact")
    print("real-world back-out is 0.133/0.406, so the 'real-eq' columns rescale by 1.25 / 1.07")
    print("=" * 122)
    hdr = (f"{'lam':>5s} {'gsim m-l':>9s} {'gsim k-m':>9s} {'greal m-l':>10s} {'greal k-m':>10s} | "
           f"{'S1':>7s} {'Sdev':>7s} {'Sboot':>7s} {'Sfix':>7s} {'bustfix%':>9s} | "
           f"{'d(Sboot)':>9s} {'safety(boot)':>16s}")
    print(hdr)
    rows = []
    base = None
    for lam in lams:
        acc = {k: [] for k in ("S1", "Sfix", "Sdev", "Sboot")}
        gs, bf, sb = [], [], []
        for p in draws:
            mk = lambda t, p=p, lam=lam: blend_gain(P43[f"score_{t}"], p, lam)
            tot, det = regimes(mk, p, Wb)
            for k in acc:
                acc[k].append(tot[k])
            gs.append(gcorr(mk("fast"), p))
            bf.append(np.mean([det[t]["bust_fix"] for t in TIERS]))
            sb.append([det[t]["s_boot"] for t in TIERS])
        g = np.mean(gs, 0); sbm = np.mean(sb, 0)
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        if base is None:
            base = m["Sboot"]
        gr = g * (REAL_G0 / np.array([0.106, 0.379]))
        print(f"{lam:5.2f} {g[0]:9.3f} {g[1]:9.3f} {gr[0]:10.3f} {gr[1]:10.3f} | "
              f"{m['S1']:7.4f} {m['Sdev']:7.4f} {m['Sboot']:7.4f} {m['Sfix']:7.4f} "
              f"{100*np.mean(bf):9.1f} | {m['Sboot']-base:+9.4f} "
              f"{'/'.join(f'{x:.2f}' for x in sbm):>16s}")
        rows.append((lam, g, gr, m))

    # ---- what is needed for +0.018 (0.7017 -> 0.72) on the Sboot / Sdev axes
    print("\n" + "=" * 122)
    print("REQUIRED GAIN QUALITY FOR +0.0183 (deployed held-out 0.7017 -> 0.7200)")
    print("=" * 122)
    for key in ("Sboot", "Sdev", "S1"):
        xs = np.array([r[3][key] for r in rows])
        b = xs[0]
        target = b + 0.0183
        if xs.max() < target:
            print(f"  {key}: unreachable on this axis (max {xs.max():.4f})")
            continue
        i = int(np.argmax(xs >= target))
        lo, hi = i - 1, i
        w = (target - xs[lo]) / (xs[hi] - xs[lo])
        lam = rows[lo][0] + w * (rows[hi][0] - rows[lo][0])
        g = rows[lo][1] + w * (rows[hi][1] - rows[lo][1])
        gr = rows[lo][2] + w * (rows[hi][2] - rows[lo][2])
        print(f"  {key}: lam={lam:.3f}  gain corr (sim) {g[0]:.3f}/{g[1]:.3f}  "
              f"(real-eq) {gr[0]:.3f}/{gr[1]:.3f}   "
              f"[deployed real 0.133/0.406 -> needs x{gr[0]/0.133:.2f} / x{gr[1]/0.406:.2f}]")

    # ---- marginal exchange rate at the operating point
    print("\nlocal slope near the operating point (Sboot):")
    for i in range(1, 5):
        dl = rows[i][3]["Sboot"] - rows[0][3]["Sboot"]
        dg = rows[i][2][0] - rows[0][2][0]
        print(f"  lam={rows[i][0]:.2f}: d(final)={dl:+.4f} for d(gain corr m-l)={dg:+.3f} "
              f"-> {dl/dg if dg else float('nan'):+.4f} final per 0.1 gain-corr: "
              f"{0.1*dl/dg if dg else float('nan'):+.4f}")


if __name__ == "__main__":
    main()
