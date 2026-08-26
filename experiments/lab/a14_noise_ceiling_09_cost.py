# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a14 step 9 -- the cost side of the honest exchange rate.

Q1  How much does the realised budget ratio move across resamples of 880?
Q2  What bust probability does the DEPLOYED configuration actually carry?
Q3  cost log-RMSE -> EV-optimal safety -> final score  (expected-score evaluation)
Q4  the interaction with a better gain head (E42's mechanism, quantified)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_WEIGHT, TIER_MULT, MODEL_IDS   # noqa: E402
from a14_noise_ceiling_05_realistic import Alloc, counts_full, counts_boot  # noqa: E402

dv = load_split("dev")
N = len(dv)
P43 = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
D = np.load(HERE / "_a14_cache/pdraws_dev.npz")
PD = D["p"].astype(np.float64)
PHAT = D["phat"].astype(np.float64)
SAFE43 = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
W1 = counts_full()
SAF = np.round(np.arange(0.20, 1.3001, 0.005), 4)


def blend_gain(pred, p, lam):
    dp = p - p.mean(1)[:, None]
    dq = pred - pred.mean(1)[:, None]
    return pred + lam * (dp - dq)


def q1_q2(Wb):
    print("=" * 110)
    print("Q1/Q2 -- budget-ratio variability and the real bust risk of the deployed config")
    print("=" * 110)
    print("deployed cost model quality (dev):")
    for t in ("fast",):
        for j, m in enumerate(MODEL_IDS):
            pc = P43[f"cost_{t}"][:, j]; tc = dv.cost[:, j]
            lr = np.log(pc) - np.log(tc)
            print(f"  {m:12s} log-bias={lr.mean():+.3f} log-sd={lr.std():.3f} "
                  f"log-RMSE={np.sqrt((lr**2).mean()):.3f} sum(pred)/sum(true)={pc.sum()/tc.sum():.4f}")
    print()
    print("k1 cost concentration: top 1% of dev items carry 18.3% of the total think cost")
    print(f"{'tier':9s} {'safe':>5s} {'ratio(dev)':>11s} {'cap':>5s} {'headr%':>7s} "
          f"{'sd(ratio)':>10s} {'cv%':>6s} {'p05':>7s} {'p50':>7s} {'p95':>7s} {'bust%':>7s}")
    for t in TIERS:
        A = Alloc(P43[f"score_{t}"], P43[f"cost_{t}"], dv.cost, PHAT)
        sc, tr, ok = A.run(W1, TIER_MULT[t], SAFE43[t])
        scb, trb, okb = A.run(Wb, TIER_MULT[t], SAFE43[t])
        q = np.percentile(trb, [5, 50, 95])
        print(f"{t:9s} {SAFE43[t]:5.2f} {tr[0]:11.4f} {TIER_MULT[t]:5.2f} "
              f"{100*(1-tr[0]/TIER_MULT[t]):6.1f}% {trb.std():10.4f} "
              f"{100*trb.std()/trb.mean():5.1f}% {q[0]:7.3f} {q[1]:7.3f} {q[2]:7.3f} "
              f"{100*np.mean(~okb):6.1f}%")
    print("\nbust probability vs safety (deployed predictions, bootstrap of dev-880):")
    print(f"{'safety':>7s} " + " ".join(f"{t:>10s}" for t in TIERS))
    for s in (0.70, 0.75, 0.80, 0.85, 0.87, 0.90, 0.93, 0.95, 0.98, 1.00):
        row = []
        for t in TIERS:
            A = Alloc(P43[f"score_{t}"], P43[f"cost_{t}"], dv.cost, PHAT)
            scb, trb, okb = A.run(Wb, TIER_MULT[t], float(s))
            row.append(f"{100*np.mean(~okb):9.1f}%")
        print(f"{s:7.2f} " + " ".join(row))


def ev_curve(score_of_tier, cost_of_tier, eval_score, Wb, label):
    """EV-optimal safety per tier + the resulting weighted EV and dev-point score."""
    tot_ev = 0.0
    tot_pt = 0.0
    det = []
    for t in TIERS:
        A = Alloc(score_of_tier(t), cost_of_tier(t), dv.cost, eval_score)
        best = (-1, None, None)
        for s in SAF:
            a, b, c = A.run(Wb, TIER_MULT[t], float(s))
            ev = float(np.mean(np.where(c, a, 0.0)))
            if ev > best[0]:
                best = (ev, float(s), float(np.mean(~c)))
        tot_ev += TIER_WEIGHT[t] * best[0]
        a1, b1, c1 = A.run(W1, TIER_MULT[t], best[1])
        tot_pt += TIER_WEIGHT[t] * (float(a1[0]) if c1[0] else 0.0)
        det.append((t, best[1], best[0], best[2], float(b1[0])))
    return tot_ev, tot_pt, det


RHO_C = 0.30   # measured cross-model correlation of the deployed log-cost errors
               # (0.435 / 0.186 / 0.284 for l-m / l-k / m-k)


def synth_cost(rng, sigma, debias=True, rho=RHO_C):
    """log c_hat = log c + sigma*z, z correlated across models like the real errors,
    optionally rescaled so each model's SUM matches (removes the Jensen bias)."""
    z = rng.standard_normal(dv.cost.shape)
    common = rng.standard_normal((dv.cost.shape[0], 1))
    z = np.sqrt(rho) * common + np.sqrt(1 - rho) * z
    C = dv.cost * np.exp(sigma * z)
    # the deployed cost head is 100% monotone (light<mid<k1 on every dev row, as is
    # the truth on 99.5%); an unconstrained synthetic model would predict mid cheaper
    # than light and force un-droppable upgrades, so repair the ordering.
    C = np.sort(C, axis=1)
    if debias:
        C = C * (dv.cost.sum(0) / C.sum(0))[None, :]
    return C


def q3_q4(Wb):
    print("\n" + "=" * 110)
    print("Q3 -- cost log-RMSE -> EV-optimal safety -> honest final score")
    print("    (score side held at the deployed E43 predictions; evaluation = E[p|k])")
    print("=" * 110)
    rng = np.random.default_rng(31337)
    print(f"{'cost model':28s} {'safety f/b/p':>16s} {'bust% f/b/p':>18s} "
          f"{'EV(final)':>10s} {'dev point':>10s}")

    def show(label, cf, sof=None, ev_sc=PHAT):
        sof = sof or (lambda t: P43[f"score_{t}"])
        ev, pt, det = ev_curve(sof, cf, ev_sc, Wb, label)
        print(f"{label:28s} {'/'.join(f'{d[1]:.2f}' for d in det):>16s} "
              f"{'/'.join(f'{100*d[3]:.1f}' for d in det):>18s} {ev:10.4f} {pt:10.4f}")
        return ev, pt

    show("deployed cost model", lambda t: P43[f"cost_{t}"])
    show("+ per-model sum calibrated",
         lambda t: P43[f"cost_{t}"] * (dv.cost.sum(0) / P43[f"cost_{t}"].sum(0))[None, :])
    for sig in (0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0):
        evs, pts = [], []
        for r in range(3):
            C = synth_cost(rng, sig)
            e, p = show(f"synthetic log-sd {sig:.2f} (rep{r})", lambda t, C=C: C)
            evs.append(e); pts.append(p)
        print(f"{'  -> mean':28s} {'':16s} {'':18s} {np.mean(evs):10.4f} {np.mean(pts):10.4f}")
    show("TRUE cost", lambda t: dv.cost)

    print("\n" + "=" * 110)
    print("Q4 -- interaction: a better gain head at each cost quality (E42's mechanism)")
    print("=" * 110)
    p = PD[0]
    print(f"{'cost model':28s} {'gain lam':>9s} {'safety f/b/p':>16s} {'EV(final)':>10s} "
          f"{'d vs lam=0':>11s}")
    for cname, cf in (("deployed", lambda t: P43[f"cost_{t}"]),
                      ("sum-calibrated", lambda t: P43[f"cost_{t}"] *
                       (dv.cost.sum(0) / P43[f"cost_{t}"].sum(0))[None, :]),
                      ("synthetic log-sd 0.30", lambda t, C=synth_cost(
                          np.random.default_rng(5), 0.30): C),
                      ("true", lambda t: dv.cost)):
        base = None
        for lam in (0.0, 0.06, 0.15, 0.50):
            sof = lambda t, lam=lam: blend_gain(P43[f"score_{t}"], p, lam)
            ev, pt, det = ev_curve(sof, cf, p, Wb, "")
            if base is None:
                base = ev
            print(f"{cname:28s} {lam:9.2f} {'/'.join(f'{d[1]:.2f}' for d in det):>16s} "
                  f"{ev:10.4f} {ev-base:+11.4f}")


if __name__ == "__main__":
    Wb = counts_boot(300, 4242)
    q1_q2(Wb)
    q3_q4(Wb)
