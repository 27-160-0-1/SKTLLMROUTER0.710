# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a14 step 12 -- which of the two gains carries the value, and per-family."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_split, TIERS, TIER_WEIGHT, TIER_MULT              # noqa: E402
from a14_noise_ceiling_05_realistic import Alloc, counts_full, counts_boot  # noqa: E402
from ossp_router.similarity import classify_family                         # noqa: E402

dv = load_split("dev")
N = len(dv)
P43 = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
D = np.load(HERE / "_a14_cache/pdraws_dev.npz")
PD = D["p"].astype(np.float64)
W1 = counts_full()
SAF = np.round(np.arange(0.30, 1.2001, 0.01), 3)
FAM = np.array([classify_family(t) for t in dv.texts])


def ev_and_point(sof, Wb, eval_score):
    ev = pt = s1 = 0.0
    safs = []
    for t in TIERS:
        A1 = Alloc(sof(t), dv.cost, dv.cost, eval_score)
        a, b, c = A1.run(W1, TIER_MULT[t], 1.0)
        s1 += TIER_WEIGHT[t] * (a[0] if c[0] else 0.0)
        A = Alloc(sof(t), P43[f"cost_{t}"], dv.cost, eval_score)
        best = (-1, None)
        for s in SAF:
            a2, b2, c2 = A.run(Wb, TIER_MULT[t], float(s))
            e = float(np.mean(np.where(c2, a2, 0.0)))
            if e > best[0]:
                best = (e, float(s))
        ev += TIER_WEIGHT[t] * best[0]
        a3, b3, c3 = A.run(W1, TIER_MULT[t], best[1])
        pt += TIER_WEIGHT[t] * (float(a3[0]) if c3[0] else 0.0)
        safs.append(best[1])
    return ev, pt, s1, safs


def fix_pair(pred, p, lam, pair):
    """move only ONE between-arm difference toward the truth."""
    out = pred.copy()
    a, b = pair
    d = (p[:, b] - p[:, a]) - (pred[:, b] - pred[:, a])
    out[:, b] = out[:, b] + lam * d
    if b == 1:                       # keep k1 gap to mid unchanged
        out[:, 2] = out[:, 2] + lam * d
    return out


def fam_only(pred, p, lam, fams):
    m = np.isin(FAM, fams)
    out = pred.copy()
    dp = p - p.mean(1)[:, None]
    dq = pred - pred.mean(1)[:, None]
    out[m] = pred[m] + lam * (dp[m] - dq[m])
    return out


def main():
    Wb = counts_boot(300, 4242)
    p = PD[0]
    print("=" * 100)
    print("WHICH GAIN?  (evaluate expected score on the latent p; safety EV-retuned)")
    print("=" * 100)
    base = ev_and_point(lambda t: P43[f"score_{t}"], Wb, p)
    print(f"{'variant':42s} {'EV':>8s} {'point':>8s} {'S1':>8s} {'dEV':>8s} {'dS1':>8s}")
    print(f"{'baseline (deployed score head)':42s} {base[0]:8.4f} {base[1]:8.4f} "
          f"{base[2]:8.4f} {0:8.4f} {0:8.4f}")
    for lam in (0.5, 1.0):
        for pair, nm in (((0, 1), "mid-light gain"), ((1, 2), "k1-mid gain")):
            r = ev_and_point(lambda t, lam=lam, pair=pair:
                             fix_pair(P43[f"score_{t}"], p, lam, pair), Wb, p)
            print(f"{f'{nm} -> truth x{lam}':42s} {r[0]:8.4f} {r[1]:8.4f} {r[2]:8.4f} "
                  f"{r[0]-base[0]:+8.4f} {r[2]-base[2]:+8.4f}")
    r = ev_and_point(lambda t: P43[f"score_{t}"] + 1.0 * (
        (p - p.mean(1)[:, None]) - (P43[f"score_{t}"] - P43[f"score_{t}"].mean(1)[:, None])),
        Wb, p)
    print(f"{'both gains -> truth':42s} {r[0]:8.4f} {r[1]:8.4f} {r[2]:8.4f} "
          f"{r[0]-base[0]:+8.4f} {r[2]-base[2]:+8.4f}")

    print("\n" + "=" * 100)
    print("WHICH FAMILY?  perfect gains inside one family only (rest unchanged)")
    print("=" * 100)
    print(f"{'family':22s} {'N':>4s} {'EV':>8s} {'dEV':>8s} {'S1':>8s} {'dS1':>8s} "
          f"{'dS1 per item':>13s}")
    for f_ in sorted(set(FAM)):
        n = int((FAM == f_).sum())
        r = ev_and_point(lambda t, f_=f_: fam_only(P43[f"score_{t}"], p, 1.0, [f_]), Wb, p)
        print(f"{f_:22s} {n:4d} {r[0]:8.4f} {r[0]-base[0]:+8.4f} {r[2]:8.4f} "
              f"{r[2]-base[2]:+8.4f} {(r[2]-base[2])/n*1000:13.3f}")


if __name__ == "__main__":
    main()
