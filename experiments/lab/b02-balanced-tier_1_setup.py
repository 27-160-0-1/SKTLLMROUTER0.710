# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b02 step 1 - baselines, compositions, EB labels, family-posterior policy.

Answers: at 2.0x, does the *expected*-score optimum spread on mid or concentrate
on k1?  The realised-score oracle concentrates (663/150/67), but the realised
label is a 2- or 4-sample binomial draw, so 'concentrate' may be a winner's
curse.  Rebuild the oracle on EB posterior means and compare.
"""
from __future__ import annotations
import importlib.util, sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b02lib", HERE / "b02-balanced-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_SAFETY  # noqa

np.set_printoptions(suppress=True, precision=4)
lab = Lab()
cv, arr = L.load_stage("base")
di = arr["idx"]
ts = lab.true_s[di]; tc = lab.true_c[di]; m = len(di)
Lt = tc[:, 0].sum()

print("\n=== 0. corner policies on dev (balanced cap 2.0) ===")
for name, pick in [("all-light", np.zeros(m, int)), ("all-mid", np.ones(m, int)),
                   ("all-k1", np.full(m, 2))]:
    sc, ratio, ok = L.realised(lab, di, pick, 2.0)
    print(f"  {name:10s} score={sc:.4f} ratio={ratio:.4f} passed={ok}")

print("\n=== 1. EB labels ===")
eb, prior = L.eb_labels(lab)
print("  mean realised (dev) :", ts.mean(axis=0))
print("  mean EB       (dev) :", eb[di].mean(axis=0))
print("  sd   realised (dev) :", ts.std(axis=0))
print("  sd   EB       (dev) :", eb[di].std(axis=0))
d1r = ts[:, 1] - ts[:, 0]; d2r = ts[:, 2] - ts[:, 1]
d1e = eb[di][:, 1] - eb[di][:, 0]; d2e = eb[di][:, 2] - eb[di][:, 1]
print(f"  d1 realised sd={d1r.std():.4f} mean={d1r.mean():+.4f} | "
      f"EB sd={d1e.std():.4f} mean={d1e.mean():+.4f}")
print(f"  d2 realised sd={d2r.std():.4f} mean={d2r.mean():+.4f} | "
      f"EB sd={d2e.std():.4f} mean={d2e.mean():+.4f}")
print("  prior strengths (family, model)->m :")
for f in sorted({k[0] for k in prior}):
    print(f"    {f:16s}", " ".join(f"{prior[(f,mm)][1]:8.2f}" for mm in range(3)))

print("\n=== 2. oracles at each tier: realised labels vs EB labels ===")
rows = []
for lbl_name, S in [("realised", ts), ("EB", eb[di])]:
    for t in TIERS:
        pick = L.P.exact_allocate(S, tc, MULTS[t], 1.0)
        sc_r, ratio, ok = L.realised(lab, di, pick, MULTS[t])
        r = np.arange(m)
        sc_e = eb[di][r, pick].mean()
        cnt = np.bincount(pick, minlength=3)
        rows.append((lbl_name, t, cnt, ratio, sc_r, sc_e))
        print(f"  {lbl_name:8s} {t:9s} L/M/K={cnt} ratio={ratio:.4f} "
              f"realised={sc_r:.4f} EB={sc_e:.4f}")

print("\n=== 3. deployed allocation (dev, honest predictions) ===")
for t in TIERS:
    ps, pc = lab.compose(arr, DEPLOYED_CFG, t)
    pick = lab.allocate(ps, pc, MULTS[t], DEPLOYED_SAFETY[t])
    sc_r, ratio, ok = L.realised(lab, di, pick, MULTS[t])
    r = np.arange(m)
    print(f"  {t:9s} L/M/K={np.bincount(pick,minlength=3)} ratio={ratio:.4f} "
          f"realised={sc_r:.4f} EB={eb[di][r,pick].mean():.4f} passed={ok} "
          f"@s={DEPLOYED_SAFETY[t]}")

print("\n=== 4. family-posterior policy (train-only family means, true cost) ===")
tr = lab.train_idx
fam_mu = {}
for f in sorted(set(lab.fam_arr)):
    rows_f = tr[lab.fam_arr[tr] == f]
    fam_mu[f] = lab.true_s[rows_f].mean(axis=0) if len(rows_f) >= 8 else lab.true_s[tr].mean(axis=0)
fam_s = np.array([fam_mu[f] for f in lab.fam_arr[di]])
for t in TIERS:
    for costname, C in [("true cost", tc)]:
        pick = L.P.exact_allocate(fam_s, C, MULTS[t], 1.0)
        sc_r, ratio, ok = L.realised(lab, di, pick, MULTS[t])
        r = np.arange(m)
        print(f"  {t:9s} {costname:10s} L/M/K={np.bincount(pick,minlength=3)} "
              f"ratio={ratio:.4f} realised={sc_r:.4f} EB={eb[di][r,pick].mean():.4f} passed={ok}")

print("\n=== 5. per-family gains and prices on dev (balanced view) ===")
print(f"  {'family':16s} {'n':>4s} {'d1_real':>8s} {'d1_EB':>8s} {'d2_real':>8s} {'d2_EB':>8s} "
      f"{'c1/c0':>7s} {'c2/c0':>7s} {'eff1':>8s} {'eff2':>8s} {'effchord':>9s}")
for f in sorted(set(lab.fam_arr[di])):
    rr = np.where(lab.fam_arr[di] == f)[0]
    c0 = tc[rr, 0].mean(); c1 = tc[rr, 1].mean(); c2 = tc[rr, 2].mean()
    e1 = (eb[di][rr, 1] - eb[di][rr, 0]).mean() / max(c1 - c0, 1e-12) * c0
    e2 = (eb[di][rr, 2] - eb[di][rr, 1]).mean() / max(c2 - c1, 1e-12) * c0
    ec = (eb[di][rr, 2] - eb[di][rr, 0]).mean() / max(c2 - c0, 1e-12) * c0
    print(f"  {f:16s} {len(rr):4d} {(ts[rr,1]-ts[rr,0]).mean():+8.4f} "
          f"{(eb[di][rr,1]-eb[di][rr,0]).mean():+8.4f} {(ts[rr,2]-ts[rr,1]).mean():+8.4f} "
          f"{(eb[di][rr,2]-eb[di][rr,1]).mean():+8.4f} {c1/c0:7.2f} {c2/c0:7.2f} "
          f"{e1:8.4f} {e2:8.4f} {ec:9.4f}")

np.savez_compressed("reports/lab/b02_eb.npz", eb=eb)
print("\n[saved] reports/lab/b02_eb.npz")
