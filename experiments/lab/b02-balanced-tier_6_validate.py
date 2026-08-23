# SPDX-License-Identifier: Apache-2.0
"""b02 step 6 - is the balanced-tier kappa gain real, or selection bias?

(a) per-seed stability of the kappa2 EV curve;
(b) nested split-half: choose kappa2 AND the safety on half of the OOF rows,
    score both the chosen and the baseline policy on the other half;
(c) repeat on the legacy-OOF-meta stage (BRIEF2 C1) to check composition;
(d) full 3-tier honest run of the shipped candidate.
"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b02lib", HERE / "b02-balanced-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, MULTS, TIERS, W, DEPLOYED_CFG, DEPLOYED_SAFETY  # noqa
import protocol as P  # noqa
import bench2 as B  # noqa

MULT = MULTS["balanced"]
GRID = np.arange(0.60, 1.201, 0.005)
K2 = [0.85, 1.0, 1.10, 1.24, 1.35, 1.50, 1.75, 2.00, 2.50]
lab = Lab()


def curve(lab, cv, k2, smp, grid=GRID, cfg=None):
    ps, pc = lab.compose(cv, cfg or DEPLOYED_CFG, "balanced")
    pc = pc.copy(); pc[:, 2] *= k2
    ci = cv["idx"]
    ev, bu, raw = P.safety_curve(ps[smp], pc[smp], lab.true_s[ci][smp], lab.true_c[ci][smp],
                                 MULT, grid)
    return ev, bu


for tag in ("base", "legoof"):
    cv, arr = L.load_stage(tag)
    ci = cv["idx"]; m = len(ci)
    print(f"\n################ stage='{tag}' ################")
    print("=== (a) per-seed kappa2 EV curve (balanced) ===")
    print(f"  {'k2':>5s} " + " ".join(f"{'seed'+str(s):>18s}" for s in (7, 17, 23)) + "   pooled")
    pooled = {}
    for k2 in K2:
        cells = []
        for s in (7, 17, 23):
            smp = np.asarray(lab.samples_for(m, s, 400, 880))
            ev, bu = curve(lab, cv, k2, smp)
            gi = int(np.argmax(ev))
            cells.append(f"{ev[gi]:.5f}@{GRID[gi]:.3f}")
        smp = np.concatenate([np.asarray(lab.samples_for(m, s, 400, 880)) for s in (7, 17, 23)])
        ev, bu = curve(lab, cv, k2, smp)
        gi = int(np.argmax(ev))
        pooled[k2] = (float(ev[gi]), float(GRID[gi]), float(bu[gi]))
        print(f"  {k2:5.2f} " + " ".join(f"{c:>18s}" for c in cells) +
              f"   {ev[gi]:.5f}@{GRID[gi]:.3f} bust={bu[gi]*100:.2f}%")
    b0 = pooled[1.0][0]
    print("  deltas vs k2=1.0: " + " ".join(f"{k}:{pooled[k][0]-b0:+.4f}" for k in K2))

    print("\n=== (b) nested split-half (choose k2+safety on A, score on B) ===")
    rng = np.random.default_rng(2024)
    wins = {k: 0 for k in K2}
    gains, chosen, oracles = [], [], []
    NS = 24
    for rep in range(NS):
        perm = rng.permutation(m)
        A, Bx = perm[: m // 2], perm[m // 2:]
        sa = rng.integers(0, len(A), size=(300, 880)); sa = A[sa]
        sb = rng.integers(0, len(Bx), size=(300, 880)); sb = Bx[sb]
        evA = {};
        for k2 in K2:
            ev, bu = curve(lab, cv, k2, sa)
            gi = int(np.argmax(ev)); evA[k2] = (float(ev[gi]), float(GRID[gi]))
        kstar = max(evA, key=lambda k: evA[k][0])
        wins[kstar] += 1
        evB = {}
        for k2 in (1.0, kstar):
            ev, bu = curve(lab, cv, k2, sb, grid=np.array([evA[k2][1]]))
            evB[k2] = float(ev[0])
        # oracle-on-B best k2 (upper bound, for the bias size)
        bestB = -1
        for k2 in K2:
            ev, bu = curve(lab, cv, k2, sb)
            bestB = max(bestB, float(ev.max()))
        gains.append(evB[kstar] - evB[1.0]); chosen.append(kstar)
        oracles.append(bestB - evB[1.0])
    g = np.array(gains); o = np.array(oracles)
    print(f"  chosen k2 histogram: " + " ".join(f"{k}:{wins[k]}" for k in K2 if wins[k]))
    print(f"  honest transferred gain  mean={g.mean():+.5f} sd={g.std():.5f} "
          f"pos={int((g>0).sum())}/{NS}")
    print(f"  same-half oracle gain    mean={o.mean():+.5f}  -> selection bias "
          f"{o.mean()-g.mean():+.5f}")

    print("\n=== (c) fixed k2=1.5 transferred (no selection at all) ===")
    gains15 = []
    rng = np.random.default_rng(777)
    for rep in range(NS):
        perm = rng.permutation(m)
        A, Bx = perm[: m // 2], perm[m // 2:]
        sa = A[rng.integers(0, len(A), size=(300, 880))]
        sb = Bx[rng.integers(0, len(Bx), size=(300, 880))]
        ev, _ = curve(lab, cv, 1.5, sa); s15 = float(GRID[int(np.argmax(ev))])
        ev, _ = curve(lab, cv, 1.0, sa); s10 = float(GRID[int(np.argmax(ev))])
        e15 = float(curve(lab, cv, 1.5, sb, grid=np.array([s15]))[0][0])
        e10 = float(curve(lab, cv, 1.0, sb, grid=np.array([s10]))[0][0])
        gains15.append(e15 - e10)
    g15 = np.array(gains15)
    print(f"  k2=1.5 vs 1.0 transferred mean={g15.mean():+.5f} sd={g15.std():.5f} "
          f"pos={int((g15>0).sum())}/{NS}")

print("\n\n################ (d) full 3-tier honest runs ################")
for tag in ("base", "legoof"):
    cv, arr = L.load_stage(tag)
    r0 = B.run(lab, cv, arr, DEPLOYED_CFG, label=f"[{tag}] baseline")
    for k2 in (1.24, 1.5, 2.0):
        r = B.run(lab, cv, arr, DEPLOYED_CFG, transform=L.kappa_transform(1.0, k2, ("balanced",)),
                  label=f"[{tag}] bal kappa2={k2}")
    for k2 in (1.5,):
        r = B.run(lab, cv, arr, DEPLOYED_CFG,
                  transform=L.kappa_transform(1.0, k2, ("balanced", "premium")),
                  label=f"[{tag}] bal+prem kappa2={k2}")
