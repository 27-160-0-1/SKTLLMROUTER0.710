# SPDX-License-Identifier: Apache-2.0
"""a11 step 16 - feasibility of a long-output veto: oracle value, false-positive
cost, and what a PREDICTED-cost veto (available today) actually buys.

All numbers are 3 seeds x R bootstrap batches of 880, rule = max EV subject to
bust probability <= 1%, safety grid 0.60..2.00 (the wide grid matters: with a
veto the optimal safety exceeds 1.0).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import importlib.util

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT
_spec = importlib.util.spec_from_file_location("a11path", HERE / "a11-postproc-router_5_path.py")
a11path = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(a11path)

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv)
opg = dv.otok[:, 2] / dv.ngen[:, 2]
G = np.arange(0.60, 2.001, 0.01)
R = 300


def samples(seed, R=R):
    g = np.random.default_rng(seed)          # ONE generator, R independent draws
    return [g.integers(0, n, size=n) for _ in range(R)]


SM = [samples(s) for s in (7, 17, 23)]


def mkpath(tier, veto=None, kappa2=1.0):
    pc = P[f"cost_{tier}"] * np.array([1.0, 1.0, kappa2])[None, :]
    pc = pc.copy()
    if veto is not None:
        pc[veto, 2] = 1e12
    pa = a11path.path_arrays(P[f"score_{tier}"], pc, dv.score, dv.cost)
    pa["bp"] = pc[:, 0]
    return pa


def frontier(tier, pa, smp):
    mult = TIER_MULT[tier]
    ev = np.zeros(len(G)); bu = np.zeros(len(G))
    for S in smp:
        m = len(S)
        bp = pa["bp"][S].sum(); bt = dv.cost[S, 0].sum(); bs = dv.score[S, 0].sum()
        v = pa["valid"][S].ravel()
        o = np.argsort(-pa["eff"][S].ravel()[v], kind="stable")
        cp = np.concatenate([[0.0], np.cumsum(pa["dc_p"][S].ravel()[v][o])])
        ct = np.concatenate([[0.0], np.cumsum(pa["dc_t"][S].ravel()[v][o])])
        st = np.concatenate([[0.0], np.cumsum(pa["ds_t"][S].ravel()[v][o])])
        for gi, s in enumerate(G):
            cap = bp * (max(1.0, mult * s) - 1.0)
            k = int(np.searchsorted(cp, cap, side="right")) - 1
            if (bt + ct[k]) / bt > mult:
                bu[gi] += 1
            else:
                ev[gi] += (bs + st[k]) / m
    return ev / len(smp), bu / len(smp)


def weval(veto=None, kappa2=1.0):
    tot = 0.0; parts = []
    for t in TIERS:
        pa = mkpath(t, veto, kappa2)
        ev = np.zeros(len(G)); bu = np.zeros(len(G))
        for smp in SM:
            e, b = frontier(t, pa, smp)
            ev += e / 3; bu += b / 3
        ok = np.flatnonzero(bu <= 0.01)
        i = ok[int(np.argmax(ev[ok]))] if len(ok) else int(np.argmax(ev))
        tot += TIER_WEIGHT[t] * ev[i]
        parts.append(f"{t[:4]}={ev[i]:.4f}@s{G[i]:.2f}/b{bu[i]:.3f}")
    return tot, parts


print(f"=== EV(bust<=1%) frontier, 3 seeds x {R} bootstraps of 880 ===")
base, p = weval()
print(f"  {'no veto, kappa2=1':45s} {base:.4f}  " + " ".join(p))
for k2 in (1.24, 1.5, 2.0):
    t, p = weval(kappa2=k2)
    print(f"  {'no veto, kappa2=%.2f' % k2:45s} {t:.4f}  " + " ".join(p) + f"   ({t-base:+.4f})")
print()
for thr in (32000, 20000, 10000, 5000):
    v = opg >= thr
    t, p = weval(veto=v)
    print(f"  {'ORACLE veto otok/gen>=%d (%d items)' % (thr, v.sum()):45s} {t:.4f}  "
          + " ".join(p) + f"   ({t-base:+.4f})")
print()
long5k = opg >= 5000
for frac in (0.06, 0.15, 0.30):
    vals = []
    for seed in range(3):
        g = np.random.default_rng(seed)
        v = np.zeros(n, bool); v[g.choice(n, int(frac * n), replace=False)] = True
        v &= ~long5k
        vals.append(weval(veto=v)[0])
    print(f"  {'RANDOM veto %d%% (true-long excluded)' % (100*frac):45s} "
          f"{np.mean(vals):.4f} +- {np.std(vals):.4f}   ({np.mean(vals)-base:+.4f})")
print()
for q in (0.94, 0.85, 0.70):
    v = P["cost_premium"][:, 2] >= np.quantile(P["cost_premium"][:, 2], q)
    t, p = weval(veto=v)
    print(f"  {'PREDICTED-cost veto top %d%% (%d items, %d/%d long)' % (100*(1-q), v.sum(), (v&long5k).sum(), long5k.sum()):45s} "
          f"{t:.4f}  " + " ".join(p) + f"   ({t-base:+.4f})")
print()
for thr, k2 in ((10000, 1.5), (5000, 1.5)):
    v = opg >= thr
    t, p = weval(veto=v, kappa2=k2)
    print(f"  {'ORACLE veto>=%d + kappa2=%.1f' % (thr, k2):45s} {t:.4f}  " + " ".join(p)
          + f"   ({t-base:+.4f})")
