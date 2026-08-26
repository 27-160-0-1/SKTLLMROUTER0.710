# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a11 step 8 - rule B as a deployable constant: relative price correction kappa.

Rule B multiplies the predicted costs by (1, kappa1, kappa2) before the
allocation.  Only the RATIOS matter (a common factor is absorbed by the safety
scalar).  Questions answered here:
  1. bootstrap EV frontier (best safety per kappa) as kappa2 is swept -> is the
     optimum flat enough to survive a +-15% error in the estimate of kappa2?
  2. the same under the four E39 stress scenarios
  3. rule D: veto model 2 for items whose predicted k1 cost is above quantile q
  4. rule C (variance-aware stopping) under stress
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router import similarity
import importlib.util
_spec = importlib.util.spec_from_file_location("a11path", HERE / "a11-postproc-router_5_path.py")
a11path = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(a11path)

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv)
fam = np.array([similarity.classify_family(x) for x in dv.texts])
DEP = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
MULT = TIER_MULT
HARD = {"aime", "hrmcr", "dmmath", "code", "gsm8k_or_other"}


def make_path(tier, kappa, veto_q=None):
    ps = P[f"score_{tier}"]
    pc = P[f"cost_{tier}"] * np.asarray(kappa)[None, :]
    if veto_q is not None and veto_q < 1.0:
        thr = np.quantile(P[f"cost_{tier}"][:, 2], veto_q)
        bad = P[f"cost_{tier}"][:, 2] >= thr
        pc = pc.copy()
        pc[bad, 2] = np.inf                      # model 2 unreachable for these
    pa = a11path.path_arrays(ps, pc, dv.score, dv.cost)
    pa["base_p"] = pc[:, 0]
    return pa


def prep(pa, S):
    v = pa["valid"][S].ravel()
    eff = pa["eff"][S].ravel()[v]
    o = np.argsort(-eff, kind="stable")
    return {k: np.concatenate([[0.0], np.cumsum(pa[k][S].ravel()[v][o])])
            for k in ("dc_p", "dc_t", "ds_t")}


def frontier(tier, pa, samples, grid):
    """EV / bust / cond for each safety value on `grid`."""
    mult = MULT[tier]
    ev = np.zeros(len(grid)); bu = np.zeros(len(grid)); co = np.zeros(len(grid))
    for S in samples:
        m = len(S)
        bp = pa["base_p"][S].sum(); bt = dv.cost[S, 0].sum(); bs = dv.score[S, 0].sum()
        cum = prep(pa, S)
        for gi, s in enumerate(grid):
            cap = bp * (max(1.0, mult * s) - 1.0)
            k = int(np.searchsorted(cum["dc_p"], cap, side="right")) - 1
            ratio = (bt + cum["dc_t"][k]) / bt
            sc = (bs + cum["ds_t"][k]) / m
            co[gi] += sc
            if ratio > mult:
                bu[gi] += 1
            else:
                ev[gi] += sc
    R = len(samples)
    return ev / R, bu / R, co / R


def make_samples(seed, size=880, weights=None, R=400):
    g = np.random.default_rng(seed)
    p = None if weights is None else weights / weights.sum()
    return [g.choice(n, size=size, replace=True, p=p) for _ in range(R)]


SCEN = {}
SCEN["nominal"] = dict()
w_hard = np.array([2.0 if f in HARD else 1.0 for f in fam])
q80 = np.quantile(dv.cost[:, 2], 0.8)
w_long = np.where(dv.cost[:, 2] >= q80, 2.0, 1.0)
SCEN["harder"] = dict(weights=w_hard)
SCEN["longer-think"] = dict(weights=w_long)
SCEN["N=440"] = dict(size=440)
SCEN["N=1760"] = dict(size=1760)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=300)
    ap.add_argument("--part", default="kappa")
    a = ap.parse_args()
    t0 = time.time()
    SAMP = {name: [make_samples(sd, R=a.R, **kw) for sd in (7, 17, 23)]
            for name, kw in SCEN.items()}
    GRID = np.arange(0.60, 1.101, 0.01)

    if "kappa" in a.part:
        print("=== 1. best-safety EV vs assumed kappa2 (nominal, 3 seeds x %d) ===" % a.R)
        KAPPAS = [(1.0, 1.0), (1.0, 1.1), (1.0, 1.24), (1.0, 1.4), (1.0, 1.6),
                  (0.93, 1.0), (0.93, 1.1), (0.93, 1.24), (0.93, 1.4), (0.93, 1.6),
                  (0.85, 1.24), (1.05, 1.24)]
        best = {}
        for tier in TIERS:
            print(f"-- {tier}")
            print(f"   {'k1':>5s} {'k2':>5s} {'bestEV':>7s} {'s*':>5s} {'bust':>6s} "
                  f"{'EV@bust<=1%':>11s} {'s':>5s}")
            for k1, k2 in KAPPAS:
                pa = make_path(tier, (1.0, k1, k2))
                ev = np.zeros(len(GRID)); bu = np.zeros(len(GRID)); co = np.zeros(len(GRID))
                for smp in SAMP["nominal"]:
                    e, b, c = frontier(tier, pa, smp, GRID)
                    ev += e / 3; bu += b / 3; co += c / 3
                i = int(np.argmax(ev))
                ok = np.flatnonzero(bu <= 0.01)
                j = ok[int(np.argmax(ev[ok]))] if len(ok) else i
                best[(tier, k1, k2)] = (ev, bu, co)
                print(f"   {k1:5.2f} {k2:5.2f} {ev[i]:7.4f} {GRID[i]:5.2f} {bu[i]:6.3f} "
                      f"{ev[j]:11.4f} {GRID[j]:5.2f}")
        np.save(HERE / "a11_kappa.npy", np.array(
            [(t, k1, k2, *v) for (t, k1, k2), v in best.items()], dtype=object), allow_pickle=True)
        print(f"[{time.time()-t0:.0f}s]")

    if "stress" in a.part:
        print("\n=== 2. stress scenarios: A (kappa=1) vs B (kappa=(1,.93,1.24)) vs D (veto) ===")
        CONFIGS = [("A", (1.0, 1.0, 1.0), None),
                   ("B", (1.0, 0.93, 1.24), None),
                   ("D95", (1.0, 1.0, 1.0), 0.95),
                   ("D90", (1.0, 1.0, 1.0), 0.90),
                   ("BD90", (1.0, 0.93, 1.24), 0.90)]
        for tier in TIERS:
            print(f"-- {tier}")
            hdr = f"   {'cfg':5s} {'s*nom':>6s} {'EVnom':>7s}"
            for name in SCEN:
                if name != "nominal":
                    hdr += f" {name[:9]:>10s}"
            print(hdr + "   (EV at the nominal-optimal s, per scenario)")
            for cname, kap, vq in CONFIGS:
                pa = make_path(tier, kap, vq)
                ev = np.zeros(len(GRID)); bu = np.zeros(len(GRID))
                for smp in SAMP["nominal"]:
                    e, b, c = frontier(tier, pa, smp, GRID)
                    ev += e / 3; bu += b / 3
                ok = np.flatnonzero(bu <= 0.01)
                i = ok[int(np.argmax(ev[ok]))] if len(ok) else int(np.argmax(ev))
                row = f"   {cname:5s} {GRID[i]:6.2f} {ev[i]:7.4f}"
                for name in SCEN:
                    if name == "nominal":
                        continue
                    e2 = np.zeros(len(GRID)); b2 = np.zeros(len(GRID))
                    for smp in SAMP[name]:
                        e, b, c = frontier(tier, pa, smp, GRID)
                        e2 += e / 3; b2 += b / 3
                    row += f" {e2[i]:6.4f}/{b2[i]:.2f}"
                print(row)
        print(f"[{time.time()-t0:.0f}s]")
