# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a11 step 10 - (i) is the delta-method upper bound calibrated?  (ii) does an
adaptive bound beat a fixed scalar under distribution shift?  (iii) is the
predicted-cost veto vacuous?  (iv) how much score does the leftover slack buy?
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result
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
KAP = np.array([1.0, 0.93, 1.24])
HARD = {"aime", "hrmcr", "dmmath", "code", "gsm8k_or_other"}

# residual moments (whole-dev; used only to BUILD the bound, its calibration is
# then checked against the realised spread)
SIG, RHO = {}, {}
for t in TIERS:
    E = np.log(dv.cost) - np.log(P[f"cost_{t}"])
    SIG[t] = E.std(0)
    RHO[t] = np.array([np.corrcoef(E[:, j], E[:, 0])[0, 1] for j in range(3)])

MEAN, VAR, COV = {}, {}, {}
for t in TIERS:
    M = P[f"cost_{t}"] * KAP[None, :]
    MEAN[t] = M
    VAR[t] = M ** 2 * (np.exp(SIG[t] ** 2) - 1.0)
    COV[t] = M * M[:, [0]] * (np.exp(RHO[t] * SIG[t] * SIG[t][0]) - 1.0)

PATH = {}
for t in TIERS:
    pa = a11path.path_arrays(P[f"score_{t}"], MEAN[t], dv.score, dv.cost)
    r = np.arange(n); frm, to, val = pa["frm"], pa["to"], pa["valid"]
    pa["dvar"] = np.where(val, VAR[t][r[:, None], to] - VAR[t][r[:, None], frm], 0.0)
    pa["dcov"] = np.where(val, COV[t][r[:, None], to] - COV[t][r[:, None], frm], 0.0)
    PATH[t] = pa

KEYS = ("dc_p", "dc_t", "ds_t", "dvar", "dcov")


def prep(pa, S):
    v = pa["valid"][S].ravel()
    o = np.argsort(-pa["eff"][S].ravel()[v], kind="stable")
    return {k: np.concatenate([[0.0], np.cumsum(pa[k][S].ravel()[v][o])]) for k in KEYS}


def make_samples(seed, size=880, weights=None, R=200):
    g = np.random.default_rng(seed)
    p = None if weights is None else weights / weights.sum()
    return [g.choice(n, size=size, replace=True, p=p) for _ in range(R)]


# ------------------------------------------------- (i) bound calibration check
print("=== (i) delta-method sd(log R) vs the realised bootstrap spread ===")
print(f"{'tier':9s} {'pred sd(logR)':>13s} {'boot sd(logR)':>13s} {'ratio':>6s} "
      f"{'pred p99/med':>12s} {'boot p99/med':>12s}")
for t in TIERS:
    pa = PATH[t]
    S0 = np.arange(n)
    cum = prep(pa, S0)
    bp = MEAN[t][:, 0].sum()
    cap = bp * (TIER_MULT[t] * DEP[t] - 1.0)
    k = int(np.searchsorted(cum["dc_p"], cap, side="right")) - 1
    N = bp + cum["dc_p"][k]; D = bp
    VN = VAR[t][:, 0].sum() + cum["dvar"][k]
    CV = VAR[t][:, 0].sum() + cum["dcov"][k]
    VD = VAR[t][:, 0].sum()
    sd_pred = np.sqrt(max(VN / N ** 2 + VD / D ** 2 - 2 * CV / (N * D), 0.0))
    # realised spread of log R for the SAME fixed selection under item resampling
    sel = np.zeros(n, dtype=int)
    v = pa["valid"].ravel(); o = np.argsort(-pa["eff"].ravel()[v], kind="stable")
    item = np.repeat(np.arange(n)[:, None], 2, 1).ravel()[v][o]
    tof = pa["to"].ravel()[v][o]
    sel[item[:k]] = tof[:k]
    c = dv.cost[np.arange(n), sel]; l = dv.cost[:, 0]
    g = np.random.default_rng(5)
    Sb = g.integers(0, n, size=(4000, n))
    R_ = c[Sb].sum(1) / l[Sb].sum(1)
    sd_boot = np.log(R_).std()
    print(f"{t:9s} {sd_pred:13.4f} {sd_boot:13.4f} {sd_pred/sd_boot:6.2f} "
          f"{np.exp(2.326*sd_pred):12.3f} {np.quantile(R_,0.99)/np.median(R_):12.3f}")
print("  (a delta-method sd built from per-item log-normal moments vs the empirical")
print("   880-resample spread of the realised ratio at the deployed stopping point)")

# ------------------------------------------------- (ii) adaptive vs fixed under shift
print("\n=== (ii) fixed scalar vs adaptive variance bound, nominal-matched, under shift ===")
w_hard = np.array([2.0 if f in HARD else 1.0 for f in fam])
q80 = np.quantile(dv.cost[:, 2], 0.8)
w_long = np.where(dv.cost[:, 2] >= q80, 2.0, 1.0)
SCEN = {"nominal": {}, "harder": dict(weights=w_hard), "longer-think": dict(weights=w_long),
        "N=440": dict(size=440), "N=1760": dict(size=1760)}
SAMP = {nm: [make_samples(sd, **kw) for sd in (7, 17, 23)] for nm, kw in SCEN.items()}
SGRID = np.arange(0.60, 1.201, 0.01)
ZGRID = np.arange(-1.0, 6.01, 0.25)


def run(tier, samples, mode, grid):
    pa = PATH[tier]; mult = TIER_MULT[tier]
    ev = np.zeros(len(grid)); bu = np.zeros(len(grid))
    for S in samples:
        m = len(S)
        bp = MEAN[tier][S, 0].sum(); bt = dv.cost[S, 0].sum(); bs = dv.score[S, 0].sum()
        bv = VAR[tier][S, 0].sum()
        cum = prep(pa, S)
        if mode == "z":
            N = bp + cum["dc_p"]; VN = bv + cum["dvar"]; CV = bv + cum["dcov"]
            var = np.maximum(VN / N ** 2 + bv / bp ** 2 - 2 * CV / (N * bp), 0.0)
            sdl = np.sqrt(var); base_r = N / bp
        for gi, g_ in enumerate(grid):
            if mode == "s":
                cap = bp * (max(1.0, mult * g_) - 1.0)
                k = int(np.searchsorted(cum["dc_p"], cap, side="right")) - 1
            else:
                ok = base_r * np.exp(g_ * sdl) <= mult
                bad = np.flatnonzero(~ok)
                k = int(bad[0] - 1) if len(bad) else len(ok) - 1
                k = max(k, 0)
            ratio = (bt + cum["dc_t"][k]) / bt
            sc = (bs + cum["ds_t"][k]) / m
            if ratio > mult:
                bu[gi] += 1
            else:
                ev[gi] += sc
    return ev / len(samples), bu / len(samples)


for tier in TIERS:
    print(f"-- {tier}")
    rows = []
    for mode, grid, label in (("s", SGRID, "fixed s"), ("z", ZGRID, "adaptive z")):
        ev = np.zeros(len(grid)); bu = np.zeros(len(grid))
        for smp in SAMP["nominal"]:
            e, b = run(tier, smp, mode, grid)
            ev += e / 3; bu += b / 3
        ok = np.flatnonzero(bu <= 0.01)
        i = ok[int(np.argmax(ev[ok]))] if len(ok) else int(np.argmax(ev))
        out = [label, f"{grid[i]:+.2f}", f"{ev[i]:.4f}", f"{bu[i]:.3f}"]
        for nm in SCEN:
            if nm == "nominal":
                continue
            e2 = np.zeros(len(grid)); b2 = np.zeros(len(grid))
            for smp in SAMP[nm]:
                e, b = run(tier, smp, mode, grid)
                e2 += e / 3; b2 += b / 3
            out.append(f"{e2[i]:.4f}/{b2[i]:.2f}")
        rows.append(out)
    print("   " + " ".join(f"{h:>12s}" for h in
                           ["rule", "param", "EVnom", "bust", *[k for k in SCEN if k != "nominal"]]))
    for r in rows:
        print("   " + " ".join(f"{x:>12s}" for x in r))

# ------------------------------------------------- (iii) is the veto vacuous?
print("\n=== (iii) predicted-cost veto: do the selected k1 items ever sit in the")
print("    top decile of PREDICTED k1 cost? ===")
for t in TIERS:
    C = P[f"cost_{t}"]
    sel = tier_result(P[f"score_{t}"], C, dv, t, DEP[t])["sel"]
    for q in (0.9, 0.8, 0.5):
        thr = np.quantile(C[:, 2], q)
        hit = int(((sel == 2) & (C[:, 2] >= thr)).sum())
        print(f"  {t:9s} veto q={q:.2f}: {hit:3d} of the {int((sel==2).sum()):3d} k1 "
              f"selections would be blocked "
              f"({100*dv.cost[(sel==2)&(C[:,2]>=thr), 2].sum()/max(dv.cost[sel==2,2].sum(),1e-9):.1f}% "
              f"of the realised k1 cost)")

# ------------------------------------------------- (iv) leftover-slack fill-up
print("\n=== (iv) greedy fill-up of the leftover slack (deterministic dev) ===")
for t in TIERS:
    ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
    pa = a11path.path_arrays(ps, pc, dv.score, dv.cost)
    v = pa["valid"].ravel(); o = np.argsort(-pa["eff"].ravel()[v], kind="stable")
    dcp = pa["dc_p"].ravel()[v][o]; dct = pa["dc_t"].ravel()[v][o]
    dst = pa["ds_t"].ravel()[v][o]; dsp = pa["ds_p"].ravel()[v][o]
    item = np.repeat(np.arange(n)[:, None], 2, 1).ravel()[v][o]
    slot = np.tile(np.array([0, 1]), n)[v][o]
    bp = pc[:, 0].sum()
    cap = bp * (TIER_MULT[t] * DEP[t] - 1.0)
    cum = np.concatenate([[0.0], np.cumsum(dcp)])
    k = int(np.searchsorted(cum, cap, side="right")) - 1
    slack = cap - cum[k]
    taken = np.zeros(n, dtype=int)
    np.add.at(taken, item[:k], 1)
    add_s_pred = add_s_true = add_c_true = 0.0
    nadd = 0
    for j in range(k, len(dcp)):
        if dcp[j] <= slack and slot[j] == taken[item[j]]:
            slack -= dcp[j]; taken[item[j]] += 1
            add_s_pred += dsp[j]; add_s_true += dst[j]; add_c_true += dct[j]; nadd += 1
    print(f"  {t:9s} slack={cap-cum[k]:.5f} ({100*(cap-cum[k])/cap:.2f}% of cap) "
          f"-> fill-up takes {nadd} extra upgrades, "
          f"pred score +{add_s_pred/n:+.5f}, TRUE score {add_s_true/n:+.5f}, "
          f"true ratio {add_c_true/dv.cost[:,0].sum():+.5f}")
