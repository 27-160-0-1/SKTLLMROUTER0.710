# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a11 step 6 - stopping rules on the Lagrangian path: bootstrap EV / bust frontier.

Every rule below is the SAME allocator (verified in step 5 to reproduce the
deployed bisection exactly); they differ only in WHERE they stop on the
efficiency-ordered path.  That is the whole design space of a post-hoc repair.

  A  scalar          stop while  N_hat <= mult * s * D_hat            (deployed)
  B  calibrated      per-model k_j (5-fold cross-fitted on dev) multiplies the
                     predicted costs (changes the path AND the test), then A
  C  variance-aware  stop while (N/D) * exp(z*sd(log R)) <= mult, delta-method
                     sd from cross-fitted per-model log-residual moments
  D  k1-tail veto    forbid model 2 for items whose predicted k1 cost is above
                     quantile q, then rule A with a re-tuned s
  E  A + greedy fill-up of the leftover slack (integrality residue)

Scenarios: nominal 880-bootstrap (3 seeds) + the four E39 stress scenarios.
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

# ---------------------------------------------------------------- cross-fitted
rng0 = np.random.default_rng(2024)
fold = rng0.integers(0, 5, size=n)
KCAL, SIG, RHO = {}, {}, {}
for t in TIERS:
    C = P[f"cost_{t}"]
    E = np.log(dv.cost) - np.log(C)
    k = np.zeros((n, 3)); sg = np.zeros((n, 3)); rh = np.zeros((n, 3))
    for f in range(5):
        m = fold != f
        k[fold == f] = dv.cost[m].sum(0) / C[m].sum(0)
        sg[fold == f] = E[m].std(0)
        rh[fold == f] = [np.corrcoef(E[m][:, j], E[m][:, 0])[0, 1] for j in range(3)]
    KCAL[t], SIG[t], RHO[t] = k, sg, rh

# per-item calibrated mean / variance / covariance-with-light of the true cost
MEAN, VAR, COV = {}, {}, {}
for t in TIERS:
    M = P[f"cost_{t}"] * KCAL[t]
    s2 = SIG[t] ** 2
    MEAN[t] = M
    VAR[t] = M ** 2 * (np.exp(s2) - 1.0)
    COV[t] = M * M[:, [0]] * (np.exp(RHO[t] * SIG[t] * SIG[t][:, [0]]) - 1.0)

PATHS = {}
for t in TIERS:
    ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
    PATHS[(t, "raw")] = a11path.path_arrays(ps, pc, dv.score, dv.cost)
    PATHS[(t, "cal")] = a11path.path_arrays(ps, MEAN[t], dv.score, dv.cost)


def add_extra(pa, t):
    """attach per-segment deltas of var / cov / calibrated cost."""
    r = np.arange(n)
    frm, to, valid = pa["frm"], pa["to"], pa["valid"]
    pa["dvar"] = np.where(valid, VAR[t][r[:, None], to] - VAR[t][r[:, None], frm], 0.0)
    pa["dcov"] = np.where(valid, COV[t][r[:, None], to] - COV[t][r[:, None], frm], 0.0)
    pa["dc_cal"] = np.where(valid, MEAN[t][r[:, None], to] - MEAN[t][r[:, None], frm], 0.0)
    return pa


for t in TIERS:
    add_extra(PATHS[(t, "raw")], t)
    add_extra(PATHS[(t, "cal")], t)

CUMKEYS = ("dc_p", "dc_t", "ds_t", "dvar", "dcov", "dc_cal")


def prep(pa, S):
    v = pa["valid"][S].ravel()
    eff = pa["eff"][S].ravel()[v]
    o = np.argsort(-eff, kind="stable")
    out = {k: np.concatenate([[0.0], np.cumsum(pa[k][S].ravel()[v][o])]) for k in CUMKEYS}
    out["dc_p_sorted"] = pa["dc_p"][S].ravel()[v][o]
    out["item_sorted"] = np.repeat(np.arange(len(S))[:, None], 2, 1).ravel()[v][o]
    out["slot_sorted"] = np.tile(np.array([0, 1]), len(S))[v][o]
    return out


def make_samples(seed, size=880, weights=None, R=400):
    g = np.random.default_rng(seed)
    p = None if weights is None else weights / weights.sum()
    return [g.choice(n, size=size, replace=True, p=p) for _ in range(R)]


def eval_rules(tier, samples, rules, path_key="raw", pa_override=None):
    """rules: dict name -> stop_fn(ctx)->k.  One pass over the samples."""
    pa = pa_override if pa_override is not None else PATHS[(tier, path_key)]
    mult = MULT[tier]
    acc = {name: [[], 0, []] for name in rules}
    for S in samples:
        m = len(S)
        base_p = P[f"cost_{tier}"][S, 0].sum()
        base_cal = MEAN[tier][S, 0].sum()
        base_t = dv.cost[S, 0].sum()
        base_s = dv.score[S, 0].sum()
        base_var = VAR[tier][S, 0].sum()
        cum = prep(pa, S)
        ctx = dict(m=m, base_p=base_p, base_cal=base_cal, base_t=base_t, base_s=base_s,
                   base_var=base_var, cum=cum, mult=mult)
        for name, fn in rules.items():
            k = fn(ctx)
            ratio = (base_t + cum["dc_t"][k]) / base_t
            sc = (base_s + cum["ds_t"][k]) / m
            a = acc[name]
            a[2].append(sc)
            if ratio > mult:
                a[1] += 1; a[0].append(0.0)
            else:
                a[0].append(sc)
    return {nm: (float(np.mean(a[0])), a[1] / len(samples), float(np.mean(a[2])))
            for nm, a in acc.items()}


# ------------------------------------------------------------------ stop rules
def scalar(s, cal=False):
    key = "dc_cal" if cal else "dc_p"
    bkey = "base_cal" if cal else "base_p"
    def f(ctx):
        cap = ctx[bkey] * (max(1.0, ctx["mult"] * s) - 1.0)
        return int(np.searchsorted(ctx["cum"][key], cap, side="right")) - 1
    return f


def var_aware(z):
    def f(ctx):
        c = ctx["cum"]
        N = ctx["base_cal"] + c["dc_cal"]
        D = ctx["base_cal"]
        VN = ctx["base_var"] + c["dvar"]
        CV = ctx["base_var"] + c["dcov"]
        VD = ctx["base_var"]
        v = VN / N ** 2 + VD / D ** 2 - 2 * CV / (N * D)
        ub = (N / D) * np.exp(z * np.sqrt(np.maximum(v, 0.0)))
        ok = ub <= ctx["mult"]
        # last index of the leading run of True
        bad = np.flatnonzero(~ok)
        return int(bad[0] - 1) if len(bad) else len(ok) - 1
    return f


# --------------------------------------------------------------------- driver
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="AB")
    ap.add_argument("--R", type=int, default=400)
    a = ap.parse_args()
    t0 = time.time()
    seeds = (7, 17, 23)
    nominal = {s: make_samples(s, R=a.R) for s in seeds}

    if "AB" in a.part:
        print("=== nominal (880 x %d x 3 seeds): rule A vs B vs C ===" % a.R)
        for tier in TIERS:
            rules = {}
            for s in np.arange(0.74, 1.041, 0.01):
                rules[f"A{s:.2f}"] = scalar(float(s))
            out = {}
            for sd in seeds:
                r = eval_rules(tier, nominal[sd], rules, "raw")
                for k, v in r.items():
                    out.setdefault(k, []).append(v)
            rulesB = {f"B{s:.2f}": scalar(float(s), cal=True) for s in np.arange(0.74, 1.081, 0.01)}
            rulesC = {f"C{z:+.2f}": var_aware(float(z)) for z in np.arange(-1.0, 3.01, 0.25)}
            for sd in seeds:
                for rl, pk in ((rulesB, "cal"), (rulesC, "cal")):
                    r = eval_rules(tier, nominal[sd], rl, pk)
                    for k, v in r.items():
                        out.setdefault(k, []).append(v)
            print(f"-- {tier} (mult {MULT[tier]})")
            print(f"   {'rule':8s} {'EV':>7s} {'bust':>6s} {'cond':>7s}")
            for k in sorted(out, key=lambda x: (x[0], float(x[1:]))):
                v = np.mean(np.array(out[k]), axis=0)
                star = "  <-- deployed" if k == f"A{DEP[tier]:.2f}" else ""
                print(f"   {k:8s} {v[0]:7.4f} {v[1]:6.3f} {v[2]:7.4f}{star}")
            np.save(HERE / f"a11_AB_{tier}.npy",
                    np.array([(k, *np.mean(np.array(v), axis=0)) for k, v in out.items()], dtype=object),
                    allow_pickle=True)
        print(f"[{time.time()-t0:.0f}s]")
