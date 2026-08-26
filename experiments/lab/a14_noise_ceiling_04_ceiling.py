# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a14 step 4 -- the honest ceiling, the noise inflation, and the honest
exchange rate  corr(pred, latent p) -> final score.

All "honest" numbers allocate with some predictor and EVALUATE THE EXPECTED
score sum(p_sel)/N, where p is the latent success probability (posterior draws
from step 2).  The tier budget check always uses the TRUE realised cost.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_WEIGHT, TIER_MULT, MODEL_IDS   # noqa: E402

dv = load_split("dev")
N = len(dv)
IDX = np.arange(N)
CACHE = HERE / "_a14_cache"
D = np.load(CACHE / "pdraws_dev.npz")
PD = D["p"].astype(np.float64)            # (NDRAW, N, 3)
PHAT = D["phat"].astype(np.float64)
P43 = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE43 = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
TRUE_COST = dv.cost
LIGHT_SUM = dv.cost[:, 0].sum()


# ---------------------------------------------------------------- fast allocator
def allocate(pred_score, pred_cost, multiplier, safety, cost_norm=None):
    """exact copy of labdata.allocate (Lagrangian bisection), kept local so the
    inner loops stay cheap."""
    ls = pred_cost[:, 0].sum() if cost_norm is None else cost_norm
    cap = ls * max(1.0, multiplier * safety)
    cn = pred_cost / ls
    ar = np.arange(pred_score.shape[0])

    def choose(pen):
        return (pred_score - pen * cn).argmax(axis=1)

    sel = choose(0.0)
    total = pred_cost[ar, sel].sum()
    if total > cap:
        low, high = 0.0, 1.0
        sel = choose(high); total = pred_cost[ar, sel].sum()
        while total > cap and high < 2 ** 60:
            low, high = high, high * 2.0
            sel = choose(high); total = pred_cost[ar, sel].sum()
        for _ in range(40):
            mid = 0.5 * (low + high)
            cand = choose(mid); ct = pred_cost[ar, cand].sum()
            if ct <= cap:
                high, sel, total = mid, cand, ct
            else:
                low = mid
    if total > cap:
        sel = np.zeros(pred_score.shape[0], dtype=int)
    return sel


def run_final(pred_score_of_tier, pred_cost_of_tier, eval_score, safety):
    """returns (weighted final, per-tier dict)."""
    tot = 0.0
    per = {}
    for t in TIERS:
        sel = allocate(pred_score_of_tier(t), pred_cost_of_tier(t), TIER_MULT[t], safety[t])
        real = TRUE_COST[IDX, sel].sum() / LIGHT_SUM
        ok = real <= TIER_MULT[t] + 1e-15
        v = float(eval_score[IDX, sel].mean())
        per[t] = (v, real, ok)
        tot += TIER_WEIGHT[t] * (v if ok else 0.0)
    return tot, per


def fmt(per):
    return "  ".join(f"{t[:4]}={v:.4f}/r{r:.3f}{'' if ok else '!BUST'}"
                     for t, (v, r, ok) in per.items())


# ================================================================= PART A: ceiling
def part_a():
    print("=" * 100)
    print("PART A -- HONEST CEILING AND NOISE INFLATION  (dev, true costs, safety 1.0)")
    print("=" * 100)
    S1 = {t: 1.0 for t in TIERS}
    tc = lambda t: TRUE_COST

    v, per = run_final(lambda t: dv.score, tc, dv.score, S1)
    print(f"{'A1 realised-score oracle, self-evaluated':58s} {v:.4f}  {fmt(per)}")
    v_infl = v

    v, per = run_final(lambda t: dv.score, tc, PHAT, S1)
    print(f"{'A2 realised-score oracle, HONEST eval E[p|k]':58s} {v:.4f}  {fmt(per)}")
    v_honest_realised = v

    v, per = run_final(lambda t: PHAT, tc, PHAT, S1)
    print(f"{'A3 posterior-mean router (best label-only router), honest':58s} {v:.4f}  {fmt(per)}")
    v_label = v

    vs = []
    pers = []
    for d in range(PD.shape[0]):
        p = PD[d]
        vv, pp = run_final(lambda t, p=p: p, tc, p, S1)
        vs.append(vv); pers.append([pp[t][0] for t in TIERS])
    vs = np.array(vs); pers = np.array(pers)
    print(f"{'A4 know-p-exactly CEILING (alloc & eval on p)':58s} {vs.mean():.4f}"
          f"  +-{vs.std()/np.sqrt(len(vs)):.4f} (post. sd {vs.std():.4f})"
          f"  fast={pers[:,0].mean():.4f} bal={pers[:,1].mean():.4f} prem={pers[:,2].mean():.4f}")
    v_ceiling = vs.mean()

    print(f"\n  NOISE INFLATION of the published oracle = {v_infl:.4f} - {v_ceiling:.4f}"
          f" = {v_infl - v_ceiling:+.4f}")
    print(f"  inflation of the realised-score ROUTER   = {v_infl:.4f} - {v_honest_realised:.4f}"
          f" = {v_infl - v_honest_realised:+.4f}")
    print(f"  headroom above deployed 0.7017 is {v_ceiling-0.7017:.4f}, "
          f"not {v_infl-0.7017:.4f}  ({100*(v_ceiling-0.7017)/(v_infl-0.7017):.0f}% of the advertised gap)")

    # reference points on the same (honest) scale
    for name, sel in (("all-light", 0), ("all-mid", 1), ("all-k1", 2)):
        print(f"  ref {name:10s} honest mean p = {PHAT[:, sel].mean():.4f} "
              f"(realised {dv.score[:, sel].mean():.4f})")

    # model check against the assumption-free split (step 3 gave 0.8142 / 0.7444)
    rng = np.random.default_rng(3)
    a_own, a_cross = [], []
    for d in range(min(20, PD.shape[0])):
        p = PD[d]
        n = dv.ngen.astype(int)
        k1 = rng.binomial(1, p)                     # one fresh generation
        v1, _ = run_final(lambda t: k1.astype(float), tc, k1.astype(float), S1)
        v2, _ = run_final(lambda t: k1.astype(float), tc, p, S1)
        a_own.append(v1); a_cross.append(v2)
    print(f"\n  MODEL CHECK: simulated 1-generation oracle self-eval={np.mean(a_own):.4f} "
          f"(data 0.8142), honest-eval={np.mean(a_cross):.4f} (data 0.7444)")
    return dict(infl=v_infl, honest_realised=v_honest_realised, label=v_label, ceiling=v_ceiling)


# ================================================================= PART B: exchange
def make_pred(p, q, rng, rho_err=0.0, mu=None, sd=None):
    """calibrated predictor of p with corr(pred, p) = q.
    pred = mu + lam (p-mu) + sqrt(lam(1-lam)) sd_p eps,  lam = q^2.
    rho_err = cross-model correlation of eps."""
    lam = q * q
    if mu is None:
        mu = p.mean(0)
    if sd is None:
        sd = p.std(0)
    e = rng.standard_normal(p.shape)
    if rho_err > 0:
        c = rng.standard_normal((p.shape[0], 1))
        e = np.sqrt(rho_err) * c + np.sqrt(1 - rho_err) * e
    return mu + lam * (p - mu) + np.sqrt(lam * (1 - lam)) * sd * e


def part_b(qgrid, rho_err, reps=8, seed=101):
    print("\n" + "=" * 100)
    print(f"PART B -- HONEST EXCHANGE RATE (true costs, safety 1.0, cross-model "
          f"error corr={rho_err})")
    print("=" * 100)
    rng = np.random.default_rng(seed)
    S1 = {t: 1.0 for t in TIERS}
    tc = lambda t: TRUE_COST
    print(f"{'corr(pred,p)':>13s} {'final':>8s} {'sd':>7s} {'fast':>8s} {'bal':>8s} {'prem':>8s}"
          f" {'implied corr(pred,realised s)':>30s}")
    out = {}
    for q in qgrid:
        vals = []
        tiers = []
        cs = []
        for r in range(reps):
            p = PD[rng.integers(PD.shape[0])]
            pr = make_pred(p, q, rng, rho_err)
            v, per = run_final(lambda t: pr, tc, p, S1)
            vals.append(v); tiers.append([per[t][0] for t in TIERS])
            cs.append([np.corrcoef(pr[:, j], p[:, j])[0, 1] for j in range(3)])
        vals = np.array(vals); tiers = np.array(tiers); cs = np.array(cs).mean(0)
        # corr with the realised score = corr(pred,p) * sd(p)/sd(s)
        ratio = np.array([PD.reshape(-1, 3)[:, j].std() / dv.score[:, j].std() for j in range(3)])
        out[q] = vals.mean()
        print(f"{q:13.3f} {vals.mean():8.4f} {vals.std():7.4f} {tiers[:,0].mean():8.4f} "
              f"{tiers[:,1].mean():8.4f} {tiers[:,2].mean():8.4f}"
              f"   {'/'.join(f'{c*rr:.2f}' for c, rr in zip(cs, ratio)):>30s}")
    return out


if __name__ == "__main__":
    res = part_a()
    QG = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 0.95, 1.00]
    part_b(QG, rho_err=0.0)
    part_b(QG, rho_err=0.6)
