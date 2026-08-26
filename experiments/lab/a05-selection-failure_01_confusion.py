# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a05 step 1: deployed-vs-oracle confusion matrix per tier + loss attribution
to each cell, plus an exact 3-factor Shapley decomposition of the per-item loss
into (i) score-prediction error, (ii) cost-prediction error, (iii) budget
pressure (shadow price), and (iv) label noise measured with an empirical-Bayes
latent-p estimate.

Everything printed here is recomputed from
  data/{train,dev}  +  reports/lab/dev_preds_e43.npz
"""
from __future__ import annotations
import sys, itertools, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result, MODEL_IDS
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.similarity import classify_family

np.set_printoptions(suppress=True, linewidth=160)

dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}   # reproduces BRIEF 0.7017
N = len(dv)
IDX = np.arange(N)
FAM = np.array([classify_family(t) for t in dv.texts])
FAMS = sorted(set(FAM))
SHORT = ["L", "M", "K"]


# ---------------------------------------------------------------- allocator with exposed price
def allocate_pen(pred_score, pred_cost, multiplier, safety):
    """Same bisection as labdata.allocate, but returns (sel, price) where
    price = pen / light_total is the absolute shadow price per unit cost."""
    L = pred_cost[:, 0].sum()
    cap = L * max(1.0, multiplier * safety)

    def choose(pen):
        return (pred_score - pen * pred_cost / L).argmax(axis=1)

    sel = choose(0.0)
    pen = 0.0
    total = pred_cost[IDX, sel].sum()
    if total > cap:
        low, high = 0.0, 1.0
        sel = choose(high); pen = high
        total = pred_cost[IDX, sel].sum()
        while total > cap and high < 2 ** 60:
            low, high = high, high * 2.0
            sel = choose(high); pen = high
            total = pred_cost[IDX, sel].sum()
        for _ in range(40):
            mid = (low + high) / 2.0
            cand = choose(mid)
            ct = pred_cost[IDX, cand].sum()
            if ct <= cap:
                high, sel, total, pen = mid, cand, ct, mid
            else:
                low = mid
    return sel, pen / L


# ---------------------------------------------------------------- empirical-Bayes latent p
def eb_posterior_mean(sp, fam, prior_scale=1.0):
    """Beta(a,b) per (family, model) by moment matching on k/n; returns E[p|k,n]."""
    n = sp.ngen.astype(int)
    k = np.rint(sp.score * n).astype(int)
    A = np.ones_like(sp.score); B = np.ones_like(sp.score)
    for f_ in set(fam):
        msk = fam == f_
        for j in range(3):
            x = sp.score[msk, j]
            mu = float(x.mean()); var = float(x.var()); nn = float(n[msk, j].mean())
            if not (0 < mu < 1):
                A[msk, j] = B[msk, j] = 1.0
                continue
            vp = (var - (mu - mu * mu) / nn) / (1 - 1.0 / nn) if nn > 1 else var
            vp = float(np.clip(vp, 1e-4, mu * (1 - mu) - 1e-4))
            c = (mu * (1 - mu) / vp - 1) * prior_scale
            A[msk, j] = max(c * mu, 0.05); B[msk, j] = max(c * (1 - mu), 0.05)
    return (A + k) / (A + B + n)


PHAT = eb_posterior_mean(dv, FAM)
print("=== empirical-Bayes latent-p sanity ===")
print("  mean realised score per model :", np.round(dv.score.mean(0), 4))
print("  mean posterior p    per model :", np.round(PHAT.mean(0), 4))
print("  shrinkage sd(s)->sd(p)        :", np.round(dv.score.std(0), 4), "->", np.round(PHAT.std(0), 4))
print("  corr(s, p) per model          :", np.round([np.corrcoef(dv.score[:, j], PHAT[:, j])[0, 1] for j in range(3)], 4))

results = {}
print("\n" + "=" * 100)
for tier in TIERS:
    ps, pc = P[f"score_{tier}"], P[f"cost_{tier}"]
    sel_d, price_d = allocate_pen(ps, pc, TIER_MULT[tier], SAFE[tier])
    sel_o, price_o = allocate_pen(dv.score, dv.cost, TIER_MULT[tier], 1.0)
    # sanity: identical to labdata
    assert (sel_d == tier_result(ps, pc, dv, tier, SAFE[tier])["sel"]).all()
    assert (sel_o == tier_result(dv.score, dv.cost, dv, tier, 1.0)["sel"]).all()

    s_d = dv.score[IDX, sel_d]; s_o = dv.score[IDX, sel_o]
    c_d = dv.cost[IDX, sel_d];  c_o = dv.cost[IDX, sel_o]
    L = dv.cost[:, 0].sum()
    print(f"### tier={tier}  mult={TIER_MULT[tier]}  safety={SAFE[tier]}")
    print(f"  deployed: score={s_d.mean():.4f} ratio={c_d.sum()/L:.4f} price={price_d:.6g} "
          f"sel={np.bincount(sel_d, minlength=3)}")
    print(f"  oracle  : score={s_o.mean():.4f} ratio={c_o.sum()/L:.4f} price={price_o:.6g} "
          f"sel={np.bincount(sel_o, minlength=3)}")
    gap = s_o.mean() - s_d.mean()
    print(f"  tier score gap = {gap:.4f}   weighted contribution to final gap = {TIER_WEIGHT[tier]*gap:.4f}")

    # ---------------- confusion matrix chosen(row) x oracle(col)
    cnt = np.zeros((3, 3), int)
    loss = np.zeros((3, 3))       # realised-score loss (sum over items)/N
    eloss = np.zeros((3, 3))      # expected (EB) loss
    dcost = np.zeros((3, 3))      # cost difference oracle - deployed, as ratio-of-light units
    for a in range(3):
        for b in range(3):
            m = (sel_d == a) & (sel_o == b)
            cnt[a, b] = m.sum()
            loss[a, b] = (dv.score[m, b] - dv.score[m, a]).sum() / N
            eloss[a, b] = (PHAT[m, b] - PHAT[m, a]).sum() / N
            dcost[a, b] = (dv.cost[m, b] - dv.cost[m, a]).sum() / L
    print("\n  confusion count  (row=deployed, col=oracle)      |  realised score loss/N  |  EB expected loss/N  |  d(cost)/L")
    print(f"  {'':6s}" + "".join(f"{SHORT[b]:>6s}" for b in range(3)) + "   |" +
          "".join(f"{SHORT[b]:>9s}" for b in range(3)) + "   |" +
          "".join(f"{SHORT[b]:>9s}" for b in range(3)) + "   |" +
          "".join(f"{SHORT[b]:>9s}" for b in range(3)))
    for a in range(3):
        print(f"  {SHORT[a]:4s}  " + "".join(f"{cnt[a,b]:6d}" for b in range(3)) + "   |" +
              "".join(f"{loss[a,b]:+9.4f}" for b in range(3)) + "   |" +
              "".join(f"{eloss[a,b]:+9.4f}" for b in range(3)) + "   |" +
              "".join(f"{dcost[a,b]:+9.3f}" for b in range(3)))
    print(f"  totals: n_disagree={int(cnt.sum()-np.trace(cnt))}  realised loss={loss.sum():+.4f} "
          f"EB expected loss={eloss.sum():+.4f}  (EB / realised = {eloss.sum()/loss.sum():.3f})")

    # ---------------- exact 3-factor Shapley on the per-item decision
    # players: S (true score), C (true cost in the utility), P (oracle price)
    Sopt = {0: ps, 1: dv.score}
    Copt = {0: pc, 1: dv.cost}
    Popt = {0: price_d, 1: price_o}
    selcache = {}
    for bs, bc, bp in itertools.product((0, 1), repeat=3):
        u = Sopt[bs] - Popt[bp] * Copt[bc]
        selcache[(bs, bc, bp)] = u.argmax(axis=1)
    assert (selcache[(0, 0, 0)] == sel_d).all(), "baseline combo must equal deployed"
    assert (selcache[(1, 1, 1)] == sel_o).all(), "full combo must equal oracle"

    def val(key, arr):
        return arr[IDX, selcache[key]]

    shap_r = np.zeros((N, 3))   # realised-score Shapley
    shap_e = np.zeros((N, 3))   # EB expected-score Shapley
    perms = list(itertools.permutations(range(3)))
    for perm in perms:
        state = [0, 0, 0]
        prev_r = val(tuple(state), dv.score)
        prev_e = val(tuple(state), PHAT)
        for pl in perm:
            state[pl] = 1
            cur_r = val(tuple(state), dv.score)
            cur_e = val(tuple(state), PHAT)
            shap_r[:, pl] += (cur_r - prev_r) / len(perms)
            shap_e[:, pl] += (cur_e - prev_e) / len(perms)
            prev_r, prev_e = cur_r, cur_e
    assert np.allclose(shap_r.sum(1), s_o - s_d)

    names = ["score-pred error", "cost-pred error", "budget pressure"]
    print("\n  Shapley attribution of the deployed->oracle score gap (per-item mean):")
    for pl in range(3):
        print(f"    {names[pl]:20s} realised {shap_r[:,pl].mean():+.4f}  "
              f"({100*shap_r[:,pl].mean()/gap:5.1f}%)   EB-expected {shap_e[:,pl].mean():+.4f}")
    print(f"    {'TOTAL':20s} realised {shap_r.sum(1).mean():+.4f}            "
          f"      EB-expected {shap_e.sum(1).mean():+.4f}")

    # ---------------- label-noise split of the realised gap
    dis = sel_d != sel_o
    d_real = dv.score[IDX, sel_o] - dv.score[IDX, sel_d]
    d_exp = PHAT[IDX, sel_o] - PHAT[IDX, sel_d]
    good = dis & (d_exp > 1e-9)      # oracle genuinely better in expectation
    bad = dis & (d_exp < -1e-9)      # deployed was the better bet -> oracle choice was noise
    tie = dis & (np.abs(d_exp) <= 1e-9)
    print(f"\n  label-noise split of {int(dis.sum())} disagreements:")
    print(f"    oracle better in E[p]  : n={int(good.sum()):4d}  realised gain {d_real[good].sum()/N:+.4f}  E[p] gain {d_exp[good].sum()/N:+.4f}")
    print(f"    deployed better in E[p]: n={int(bad.sum()):4d}  realised gain {d_real[bad].sum()/N:+.4f}  E[p] gain {d_exp[bad].sum()/N:+.4f}")
    print(f"    tie in E[p]            : n={int(tie.sum()):4d}  realised gain {d_real[tie].sum()/N:+.4f}")
    rec = d_exp[dis].sum() / N
    print(f"    => RECOVERABLE (E[p]) part of the {gap:.4f} tier gap = {rec:+.4f} "
          f"({100*rec/gap:.1f}%); noise-only part = {gap-rec:+.4f}")

    results[tier] = dict(sel_d=sel_d, sel_o=sel_o, price_d=price_d, price_o=price_o,
                         shap_r=shap_r, shap_e=shap_e, gap=gap, rec=rec,
                         cnt=cnt, loss=loss, eloss=eloss)
    print("=" * 100)

# ---------------- weighted totals
print("\n=== WEIGHTED FINAL-SCORE ACCOUNTING ===")
tot_gap = sum(TIER_WEIGHT[t] * results[t]["gap"] for t in TIERS)
tot_rec = sum(TIER_WEIGHT[t] * results[t]["rec"] for t in TIERS)
print(f"  deployed final = {sum(TIER_WEIGHT[t]*dv.score[IDX, results[t]['sel_d']].mean() for t in TIERS):.4f}")
print(f"  oracle   final = {sum(TIER_WEIGHT[t]*dv.score[IDX, results[t]['sel_o']].mean() for t in TIERS):.4f}")
print(f"  gap = {tot_gap:.4f}   recoverable (E[p]) = {tot_rec:.4f}   noise-only = {tot_gap-tot_rec:.4f}")
names = ["score-pred error", "cost-pred error", "budget pressure"]
for pl in range(3):
    r = sum(TIER_WEIGHT[t] * results[t]["shap_r"][:, pl].mean() for t in TIERS)
    e = sum(TIER_WEIGHT[t] * results[t]["shap_e"][:, pl].mean() for t in TIERS)
    print(f"  {names[pl]:20s} weighted realised {r:+.4f}   weighted EB-expected {e:+.4f}")

np.savez_compressed(ROOT / "experiments/lab/a05-selection-failure_cache.npz",
                    phat=PHAT, fam=FAM,
                    **{f"sel_d_{t}": results[t]["sel_d"] for t in TIERS},
                    **{f"sel_o_{t}": results[t]["sel_o"] for t in TIERS},
                    **{f"price_d_{t}": results[t]["price_d"] for t in TIERS},
                    **{f"price_o_{t}": results[t]["price_o"] for t in TIERS},
                    **{f"shap_r_{t}": results[t]["shap_r"] for t in TIERS},
                    **{f"shap_e_{t}": results[t]["shap_e"] for t in TIERS})
print("\nwrote experiments/lab/a05-selection-failure_cache.npz")
