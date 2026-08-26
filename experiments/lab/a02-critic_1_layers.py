# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a02-critic #1.  How much information does the 58-feature meta stack actually
carry beyond a 9-row family lookup table?  And how entangled are the cost
model's bias and the 'safety scalar'?

All numbers are computed on the honest held-out dev predictions
(reports/lab/dev_preds_e43.npz, produced from the Train-only artifact in
reports/holdout_local/learned-router.v1.json -- num_train_episodes=1760).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, TIER_MULT, tier_result
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
n = len(dv)
fam = np.array([classify_family(t) for t in dv.texts])
fams = sorted(set(fam))
F = np.stack([(fam == f).astype(float) for f in fams], 1)


def final(mk_s, mk_c, safety=SAFE):
    tot = 0.0
    parts = []
    for t in TIERS:
        r = tier_result(mk_s(t), mk_c(t), dv, t, safety[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else '!BUST'}")
    return tot, "  ".join(parts)


def show(name, mk_s, mk_c, safety=SAFE):
    tot, parts = final(mk_s, mk_c, safety)
    print(f"{name:58s} {tot:.4f}   {parts}")
    return tot


ps = lambda t: P[f"score_{t}"]
pc = lambda t: P[f"cost_{t}"]

print("=" * 100)
print("A.  How much of the deployed prediction is just the family lookup table?")
print("=" * 100)


def famproj(X):
    """replace every column by its within-family mean -> strips all within-family info"""
    Y = X.copy()
    for f in fams:
        m = fam == f
        Y[m] = X[m].mean(0)
    return Y


print("\nR^2 of the deployed predictions explained by the 9 family dummies alone:")
for t in TIERS:
    row = []
    for j, m in enumerate(MODEL_IDS):
        x = P[f"score_{t}"][:, j]
        xf = famproj(x[:, None])[:, 0]
        r2 = 1 - ((x - xf) ** 2).sum() / ((x - x.mean()) ** 2).sum()
        row.append(f"{m}={r2:.3f}")
    print(f"  score  {t:9s} " + "  ".join(row))
for t in ("premium",):
    row = []
    for j, m in enumerate(MODEL_IDS):
        x = np.log(P[f"cost_{t}"][:, j])
        xf = famproj(x[:, None])[:, 0]
        r2 = 1 - ((x - xf) ** 2).sum() / ((x - x.mean()) ** 2).sum()
        row.append(f"{m}={r2:.3f}")
    print(f"  logcost{t:9s} " + "  ".join(row))

# and the same for the TRUE labels: how much family-explainable signal exists at all?
print("\nR^2 of the TRUE labels explained by the 9 family dummies alone:")
row = []
for j, m in enumerate(MODEL_IDS):
    x = dv.score[:, j]
    xf = famproj(x[:, None])[:, 0]
    r2 = 1 - ((x - xf) ** 2).sum() / ((x - x.mean()) ** 2).sum()
    row.append(f"{m}={r2:.3f}")
print("  score            " + "  ".join(row))
row = []
for j, m in enumerate(MODEL_IDS):
    x = np.log(dv.cost[:, j])
    xf = famproj(x[:, None])[:, 0]
    r2 = 1 - ((x - xf) ** 2).sum() / ((x - x.mean()) ** 2).sum()
    row.append(f"{m}={r2:.3f}")
print("  logcost          " + "  ".join(row))

print("\nwithin-family correlation of prediction residual with truth residual")
print("(this is the ONLY thing the 16,414-dim ridge + 22 GBM heads add over a 9-row table):")
for j, m in enumerate(MODEL_IDS):
    p_ = P["premium"[:0] + "fast"][:, j] if False else P["score_fast"][:, j]
    pr = p_ - famproj(p_[:, None])[:, 0]
    tr_ = dv.score[:, j]
    tres = tr_ - famproj(tr_[:, None])[:, 0]
    c_raw = np.corrcoef(p_, tr_)[0, 1]
    c_res = np.corrcoef(pr, tres)[0, 1]
    print(f"  {m:11s} raw corr={c_raw:.3f}   within-family corr={c_res:.3f}")

print("\n--- decision impact ---")
print("baseline / family-projected score / family-projected cost / both")
base = show("deployed (E43 held-out preds)", ps, pc)
sf = show("score -> within-family mean (no item-level score info)", lambda t: famproj(P[f"score_{t}"]), pc)
cf = show("cost  -> within-family geo-mean (no item-level cost info)",
          ps, lambda t: np.exp(famproj(np.log(P[f"cost_{t}"]))))
bf = show("both -> family means (a 9x6 lookup table router)",
          lambda t: famproj(P[f"score_{t}"]),
          lambda t: np.exp(famproj(np.log(P[f"cost_{t}"]))))

print("\nselection agreement with the deployed router (fraction of items):")
for name, mk_s, mk_c in (("family-mean score", lambda t: famproj(P[f"score_{t}"]), pc),
                         ("family-mean cost", ps, lambda t: np.exp(famproj(np.log(P[f"cost_{t}"])))),
                         ("family table only", lambda t: famproj(P[f"score_{t}"]),
                          lambda t: np.exp(famproj(np.log(P[f"cost_{t}"]))))):
    ag = []
    for t in TIERS:
        a = tier_result(ps(t), pc(t), dv, t, SAFE[t])["sel"]
        b = tier_result(mk_s(t), mk_c(t), dv, t, SAFE[t])["sel"]
        ag.append(f"{t[:4]}={np.mean(a == b):.3f}")
    print(f"  {name:22s} " + "  ".join(ag))

print()
print("=" * 100)
print("B.  The 'safety scalar' is not 0.98/0.89/0.88 -- the cost model's own bias")
print("    supplies most of the margin, and the bias is NOT model-neutral.")
print("=" * 100)
for t in TIERS:
    C = P[f"cost_{t}"]
    predL = C[:, 0].sum()
    trueL = dv.cost[:, 0].sum()
    eff = SAFE[t] * predL / trueL
    print(f"  {t:9s} nominal safety={SAFE[t]:.3f}  pred_light_sum/true_light_sum={predL/trueL:.4f}"
          f"  -> EFFECTIVE safety vs the real budget = {eff:.4f}")
print("\n  per-model sum bias (true/pred), i.e. the multiplicative correction each column needs:")
for t in TIERS:
    k = dv.cost.sum(0) / P[f"cost_{t}"].sum(0)
    print(f"  {t:9s} light={k[0]:.4f} mid={k[1]:.4f} k1={k[2]:.4f}"
          f"   RELATIVE mispricing k1/light={k[2]/k[0]:.4f}  mid/light={k[1]/k[0]:.4f}")
print("\n  => a global safety scalar can absorb a COMMON factor.  It cannot absorb the")
print("     fact that k1 upgrades are priced ~21% cheaper than light relative to truth.")

print("\n--- where does the cost-sum error come from?  (premium tier, k1 column) ---")
C = P["cost_premium"][:, 2]
T = dv.cost[:, 2]
err = T - C
o = np.argsort(-np.abs(err))
tot_err = err.sum()
for q in (0.01, 0.02, 0.05, 0.10, 0.25):
    kk = int(np.ceil(q * n))
    print(f"  top {q*100:5.1f}% items by |error| carry {err[o[:kk]].sum()/tot_err*100:6.1f}% of the net sum error"
          f"   (they are {T[o[:kk]].sum()/T.sum()*100:5.1f}% of true k1 cost)")
print(f"  log-space RMSE is dominated by cheap items: corr(|log err|, log true cost) ="
      f" {np.corrcoef(np.abs(np.log(C) - np.log(T)), np.log(T))[0,1]:+.3f}")
print(f"  cost-weighted mean relative error  sum(T-C)/sum(T) = {tot_err/T.sum():+.4f}")
print(f"  unweighted mean relative error     mean((T-C)/T)   = {np.mean((T-C)/T):+.4f}")
