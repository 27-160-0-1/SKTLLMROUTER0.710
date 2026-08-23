# SPDX-License-Identifier: Apache-2.0
"""a12 P6 -- is the deployed gain prediction mis-scaled PER FAMILY?

P5.5 showed the deployed premium tier sends 43 belebele items to k1 where the
oracle sends 6, and 0 ruletaker/longdoc items where the oracle sends 9/8.
This script asks whether the per-family *gain* (not score) is biased, and tests
a train-only family gain prior on top of the deployed predictions.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from labdata import load_split, TIERS, TIER_WEIGHT, tier_result, Split  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

ROOT = HERE.parents[1]
dv, tr = load_split("dev"), load_split("train")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
fam = np.array([classify_family(t) for t in dv.texts])
famtr = np.array([classify_family(t) for t in tr.texts])
fams = sorted(set(fam))
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
rng = np.random.default_rng(7)
n = len(dv)


def hdr(t):
    print("\n" + "=" * 108)
    print(t)
    print("=" * 108)


def run(S, C, safety=SAFE, tag="", verbose=True):
    tot, out = 0.0, []
    for t in TIERS:
        r = tier_result(S(t), C(t), dv, t, safety[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        out.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else ' BUST'}")
    if verbose:
        print(f"  {tag:52s} {tot:.4f}   " + "  ".join(out))
    return tot


ps = lambda t: P[f"score_{t}"]
pc = lambda t: P[f"cost_{t}"]
base = run(ps, pc, tag="deployed E43")

hdr("P6.1  predicted vs true mean gains per family (dev), by tier")
for t in TIERS:
    S = P[f"score_{t}"]
    C = P[f"cost_{t}"]
    print(f"\n  -- {t}")
    print(f"  {'family':16s} {'n':>4s} | {'d10 pred':>9s} {'d10 true':>9s} {'bias':>7s} | "
          f"{'d21 pred':>9s} {'d21 true':>9s} {'bias':>7s} {'ratio':>6s} | "
          f"{'dc21 pred':>10s} {'dc21 true':>10s} | {'eff pred':>9s} {'eff true':>9s}")
    for f in fams:
        m = fam == f
        d10p = (S[m, 1] - S[m, 0]).mean(); d10t = (dv.score[m, 1] - dv.score[m, 0]).mean()
        d21p = (S[m, 2] - S[m, 1]).mean(); d21t = (dv.score[m, 2] - dv.score[m, 1]).mean()
        dcp = (C[m, 2] - C[m, 1]).mean() / C[:, 0].mean()
        dct = (dv.cost[m, 2] - dv.cost[m, 1]).mean() / dv.cost[:, 0].mean()
        print(f"  {f:16s} {m.sum():4d} | {d10p:+9.3f} {d10t:+9.3f} {d10p-d10t:+7.3f} | "
              f"{d21p:+9.3f} {d21t:+9.3f} {d21p-d21t:+7.3f} {d21p/max(d21t,1e-3):6.2f} | "
              f"{dcp:10.2f} {dct:10.2f} | {d21p/max(dcp,1e-9):9.4f} {d21t/max(dct,1e-9):9.4f}")

hdr("P6.2  per-item gain correlation with the truth, per family (premium tier)")
S = P["score_premium"]
print(f"  {'family':16s} {'n':>4s} {'corr(d10)':>10s} {'corr(d21)':>10s} {'corr(s_k1)':>11s} "
      f"{'corr(logc_k1)':>14s}")
for f in fams + ["ALL"]:
    m = np.ones(n, bool) if f == "ALL" else (fam == f)
    def cc(a, b):
        a, b = a[m], b[m]
        return np.corrcoef(a, b)[0, 1] if a.std() > 1e-12 and b.std() > 1e-12 else np.nan
    print(f"  {f:16s} {m.sum():4d} {cc(S[:,1]-S[:,0], dv.score[:,1]-dv.score[:,0]):10.3f} "
          f"{cc(S[:,2]-S[:,1], dv.score[:,2]-dv.score[:,1]):10.3f} "
          f"{cc(S[:,2], dv.score[:,2]):11.3f} "
          f"{cc(np.log(P['cost_premium'][:,2]), np.log(dv.cost[:,2])):14.3f}")

hdr("P6.3  TRAIN-ONLY family gain prior applied to the deployed dev predictions")
# w_f = (train true family d21) / (train true global d21), shrunk toward 1 by gamma.
d21_tr = {f: float((tr.score[famtr == f, 2] - tr.score[famtr == f, 1]).mean()) for f in fams}
d21_tr_all = float((tr.score[:, 2] - tr.score[:, 1]).mean())
d10_tr = {f: float((tr.score[famtr == f, 1] - tr.score[famtr == f, 0]).mean()) for f in fams}
d10_tr_all = float((tr.score[:, 1] - tr.score[:, 0]).mean())
print("  train-only relative family gain factors (clipped at 0):")
for f in fams:
    print(f"    {f:16s} w21={max(d21_tr[f],0)/d21_tr_all:6.3f}  w10={max(d10_tr[f],0)/d10_tr_all:6.3f}")


def reweight(gamma21, gamma10=0.0):
    def mk(t):
        S = P[f"score_{t}"].copy()
        d10 = S[:, 1] - S[:, 0]
        d21 = S[:, 2] - S[:, 1]
        w21 = np.array([(1 - gamma21) + gamma21 * max(d21_tr[f], 0.0) / d21_tr_all for f in fam])
        w10 = np.array([(1 - gamma10) + gamma10 * max(d10_tr[f], 0.0) / d10_tr_all for f in fam])
        out = S.copy()
        out[:, 1] = S[:, 0] + d10 * w10
        out[:, 2] = out[:, 1] + d21 * w21
        return out
    return mk


print()
for g in (0.0, 0.25, 0.5, 0.75, 1.0):
    tot = run(reweight(g), pc, tag=f"family gain prior on d21, gamma={g}")
    print(f"  {'':52s}    delta={tot-base:+.4f}")
print()
for g in (0.25, 0.5, 1.0):
    tot = run(reweight(g, g), pc, tag=f"family gain prior on d21 AND d10, gamma={g}")
    print(f"  {'':52s}    delta={tot-base:+.4f}")

hdr("P6.4  bootstrap EV of the family gain prior (880 resamples of dev, 300 reps)")


def boot(mk_s, mk_c, reps=300, seed=7):
    r = np.random.default_rng(seed)
    tots = []
    for _ in range(reps):
        take = r.choice(n, size=n, replace=True)
        sub = Split(dv.name, [dv.episode_ids[i] for i in take], [dv.texts[i] for i in take],
                    dv.score[take], dv.cost[take], dv.itok[take], dv.otok[take], dv.ngen[take])
        tot = 0.0
        for t in TIERS:
            rr = tier_result(mk_s(t)[take], mk_c(t)[take], sub, t, SAFE[t])
            tot += TIER_WEIGHT[t] * rr["tier_score"]
        tots.append(tot)
    return float(np.mean(tots)), float(np.std(tots))


b0 = boot(ps, pc)
print(f"  deployed                          EV={b0[0]:.4f} sd={b0[1]:.4f}")
for g in (0.25, 0.5, 1.0):
    b = boot(reweight(g), pc)
    print(f"  family gain prior gamma={g:<4}      EV={b[0]:.4f} sd={b[1]:.4f}  delta={b[0]-b0[0]:+.4f}")

hdr("P6.5  where does the deployed premium tier waste money?  (per family, k1 picks)")
r = tier_result(P["score_premium"], P["cost_premium"], dv, "premium", SAFE["premium"])
sel = r["sel"]
L = dv.cost[:, 0].sum()
print(f"  {'family':16s} {'nK':>4s} {'true cost of those k1 (light-units)':>36s} "
      f"{'% of premium budget':>20s} {'score bought':>13s} {'score/budget%':>14s}")
rows = []
for f in fams:
    m = (fam == f) & (sel == 2)
    if m.sum() == 0:
        continue
    c = dv.cost[m, 2].sum() / L
    g = (dv.score[m, 2] - dv.score[m, 1]).sum() / n
    rows.append((f, int(m.sum()), c, g))
for f, k, c, g in sorted(rows, key=lambda z: -z[2]):
    print(f"  {f:16s} {k:4d} {c:36.3f} {c/4*100:20.2f} {g:13.5f} {g/(c/4*100):14.5f}")
print(f"  {'TOTAL':16s} {sum(r[1] for r in rows):4d} {sum(r[2] for r in rows):36.3f} "
      f"{sum(r[2] for r in rows)/4*100:20.2f} {sum(r[3] for r in rows):13.5f}")
print("\n  counterfactual: send those k1 picks to mid instead, spend nothing else")
for f, k, c, g in sorted(rows, key=lambda z: -z[2]):
    print(f"    drop k1 on {f:16s}: frees {c/4*100:5.2f}% of the premium budget, loses {g:.5f} score")
