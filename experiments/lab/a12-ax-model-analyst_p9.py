# SPDX-License-Identifier: Apache-2.0
"""a12 P9 -- confirmation run for the rescue reconstruction + waste accounting."""
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
n = len(dv)
AB21 = {}
for f in fams:
    m = famtr == f
    AB21[f] = tuple(np.linalg.lstsq(np.column_stack([np.ones(m.sum()), tr.score[m, 1]]),
                                    tr.score[m, 2], rcond=None)[0])
A = np.array([AB21[f][0] for f in fam]); B = np.array([AB21[f][1] for f in fam])


def hdr(t):
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


def mkS(alpha, tiers=("premium",)):
    def f(t):
        S = P[f"score_{t}"].copy()
        if t in tiers:
            S[:, 2] = (1 - alpha) * S[:, 2] + alpha * np.clip(A + B * S[:, 1], 0, 1)
        return S
    return f


hdr("P9.1  WASTE accounting: what the deployed premium k1 picks actually bought")
r = tier_result(P["score_premium"], P["cost_premium"], dv, "premium", SAFE["premium"])
sel = r["sel"]
k1 = sel == 2
L = dv.cost[:, 0].sum()
print(f"  premium k1 picks: {k1.sum()}  true cost {dv.cost[k1,2].sum()/L:.3f} light-units "
      f"({dv.cost[k1,2].sum()/L/4*100:.1f}% of the premium budget)")
for lab, m in (("mid already scored 1.0", k1 & (dv.score[:, 1] == 1.0)),
               ("mid scored >=0.5", k1 & (dv.score[:, 1] >= 0.5)),
               ("mid scored 0", k1 & (dv.score[:, 1] == 0.0)),
               ("k1 <= mid (no gain)", k1 & (dv.score[:, 2] <= dv.score[:, 1])),
               ("k1 < mid (regression)", k1 & (dv.score[:, 2] < dv.score[:, 1]))):
    c = dv.cost[m, 2].sum() / L
    g = (dv.score[m, 2] - dv.score[m, 1]).sum() / n
    print(f"    {lab:26s} n={m.sum():4d}  cost={c:6.3f} light-units ({c/4*100:5.2f}% of budget)  "
          f"score bought={g:+.5f}")
print(f"  -> {(k1 & (dv.score[:,2] <= dv.score[:,1])).sum()}/{k1.sum()} = "
      f"{(k1 & (dv.score[:,2] <= dv.score[:,1])).sum()/k1.sum()*100:.0f}% of premium k1 spend bought nothing")
print(f"     and they consumed "
      f"{dv.cost[k1 & (dv.score[:,2] <= dv.score[:,1]),2].sum()/dv.cost[k1,2].sum()*100:.0f}% of the k1 spend")

hdr("P9.2  multi-seed bootstrap confirmation of the premium-only rescue blend")


def boot(mk, reps=400, seed=7):
    r = np.random.default_rng(seed)
    tots = []
    for _ in range(reps):
        take = r.choice(n, size=n, replace=True)
        sub = Split(dv.name, [dv.episode_ids[i] for i in take], [dv.texts[i] for i in take],
                    dv.score[take], dv.cost[take], dv.itok[take], dv.otok[take], dv.ngen[take])
        tot = 0.0
        for t in TIERS:
            rr = tier_result(mk(t)[take], P[f"cost_{t}"][take], sub, t, SAFE[t])
            tot += TIER_WEIGHT[t] * rr["tier_score"]
        tots.append(tot)
    return float(np.mean(tots))


alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
print(f"  {'alpha':>6s} " + " ".join(f"{'seed'+str(s):>9s}" for s in (7, 17, 23)) + f" {'mean':>9s} {'delta':>9s}")
base_rows = None
for a in alphas:
    vals = [boot(mkS(a), seed=s) for s in (7, 17, 23)]
    if base_rows is None:
        base_rows = np.mean(vals)
    print(f"  {a:6.2f} " + " ".join(f"{v:9.4f}" for v in vals) +
          f" {np.mean(vals):9.4f} {np.mean(vals)-base_rows:+9.4f}")

hdr("P9.3  point held-out score on dev, premium-only rescue, fine alpha grid")
print(f"  {'alpha':>6s} {'final':>8s} {'premium':>9s} {'ratio':>7s} {'n_k1':>5s}")
for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    tot = 0.0
    for t in TIERS:
        rr = tier_result(mkS(a)(t), P[f"cost_{t}"](0) if False else P[f"cost_{t}"], dv, t, SAFE[t])
        tot += TIER_WEIGHT[t] * rr["tier_score"]
        if t == "premium":
            pr = rr
    print(f"  {a:6.2f} {tot:8.4f} {pr['score']:9.4f} {pr['ratio']:7.3f} {int((pr['sel']==2).sum()):5d}")

hdr("P9.4  distribution-shift stress of the rescue rule (family reweighting)")
rng = np.random.default_rng(11)
scen = {"nominal": {}, "reasoning x3": {"aime": 3.0, "code": 3.0, "dmmath": 3.0},
        "ruletaker/longdoc x3": {"ruletaker": 3.0, "longdoc": 3.0},
        "belebele/truthfulqa x3": {"belebele": 3.0, "truthfulqa": 3.0},
        "unseen-family proxy: drop code+dmmath": {"code": 0.0, "dmmath": 0.0}}
print(f"  {'scenario':40s} {'deployed':>9s} {'rescue a=0.5':>13s} {'rescue a=1.0':>13s}")
for name, w in scen.items():
    ww = np.array([w.get(f, 1.0) for f in fam]); ww = ww / ww.sum()
    out = []
    for a in (0.0, 0.5, 1.0):
        r2 = np.random.default_rng(11)
        tots = []
        for _ in range(250):
            take = r2.choice(n, size=n, replace=True, p=ww)
            sub = Split(dv.name, [dv.episode_ids[i] for i in take], [dv.texts[i] for i in take],
                        dv.score[take], dv.cost[take], dv.itok[take], dv.otok[take], dv.ngen[take])
            tot = 0.0
            for t in TIERS:
                rr = tier_result(mkS(a)(t)[take], P[f"cost_{t}"][take], sub, t, SAFE[t])
                tot += TIER_WEIGHT[t] * rr["tier_score"]
            tots.append(tot)
        out.append(np.mean(tots))
    print(f"  {name:40s} {out[0]:9.4f} {out[1]:13.4f} {out[2]:13.4f}")
