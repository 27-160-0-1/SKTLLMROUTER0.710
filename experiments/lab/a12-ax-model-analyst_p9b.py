# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a12 P9b -- mechanism check: WHY does the rescue blend raise the bootstrap EV?
Is it more score, or fewer busts?"""
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
AB = {f: tuple(np.linalg.lstsq(np.column_stack([np.ones((famtr == f).sum()), tr.score[famtr == f, 1]]),
                               tr.score[famtr == f, 2], rcond=None)[0]) for f in fams}
A = np.array([AB[f][0] for f in fam]); B = np.array([AB[f][1] for f in fam])


def mkS(alpha):
    def f(t):
        S = P[f"score_{t}"].copy()
        if t == "premium":
            S[:, 2] = (1 - alpha) * S[:, 2] + alpha * np.clip(A + B * S[:, 1], 0, 1)
        return S
    return f


print("=" * 96)
print("P9b  per-tier bust rate and pass-conditional score, 3 seeds x 400 resamples of dev")
print("=" * 96)
print(f"  {'alpha':>6s} " + " ".join(f"{t+'_bust':>13s} {t+'_score':>13s}" for t in TIERS) + f" {'EV':>8s}")
for a in (0.0, 0.5, 1.0):
    agg = {t: {"bust": [], "sc": []} for t in TIERS}
    evs = []
    for seed in (7, 17, 23):
        r = np.random.default_rng(seed)
        for _ in range(400):
            take = r.choice(n, size=n, replace=True)
            sub = Split(dv.name, [dv.episode_ids[i] for i in take], [dv.texts[i] for i in take],
                        dv.score[take], dv.cost[take], dv.itok[take], dv.otok[take], dv.ngen[take])
            tot = 0.0
            for t in TIERS:
                rr = tier_result(mkS(a)(t)[take], P[f"cost_{t}"][take], sub, t, SAFE[t])
                agg[t]["bust"].append(0 if rr["passed"] else 1)
                agg[t]["sc"].append(rr["score"])
                tot += TIER_WEIGHT[t] * rr["tier_score"]
            evs.append(tot)
    print(f"  {a:6.2f} " + " ".join(f"{np.mean(agg[t]['bust']):13.4f} {np.mean(agg[t]['sc']):13.4f}"
                                    for t in TIERS) + f" {np.mean(evs):8.4f}")

print("\n  composition of the premium k1 basket on dev itself:")
for a in (0.0, 1.0):
    rr = tier_result(mkS(a)("premium"), P["cost_premium"], dv, "premium", SAFE["premium"])
    sel = rr["sel"]
    k1 = sel == 2
    L = dv.cost[:, 0].sum()
    comp = ", ".join(f"{f}:{int(((fam==f)&k1).sum())}" for f in fams if ((fam == f) & k1).sum())
    print(f"   alpha={a}: n_k1={k1.sum()} cost={dv.cost[k1,2].sum()/L:.3f} light-units  "
          f"mean k1 cost/item={dv.cost[k1,2].mean()/dv.cost[:,0].mean():.1f} units  "
          f"realised d21 on picks={(dv.score[k1,2]-dv.score[k1,1]).mean():+.3f}")
    print(f"      {comp}")
    print(f"      picks where true s_mid==1: {int((k1 & (dv.score[:,1]==1.0)).sum())}"
          f"  where true s_mid==0: {int((k1 & (dv.score[:,1]==0.0)).sum())}")
