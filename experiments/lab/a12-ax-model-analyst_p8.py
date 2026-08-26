# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a12 P8 -- the RESCUE reconstruction of the k1 score.

P1.3 showed that conditional on the mid outcome, k1's behaviour is almost a
family constant:
    a_f = P(k1 ok | mid ok)      (0.81..0.99, mostly ~0.97)
    b_f = P(k1 ok | mid fails)   (0.12..0.91, the informative one)
so
    E[s_k1] ~= a_f * p_mid + b_f * (1 - p_mid).
This is a 2-parameter-per-family linear map from the mid score to the k1 score.
It is NOT a new feature and NOT a new learned head: it re-uses the existing mid
prediction.  Question: does it beat the deployed k1 head, and does blending it in
help or hurt the final score (E42 selection-bias check included)?
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
n = len(dv)


def hdr(t):
    print("\n" + "=" * 108)
    print(t)
    print("=" * 108)


# ---- family constants estimated on TRAIN ONLY ------------------------------
hdr("P8.1  train-only conditional constants   E[s_hi | s_lo] = A + B * s_lo")
AB21, AB10 = {}, {}
print(f"  {'family':16s} {'n_tr':>5s} | {'A21':>7s} {'B21':>7s} | {'A10':>7s} {'B10':>7s} | "
      f"{'P(k|m ok)':>10s} {'P(k|m fail)':>12s}")
for f in fams:
    m = famtr == f
    x, y = tr.score[m, 1], tr.score[m, 2]
    Xd = np.column_stack([np.ones(m.sum()), x])
    w = np.linalg.lstsq(Xd, y, rcond=None)[0]
    AB21[f] = tuple(w)
    x0, y0 = tr.score[m, 0], tr.score[m, 1]
    w0 = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), x0]), y0, rcond=None)[0]
    AB10[f] = tuple(w0)
    ok = tr.score[m, 1] >= 0.5
    a = float((tr.score[m, 2] >= 0.5)[ok].mean()) if ok.sum() else np.nan
    b = float((tr.score[m, 2] >= 0.5)[~ok].mean()) if (~ok).sum() else np.nan
    print(f"  {f:16s} {m.sum():5d} | {w[0]:+7.3f} {w[1]:+7.3f} | {w0[0]:+7.3f} {w0[1]:+7.3f} | "
          f"{a:10.3f} {b:12.3f}")

hdr("P8.2  quality of the reconstruction on DEV")


def rec21(smid):
    A = np.array([AB21[f][0] for f in fam]); B = np.array([AB21[f][1] for f in fam])
    return np.clip(A + B * smid, 0.0, 1.0)


def rec10(slight):
    A = np.array([AB10[f][0] for f in fam]); B = np.array([AB10[f][1] for f in fam])
    return np.clip(A + B * slight, 0.0, 1.0)


Sp = P["score_premium"]
tests = {
    "deployed k1 head": Sp[:, 2],
    "rescue(TRUE s_mid)": rec21(dv.score[:, 1]),
    "rescue(PRED s_mid)": rec21(Sp[:, 1]),
    "family mean k1 (train)": np.array([tr.score[famtr == f, 2].mean() for f in fam]),
    "0.5*deployed + 0.5*rescue(pred)": 0.5 * Sp[:, 2] + 0.5 * rec21(Sp[:, 1]),
}
print(f"  {'estimator':34s} {'corr(s_k1)':>11s} {'rmse':>8s} | {'corr(d21)':>10s} {'rmse(d21)':>10s}")
for k, v in tests.items():
    d = v - Sp[:, 1]
    dt = dv.score[:, 2] - dv.score[:, 1]
    print(f"  {k:34s} {np.corrcoef(v, dv.score[:,2])[0,1]:11.3f} "
          f"{np.sqrt(((v-dv.score[:,2])**2).mean()):8.3f} | "
          f"{np.corrcoef(d, dt)[0,1]:10.3f} {np.sqrt(((d-dt)**2).mean()):10.3f}")
print("\n  per family corr(d21) of deployed vs rescue(pred s_mid):")
print(f"  {'family':16s} {'n':>4s} {'deployed':>9s} {'rescue':>9s} {'blend .5':>9s}")
for f in fams:
    m = fam == f
    dt = (dv.score[:, 2] - dv.score[:, 1])[m]
    a = (Sp[:, 2] - Sp[:, 1])[m]
    b = (rec21(Sp[:, 1]) - Sp[:, 1])[m]
    c = 0.5 * a + 0.5 * b
    def cc(u):
        return np.corrcoef(u, dt)[0, 1] if u.std() > 1e-12 and dt.std() > 1e-12 else np.nan
    print(f"  {f:16s} {m.sum():4d} {cc(a):9.3f} {cc(b):9.3f} {cc(c):9.3f}")

hdr("P8.3  end-to-end effect on the deployed allocator (dev held-out)")


def mk(alpha, tiers=("fast", "balanced", "premium")):
    def f(t):
        S = P[f"score_{t}"].copy()
        if t in tiers:
            S[:, 2] = (1 - alpha) * S[:, 2] + alpha * rec21(S[:, 1])
        return S
    return f


pc = lambda t: P[f"cost_{t}"]


def run(S, tag):
    tot, parts = 0.0, []
    sels = {}
    for t in TIERS:
        r = tier_result(S(t), pc(t), dv, t, SAFE[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else ' BUST'}")
        sels[t] = r
    print(f"  {tag:44s} {tot:.4f}   " + "  ".join(parts))
    return tot, sels


base, base_sel = run(lambda t: P[f"score_{t}"], "deployed E43")
for a in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
    tot, sel = run(mk(a), f"blend rescue into k1 head, alpha={a}")
    print(f"  {'':44s}     delta={tot-base:+.4f}  premium k1 picks="
          f"{int((sel['premium']['sel']==2).sum())} (deployed {int((base_sel['premium']['sel']==2).sum())})")
print()
for a in (0.25, 0.5, 1.0):
    tot, sel = run(mk(a, tiers=("premium",)), f"premium tier only, alpha={a}")
    print(f"  {'':44s}     delta={tot-base:+.4f}")

hdr("P8.4  E42 selection-bias check: true/predicted cost of the UPGRADED set")
for name, S in (("deployed", lambda t: P[f"score_{t}"]), ("rescue a=0.5", mk(0.5)),
                ("rescue a=1.0", mk(1.0))):
    row = []
    for t in TIERS:
        r = tier_result(S(t), pc(t), dv, t, SAFE[t])
        sel = r["sel"]
        up = sel > 0
        tc = dv.cost[np.arange(n), sel][up].sum()
        pcst = P[f"cost_{t}"][np.arange(n), sel][up].sum()
        row.append(f"{t[:4]}: n_up={up.sum():3d} true/pred cost of upgrades={tc/pcst:.3f}")
    print(f"  {name:14s} " + "  ".join(row))

hdr("P8.5  bootstrap EV (880 resamples of dev, 300 reps)")


def boot(mkS, reps=300, seed=7):
    r = np.random.default_rng(seed)
    tots = []
    for _ in range(reps):
        take = r.choice(n, size=n, replace=True)
        sub = Split(dv.name, [dv.episode_ids[i] for i in take], [dv.texts[i] for i in take],
                    dv.score[take], dv.cost[take], dv.itok[take], dv.otok[take], dv.ngen[take])
        tot = 0.0
        for t in TIERS:
            rr = tier_result(mkS(t)[take], P[f"cost_{t}"][take], sub, t, SAFE[t])
            tot += TIER_WEIGHT[t] * rr["tier_score"]
        tots.append(tot)
    return float(np.mean(tots)), float(np.std(tots))


b0 = boot(lambda t: P[f"score_{t}"])
print(f"  deployed              EV={b0[0]:.4f} sd={b0[1]:.4f}")
for a in (0.2, 0.5, 1.0):
    b = boot(mk(a))
    print(f"  rescue alpha={a:<4}      EV={b[0]:.4f} sd={b[1]:.4f}  delta={b[0]-b0[0]:+.4f}")
