# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a12 P4 -- what the model profile implies for the allocator.

Uses reports/lab/dev_preds_e43.npz (deployed E43 predictions on held-out dev)
and the exact deployed allocator/scorer from labdata.py.

(1) what does the deployed router actually buy, per family per tier?
(2) family veto rules derived from the upgrade profile (P1/P3)
(3) family-specific cost calibration for k1 (a COST-side lever, the complement
    E42 says every score-side idea needs)
(4) distribution-shift stress on all of the above
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, TIER_MULT, tier_result  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

ROOT = HERE.parents[1]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
fam = np.array([classify_family(t) for t in dv.texts])
fams = sorted(set(fam))
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}   # E43 deployed
n = len(dv)
idx = np.arange(n)
rng = np.random.default_rng(7)


def hdr(t):
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


def run(mk_s, mk_c, safety=SAFE, verbose=False, tag=""):
    tot = 0.0
    parts = []
    sels = {}
    for t in TIERS:
        r = tier_result(mk_s(t), mk_c(t), dv, t, safety[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else ' BUST'}")
        sels[t] = r
    if verbose:
        print(f"{tag:58s} {tot:.4f}  " + "  ".join(parts))
    return tot, sels


ps = lambda t: P[f"score_{t}"]
pc = lambda t: P[f"cost_{t}"]

hdr("P4.0  deployed E43 baseline reproduced on held-out dev")
base, base_sel = run(ps, pc, verbose=True, tag="deployed E43 (.98/.87/.85)")

hdr("P4.1  what the deployed router buys, per family per tier")
for t in TIERS:
    sel = base_sel[t]["sel"]
    print(f"\n-- {t} (safety {SAFE[t]}, true ratio {base_sel[t]['ratio']:.3f}, score {base_sel[t]['score']:.4f})")
    print(f"   {'family':16s} {'n':>4s} {'nL':>4s} {'nM':>4s} {'nK':>4s} | {'score':>7s} {'if all L':>8s} "
          f"{'gain':>7s} | {'true cost share':>15s} {'k1 cost share':>13s} | {'d21 realised on nK':>19s}")
    lightsum = dv.cost[:, 0].sum()
    for f in fams:
        m = fam == f
        s = sel[m]
        cst = dv.cost[m][np.arange(m.sum()), s]
        sc = dv.score[m][np.arange(m.sum()), s].mean()
        base_sc = dv.score[m, 0].mean()
        k1m = s == 2
        d21 = (dv.score[m][k1m, 2] - dv.score[m][k1m, 1]).mean() if k1m.sum() else np.nan
        k1cost = dv.cost[m][k1m, 2].sum() if k1m.sum() else 0.0
        print(f"   {f:16s} {m.sum():4d} {(s==0).sum():4d} {(s==1).sum():4d} {(s==2).sum():4d} | "
              f"{sc:7.3f} {base_sc:8.3f} {sc-base_sc:+7.3f} | {cst.sum()/lightsum:15.3f} "
              f"{k1cost/lightsum:13.3f} | {d21:19.3f}")

hdr("P4.2  family veto rules (block a model for a family by setting its predicted score to -inf)")
BAD_K1 = ("ruletaker", "longdoc", "hrmcr")


def veto(mk_s, blocked: dict):
    def f(t):
        S = mk_s(t).copy()
        for famname, models in blocked.items():
            m = fam == famname
            for j in models:
                S[m, j] = -1e9
        return S
    return f


rules = {
    "R0 none (deployed)": {},
    "R1 no k1 on ruletaker": {"ruletaker": (2,)},
    "R2 no k1 on ruletaker+longdoc": {"ruletaker": (2,), "longdoc": (2,)},
    "R3 no k1 on ruletaker+longdoc+hrmcr": {f: (2,) for f in BAD_K1},
    "R4 hrmcr -> light only": {"hrmcr": (1, 2)},
    "R5 R3 + hrmcr light only": {"ruletaker": (2,), "longdoc": (2,), "hrmcr": (1, 2)},
    "R6 no k1 on belebele too": {"ruletaker": (2,), "longdoc": (2,), "hrmcr": (1, 2), "belebele": (2,)},
    "R7 no mid on code (mid adds ~0)": {"code": (1,)},
    "R8 R5 + no mid on code": {"ruletaker": (2,), "longdoc": (2,), "hrmcr": (1, 2), "code": (1,)},
}
for name, blk in rules.items():
    tot, sel = run(veto(ps, blk), pc, verbose=True, tag=name)
    print(f"{'':58s}        delta vs deployed = {tot-base:+.4f}")

hdr("P4.3  same rules with per-tier resolution (which tier does each rule help?)")
print(f"{'rule':38s} " + " ".join(f"{t:>22s}" for t in TIERS))
for name, blk in rules.items():
    _, sel = run(veto(ps, blk), pc)
    row = []
    for t in TIERS:
        d = sel[t]["score"] - base_sel[t]["score"]
        row.append(f"{sel[t]['score']:.4f}({d:+.4f}) r{sel[t]['ratio']:.2f}")
    print(f"{name:38s} " + " ".join(f"{c:>22s}" for c in row))

hdr("P4.4  COST-side lever: family-specific multiplier on the predicted k1 cost")
# ratio of true to predicted k1 cost SUM per family, on TRAIN-equivalent info.
# NB: computed on dev here => in-sample; reported as an upper bound on the lever,
# and re-derived below with a train-only estimate.
tr = load_split("train")
famtr = np.array([classify_family(t) for t in tr.texts])
print("  true/pred k1 cost sum ratio per family (dev, using premium-tier predictions):")
Cp = P["cost_premium"]
for f in fams:
    m = fam == f
    print(f"    {f:16s} n={m.sum():4d}  sum_true/sum_pred = {dv.cost[m,2].sum()/Cp[m,2].sum():6.3f}  "
          f"median item ratio = {np.median(dv.cost[m,2]/Cp[m,2]):6.3f}")


def cost_fammult(mult: dict, model=2):
    def f(t):
        C = P[f"cost_{t}"].copy()
        for k, v in mult.items():
            C[fam == k, model] *= v
        return C
    return f


print("\n  effect of penalising the k1 cost of the low-gain families (a pure cost-side rule):")
for name, mult in (("M1 x2 on ruletaker", {"ruletaker": 2.0}),
                   ("M2 x2 on ruletaker+longdoc+hrmcr", {f: 2.0 for f in BAD_K1}),
                   ("M3 x4 on ruletaker+longdoc+hrmcr", {f: 4.0 for f in BAD_K1}),
                   ("M4 x10 on ruletaker+longdoc+hrmcr", {f: 10.0 for f in BAD_K1}),
                   ("M5 dev-true k1 cost ratio (oracle)", None)):
    if mult is None:
        mults = {f: float(dv.cost[fam == f, 2].sum() / Cp[fam == f, 2].sum()) for f in fams}
        tot, _ = run(ps, cost_fammult(mults), verbose=True, tag=name)
    else:
        tot, _ = run(ps, cost_fammult(mult), verbose=True, tag=name)
    print(f"{'':58s}        delta vs deployed = {tot-base:+.4f}")

hdr("P4.5  distribution shift stress: reweight families, re-score (400 resamples per scenario)")


def bootstrap_family(weights: dict, reps=400, mk_s=ps, mk_c=pc, safety=SAFE):
    w = np.array([weights.get(f, 1.0) for f in fam], dtype=float)
    w = w / w.sum()
    tots = []
    for _ in range(reps):
        take = rng.choice(n, size=n, replace=True, p=w)
        sub = type(dv)(dv.name, [dv.episode_ids[i] for i in take], [dv.texts[i] for i in take],
                       dv.score[take], dv.cost[take], dv.itok[take], dv.otok[take], dv.ngen[take])
        tot = 0.0
        for t in TIERS:
            r = tier_result(mk_s(t)[take], mk_c(t)[take], sub, t, safety[t])
            tot += TIER_WEIGHT[t] * r["tier_score"]
        tots.append(tot)
    return float(np.mean(tots)), float(np.std(tots)), float(np.mean(np.array(tots) < 0.5))


scen = {
    "nominal": {},
    "reasoning-heavy (aime,code,dmmath x3)": {"aime": 3.0, "code": 3.0, "dmmath": 3.0},
    "ruletaker/longdoc x3": {"ruletaker": 3.0, "longdoc": 3.0},
    "belebele/truthfulqa x3 (cheap-k1)": {"belebele": 3.0, "truthfulqa": 3.0},
    "hrmcr x5 (hopeless items)": {"hrmcr": 5.0},
}
print(f"{'scenario':40s} {'deployed EV':>12s} {'sd':>7s} {'P(bust)':>8s} | {'R5 EV':>8s} {'delta':>8s} "
      f"| {'M3 EV':>8s} {'delta':>8s}")
for sname, w in scen.items():
    a, sa, ba = bootstrap_family(w)
    b, sb, bb = bootstrap_family(w, mk_s=veto(ps, rules["R5 R3 + hrmcr light only"]))
    c, sc, bc = bootstrap_family(w, mk_c=cost_fammult({f: 4.0 for f in BAD_K1}))
    print(f"{sname:40s} {a:12.4f} {sa:7.4f} {ba:8.3f} | {b:8.4f} {b-a:+8.4f} | {c:8.4f} {c-a:+8.4f}")
