# SPDX-License-Identifier: Apache-2.0
"""a06 step 8: (i) ablate the sub-family counterfactual split by split,
(ii) test shrinking the predicted score to the family mean inside the
families that have no measurable within-family signal.

All corrections are estimated on TRAIN, evaluated on the deployed E43 dev
held-out predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from a06_features import build  # noqa: E402
from labdata import TIERS, TIER_WEIGHT, tier_result  # noqa: E402

SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
tr, dv = build("train"), build("dev")
names = tr["names"]
sptr, spdv = tr["split"], dv["split"]
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
iq, idg = names.index("n_qmark"), names.index("frac_digit")


def make_groups(which):
    """which subset of splits to apply.  Returns (gtr, gdv)."""
    def lab(fam, X):
        out = np.array(fam, dtype=object).copy()
        if "aime" in which:
            m = fam == "aime"
            out[m & (X[:, iq] == 0)] = "aime.hard"
            out[m & (X[:, iq] >= 1)] = "aime.wordprob"
        if "gsm" in which:
            g = fam == "gsm8k_or_other"
            out[g & (X[:, idg] >= 0.08)] = "gsm.dmlike"
            out[g & (X[:, idg] < 0.08)] = "gsm.easy"
        return out.astype(str)
    return lab(tr["fam"], tr["X"]), lab(dv["fam"], dv["X"])


def deltas(gtr, gdv):
    ds, dc = {}, {}
    for g in sorted(set(gtr)):
        mt = gtr == g
        base = {"gsm": "gsm8k_or_other"}.get(g.split(".")[0], g.split(".")[0])
        mf = tr["fam"] == base
        ds[g] = sptr.score[mt].mean(0) - sptr.score[mf].mean(0)
        dc[g] = np.log(sptr.cost[mt]).mean(0) - np.log(sptr.cost[mf]).mean(0)
    return np.array([ds[g] for g in gdv]), np.array([dc[g] for g in gdv])


def run(tag, S_fn, C_fn, safety=SAFE):
    tot, parts = 0.0, []
    for t in TIERS:
        r = tier_result(S_fn(t), C_fn(t), spdv, t, safety[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else 'X'}")
    print(f"  {tag:46s} {tot:.4f}  " + " ".join(parts))
    return tot


print("=== (i) ablation of the sub-family correction, deployed safety .98/.87/.85 ===")
base = run("baseline", lambda t: P[f"score_{t}"], lambda t: P[f"cost_{t}"])
for which in (("aime",), ("gsm",), ("aime", "gsm")):
    gtr, gdv = make_groups(which)
    DS, DC = deltas(gtr, gdv)
    for w in (0.5, 1.0):
        run(f"{'+'.join(which):10s} score+cost w={w}",
            lambda t, DS=DS, w=w: P[f"score_{t}"] + w * DS,
            lambda t, DC=DC, w=w: P[f"cost_{t}"] * np.exp(w * DC))
    # cost-side only for the aime.hard group (the 3.25x under-prediction)
    if which == ("aime",):
        maskh = (gdv == "aime.hard")[:, None] * np.ones((1, 3))
        for w in (1.0,):
            run(f"aime.hard cost-only w={w}",
                lambda t: P[f"score_{t}"],
                lambda t, DC=DC, w=w: P[f"cost_{t}"] * np.exp(w * DC * maskh))
            run(f"aime.hard score+cost w={w}",
                lambda t, DS=DS, w=w: P[f"score_{t}"] + w * DS * maskh,
                lambda t, DC=DC, w=w: P[f"cost_{t}"] * np.exp(w * DC * maskh))

print("\n=== (ii) shrink predicted score to the family mean where within-family signal ~0 ===")
FLAT = {"belebele", "truthfulqa", "code", "longdoc", "ruletaker", "hrmcr"}
famdv = dv["fam"]
mu_tr = {f: sptr.score[tr["fam"] == f].mean(0) for f in set(tr["fam"])}
MU = np.array([mu_tr[f] for f in famdv])
IN = np.array([f in FLAT for f in famdv])[:, None]
for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
    run(f"shrink lam={lam} in {sorted(FLAT)[:3]}...",
        lambda t, lam=lam: np.where(IN, (1 - lam) * P[f"score_{t}"] + lam * MU, P[f"score_{t}"]),
        lambda t: P[f"cost_{t}"])
print("  (per-family, lam=1.0, one family at a time)")
for f in sorted(set(famdv)):
    INf = (famdv == f)[:, None]
    run(f"    shrink only {f}",
        lambda t, INf=INf: np.where(INf, MU, P[f"score_{t}"]),
        lambda t: P[f"cost_{t}"])

print("\n=== (iii) combined: sub-family correction + shrink on the flat families ===")
gtr, gdv = make_groups(("aime", "gsm"))
DS, DC = deltas(gtr, gdv)
for lam in (0.0, 0.5, 1.0):
    run(f"subfam w=1 + shrink lam={lam}",
        lambda t, lam=lam: np.where(IN, (1 - lam) * (P[f"score_{t}"] + DS) + lam * MU,
                                    P[f"score_{t}"] + DS),
        lambda t: P[f"cost_{t}"] * np.exp(DC))
