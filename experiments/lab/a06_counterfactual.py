# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a06 step 6: honest counterfactual for a SUB-FAMILY refinement.

The correction is estimated on TRAIN only (sub-family mean minus 9-family mean,
per model, for score and for log cost) and applied additively to the deployed
E43 held-out dev predictions.  Nothing is fitted on dev except the reported
safety sweep, which is shown as a full curve rather than a tuned point.
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

SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}   # E43 deployed


def subfam(fam, X, names):
    """Prompt-only sub-family label (refines 3 of the 9 regex families)."""
    iq, idg, ilx = names.index("n_qmark"), names.index("frac_digit"), names.index("latex_hits")
    ich = names.index("n_chars")
    out = np.array(fam, dtype=object).copy()
    m = fam == "aime"
    out[m & (X[:, iq] == 0)] = "aime.hard"
    out[m & (X[:, iq] >= 1)] = "aime.wordprob"
    g = fam == "gsm8k_or_other"
    out[g & (X[:, idg] >= 0.08)] = "gsm.dmlike"
    out[g & (X[:, idg] < 0.08)] = "gsm.easy"
    c = fam == "code"
    out[c & (X[:, ich] >= 600)] = "code.long"
    out[c & (X[:, ich] < 600)] = "code.short"
    return out.astype(str)


def main():
    tr, dv = build("train"), build("dev")
    names = tr["names"]
    str_, sdv = tr["split"], dv["split"]
    gtr = subfam(tr["fam"], tr["X"], names)
    gdv = subfam(dv["fam"], dv["X"], names)

    print("Sub-family table (train means; dev means in parentheses)")
    print(f"{'subfam':16s} {'n_tr':>5s} {'n_dv':>5s} | {'train s (l/m/k)':>24s} | "
          f"{'dev s (l/m/k)':>24s} | {'tr logc k1':>10s} {'dv logc k1':>10s}")
    for g in sorted(set(gtr)):
        mt, md = gtr == g, gdv == g
        st = str_.score[mt].mean(0)
        sd = sdv.score[md].mean(0) if md.sum() else np.full(3, np.nan)
        print(f"{g:16s} {mt.sum():5d} {md.sum():5d} | "
              f"{st[0]:7.3f}{st[1]:8.3f}{st[2]:8.3f} | {sd[0]:7.3f}{sd[1]:8.3f}{sd[2]:8.3f} | "
              f"{np.log(str_.cost[mt,2]).mean():10.3f} "
              f"{np.log(sdv.cost[md,2]).mean() if md.sum() else np.nan:10.3f}")

    # train-estimated corrections
    ds = {}
    dc = {}
    for g in sorted(set(gtr)):
        mt = gtr == g
        base_fam = g.split(".")[0]
        base_fam = {"gsm": "gsm8k_or_other"}.get(base_fam, base_fam)
        mf = tr["fam"] == base_fam
        ds[g] = str_.score[mt].mean(0) - str_.score[mf].mean(0)
        dc[g] = np.log(str_.cost[mt]).mean(0) - np.log(str_.cost[mf]).mean(0)
    print("\nTrain-estimated corrections (score delta / log-cost delta vs the parent family)")
    for g in sorted(set(gtr)):
        if np.abs(ds[g]).max() < 1e-9 and np.abs(dc[g]).max() < 1e-9:
            continue
        print(f"  {g:16s} ds={np.round(ds[g],3)}  dlogc={np.round(dc[g],3)}")

    DS = np.array([ds[g] for g in gdv])
    DC = np.array([dc[g] for g in gdv])

    P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)

    def run(tag, w_s, w_c, safety):
        tot, parts = 0.0, []
        for t in TIERS:
            S = P[f"score_{t}"] + w_s * DS
            C = P[f"cost_{t}"] * np.exp(w_c * DC)
            r = tier_result(S, C, sdv, t, safety[t])
            tot += TIER_WEIGHT[t] * r["tier_score"]
            parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else 'X'}")
        print(f"  {tag:44s} {tot:.4f}  " + " ".join(parts))
        return tot

    print("\n=== deployed safety .98/.87/.85 ===")
    base = run("baseline (E43 dev held-out preds)", 0.0, 0.0, SAFE)
    for w in (0.25, 0.5, 0.75, 1.0):
        run(f"score+cost correction w={w}", w, w, SAFE)
    for w in (0.5, 1.0):
        run(f"score-only correction w={w}  (E42 trap test)", w, 0.0, SAFE)
        run(f"cost-only  correction w={w}", 0.0, w, SAFE)

    print("\n=== best safety per tier (dev-tuned upper bound, same grid for all rows) ===")
    def best(w_s, w_c):
        tot = 0.0
        det = []
        for t in TIERS:
            S = P[f"score_{t}"] + w_s * DS
            C = P[f"cost_{t}"] * np.exp(w_c * DC)
            bb = None
            for sf in np.arange(0.60, 1.201, 0.005):
                r = tier_result(S, C, sdv, t, float(sf))
                if r["passed"] and (bb is None or r["score"] > bb[0]):
                    bb = (r["score"], float(sf), r["ratio"])
            tot += TIER_WEIGHT[t] * bb[0]
            det.append(f"{t[:4]}={bb[0]:.4f}@{bb[1]:.3f}")
        return tot, det
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        tot, det = best(w, w)
        print(f"  w={w:.2f} both      {tot:.4f}  " + " ".join(det))
    for w in (0.5, 1.0):
        tot, det = best(w, 0.0)
        print(f"  w={w:.2f} score-only{tot:.4f}  " + " ".join(det))
        tot, det = best(0.0, w)
        print(f"  w={w:.2f} cost-only {tot:.4f}  " + " ".join(det))

    print("\n=== upgrade-target cost bias (E42 diagnostic) at premium, safety .85 ===")
    for tag, w_s, w_c in (("baseline", 0.0, 0.0), ("both w=1", 1.0, 1.0),
                          ("score-only w=1", 1.0, 0.0)):
        S = P["score_premium"] + w_s * DS
        C = P["cost_premium"] * np.exp(w_c * DC)
        r = tier_result(S, C, sdv, "premium", 0.85)
        sel = r["sel"]
        up = sel > 0
        pc = C[np.arange(len(sdv)), sel][up].sum()
        tc = sdv.cost[np.arange(len(sdv)), sel][up].sum()
        print(f"  {tag:16s} n_up={int(up.sum()):4d} n_k1={int((sel==2).sum()):4d} "
              f"pred/true cost of upgrades = {pc/tc:.3f}  realised ratio={r['ratio']:.3f} "
              f"score={r['score']:.4f}")


if __name__ == "__main__":
    main()
