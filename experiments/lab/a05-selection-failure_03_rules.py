# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a05 step 3: candidate decision rules.  Each rule is a mask/transform applied to
the DEPLOYED predictions only (no label access at decision time); the resulting
allocation is scored honestly on the true score/cost, and also on the EB
expected-p surface to check the label-noise caveat.

Rule families whose parameters are fitted are fitted on TRAIN ONLY where noted.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result, MODEL_IDS
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.similarity import classify_family

tr = load_split("train"); dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
Cc = np.load(ROOT / "experiments/lab/a05-selection-failure_cache.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
N = len(dv); IDX = np.arange(N)
FAM = Cc["fam"]; PHAT = Cc["phat"]
FAMTR = np.array([classify_family(t) for t in tr.texts])
L = dv.cost[:, 0].sum()
NEG = -1e9

# ------------------------------------------------------------------ E42 quantified
print("=== E42 restated with numbers: perfect scores + today's cost model ===")
for lbl, S, Cst in (("pred score, pred cost", lambda t: P[f"score_{t}"], lambda t: P[f"cost_{t}"]),
                    ("TRUE score, pred cost", lambda t: dv.score, lambda t: P[f"cost_{t}"]),
                    ("TRUE score, TRUE cost", lambda t: dv.score, lambda t: dv.cost)):
    parts = []
    tot = 0.0
    for t in TIERS:
        r = tier_result(S(t), Cst(t), dv, t, SAFE[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]}: sc={r['score']:.4f} true_ratio={r['ratio']:.3f}/{TIER_MULT[t]}"
                     f"{'' if r['passed'] else ' BUST'}")
    print(f"  {lbl:22s} final={tot:.4f}  " + "  ".join(parts))

# ------------------------------------------------------------------ helper
def evaluate(name, mk_score, mk_cost=None, safeties=None, quiet=False):
    """mk_score(tier)->(880,3) decision scores.  Returns dict of results."""
    mk_cost = mk_cost or (lambda t: P[f"cost_{t}"])
    safeties = safeties or SAFE
    tot_r = tot_e = 0.0
    parts = []
    sels = {}
    for t in TIERS:
        ps = mk_score(t)
        pc = mk_cost(t)
        r = tier_result(ps, pc, dv, t, safeties[t])
        sel = r["sel"]; sels[t] = sel
        e = PHAT[IDX, sel].mean() if r["passed"] else 0.0
        tot_r += TIER_WEIGHT[t] * r["tier_score"]
        tot_e += TIER_WEIGHT[t] * e
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else '!'}")
    if not quiet:
        print(f"  {name:46s} realised={tot_r:.4f}  EB-exp={tot_e:.4f}  " + " ".join(parts))
    return dict(realised=tot_r, eb=tot_e, sels=sels)

print("\n=== rule battery: deployed safety (.98/.89/.88), decisions from predictions only ===")
base = evaluate("R0 baseline (deployed E43)", lambda t: P[f"score_{t}"])

def ban(js):
    def mk(t):
        S = P[f"score_{t}"].copy()
        for j in js:
            S[:, j] = NEG
        return S
    return mk

evaluate("R1 ban MID (light/k1 only)", ban([1]))
evaluate("R2 ban K1  (light/mid only)", ban([2]))

# R3 gain floor on mid
print()
for th in (0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.2):
    def mk(t, th=th):
        S = P[f"score_{t}"].copy()
        g = S[:, 1] - S[:, 0]
        S[g < th, 1] = NEG
        return S
    evaluate(f"R3 mid only if pred gain(mid-light) >= {th:.2f}", mk)

# R4 force light when predicted light score is already high
print()
for th in (0.75, 0.8, 0.85, 0.9, 0.95):
    def mk(t, th=th):
        S = P[f"score_{t}"].copy()
        m = S[:, 0] >= th
        S[m, 1] = NEG; S[m, 2] = NEG
        return S
    evaluate(f"R4 force light when pred s_light >= {th:.2f}", mk)

# R5 monotone projection across models (isotonic in model strength)
print()
def mono(t):
    S = P[f"score_{t}"].copy()
    S[:, 1] = np.maximum(S[:, 1], S[:, 0])
    S[:, 2] = np.maximum(S[:, 2], S[:, 1])
    return S
evaluate("R5a monotone-up projection s_L<=s_M<=s_K", mono)
def mono_ban(t):
    S = P[f"score_{t}"].copy()
    S[S[:, 1] < S[:, 0], 1] = NEG
    S[S[:, 2] < S[:, 1], 2] = NEG
    return S
evaluate("R5b ban any model whose pred score is not monotone", mono_ban)

# R6 per-family bans, fitted on TRAIN (efficiency of the family's mean upgrade)
print("\n=== R6 per-family diagnostics fitted on TRAIN (gain per unit of extra cost) ===")
famlist = sorted(set(FAMTR.tolist()) | set(FAM.tolist()))
print(f"{'family':14s} {'n_tr':>5s} {'train gain M-L':>14s} {'train dc M/L':>12s} {'eff_M':>8s} "
      f"{'train gain K-L':>14s} {'train dc K/L':>12s} {'eff_K':>8s}")
eff_m = {}; eff_k = {}
for f_ in famlist:
    m = FAMTR == f_
    if m.sum() == 0:
        continue
    gm = (tr.score[m, 1] - tr.score[m, 0]).mean()
    gk = (tr.score[m, 2] - tr.score[m, 0]).mean()
    dcm = (tr.cost[m, 1] - tr.cost[m, 0]).mean() / tr.cost[m, 0].mean()
    dck = (tr.cost[m, 2] - tr.cost[m, 0]).mean() / tr.cost[m, 0].mean()
    eff_m[f_] = gm / max(dcm, 1e-9); eff_k[f_] = gk / max(dck, 1e-9)
    print(f"{f_:14s} {m.sum():5d} {gm:+14.4f} {dcm:12.2f} {eff_m[f_]:8.4f} "
          f"{gk:+14.4f} {dck:12.2f} {eff_k[f_]:8.4f}")

print("\n  per-family bans (threshold on TRAIN efficiency):")
for th in (0.0, 0.01, 0.02, 0.03, 0.05):
    banm = {f for f in famlist if eff_m.get(f, 1) < th}
    bank = {f for f in famlist if eff_k.get(f, 1) < th}
    def mk(t, banm=banm, bank=bank):
        S = P[f"score_{t}"].copy()
        S[np.isin(FAM, list(banm)), 1] = NEG
        S[np.isin(FAM, list(bank)), 2] = NEG
        return S
    evaluate(f"R6 ban fam-mid{sorted(banm)} fam-k1{sorted(bank)}"[:44], mk)

# R7 efficiency floor: require pred gain / pred extra cost above a quantile
print()
for q in (0.0, 0.1, 0.2, 0.3):
    def mk(t, q=q):
        S = P[f"score_{t}"].copy(); Cq = P[f"cost_{t}"]
        for j in (1, 2):
            e = (S[:, j] - S[:, 0]) / np.maximum(Cq[:, j] - Cq[:, 0], 1e-9)
            if q > 0:
                thr = np.quantile(e, q)
                S[e < thr, j] = NEG
        return S
    evaluate(f"R7 efficiency floor at quantile {q:.1f}", mk)
