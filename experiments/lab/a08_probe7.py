# SPDX-License-Identifier: Apache-2.0
"""a08 probe 7: does the deployed pipeline already know that `gsm8k_or_other`
is TWO sources?  (ngen=4 easy vs ngen=2 hard)  And what is a correct split worth?

Everything below is measured on dev with reports/lab/dev_preds_e43.npz.
The "corrected" predictions use train-fitted subgroup residual offsets only.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import (load_all, MODEL_IDS, TIERS, TIER_WEIGHT, tier_result)  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

tr, dv = load_all()
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
fam_tr = np.array([classify_family(t) for t in tr.texts])
fam_dv = np.array([classify_family(t) for t in dv.texts])

g_tr = fam_tr == "gsm8k_or_other"
g_dv = fam_dv == "gsm8k_or_other"
n4_tr = tr.ngen[:, 0] == 4
n4_dv = dv.ngen[:, 0] == 4

print("=== dev gsm8k_or_other split by hidden ngen ===")
for tag, m in (("ngen=4 (real GSM8K)", g_dv & n4_dv), ("ngen=2 (other)", g_dv & ~n4_dv)):
    print(f"  {tag:22s} n={m.sum():3d}  true score={np.round(dv.score[m].mean(0),3)}  "
          f"pred score={np.round(P['score_fast'][m].mean(0),3)}")
    print(f"  {'':22s}      true logcost={np.round(np.log(dv.cost[m]).mean(0),3)}  "
          f"pred logcost={np.round(np.log(P['cost_fast'][m]).mean(0),3)}")

print("\n=== residuals of the deployed prediction on the two subgroups ===")
for j, m_ in enumerate(MODEL_IDS):
    a = g_dv & n4_dv
    b = g_dv & ~n4_dv
    rs_a = (P["score_fast"][a, j] - dv.score[a, j]).mean()
    rs_b = (P["score_fast"][b, j] - dv.score[b, j]).mean()
    rc_a = (np.log(P["cost_fast"][a, j]) - np.log(dv.cost[a, j])).mean()
    rc_b = (np.log(P["cost_fast"][b, j]) - np.log(dv.cost[b, j])).mean()
    print(f"  {m_:11s} score bias  ngen4={rs_a:+.3f}  ngen2={rs_b:+.3f}   "
          f"logcost bias ngen4={rc_a:+.3f} ngen2={rc_b:+.3f}")

print("\n=== every family: deployed bias, to put the gsm8k split in context ===")
print(f"  {'family':16s} {'n':>4s} " + " ".join(f"{'s'+m[:4]:>7s}" for m in MODEL_IDS) +
      " " + " ".join(f"{'c'+m[:4]:>7s}" for m in MODEL_IDS))
for f in sorted(set(fam_dv)):
    m = fam_dv == f
    sb = [(P["score_fast"][m, j] - dv.score[m, j]).mean() for j in range(3)]
    cb = [(np.log(P["cost_fast"][m, j]) - np.log(dv.cost[m, j])).mean() for j in range(3)]
    print(f"  {f:16s} {m.sum():4d} " + " ".join(f"{v:+7.3f}" for v in sb) +
          " " + " ".join(f"{v:+7.3f}" for v in cb))

# ---------------------------------------------------------------- ngen detector
print("\n=== prompt-only ngen detector (train-fit -> dev), gsm8k_or_other only ===")


def feats(texts):
    out = []
    for t in texts:
        out.append([len(t), t.count("$"), t.count("\\"), len(re.findall(r"\d", t)),
                    t.count("?"), t.count("\n"), len(re.findall(r"[A-Za-z]+", t)),
                    float("\\frac" in t or "\\sqrt" in t or "$" in t),
                    float(t.rstrip().endswith("?")),
                    len(re.findall(r"\b(?:[Ff]ind|[Cc]ompute|[Ll]et|[Pp]rove|[Dd]enote)\b", t))])
    return np.array(out, dtype=float)


from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

Xtr = feats([t for t, m in zip(tr.texts, g_tr) if m])
ytr = n4_tr[g_tr].astype(int)
Xdv = feats([t for t, m in zip(dv.texts, g_dv) if m])
ydv = n4_dv[g_dv].astype(int)
clf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
                                     random_state=0).fit(Xtr, ytr)
pp = clf.predict_proba(Xdv)[:, 1]
print(f"  dev AUC={roc_auc_score(ydv, pp):.4f} acc={( (pp>0.5).astype(int)==ydv).mean():.4f} "
      f"(base {max(ydv.mean(),1-ydv.mean()):.4f}, n={len(ydv)})")

# ---------------------------------------------------------------- value of the split
print("\n=== what is a correct subgroup correction worth on the final score? ===")
SUB = {}
sub_tr = np.where(g_tr, np.where(n4_tr, "gsm4", "gsm2"), fam_tr)
sub_dv_true = np.where(g_dv, np.where(n4_dv, "gsm4", "gsm2"), fam_dv)
sub_dv_pred = sub_dv_true.copy()
k = 0
for i in np.where(g_dv)[0]:
    sub_dv_pred[i] = "gsm4" if pp[k] > 0.5 else "gsm2"
    k += 1


def corrected(tier, subs_dv, only=("gsm2", "gsm4")):
    """shift predicted score / log-cost by the TRAIN mean residual of the subgroup.
    Train residuals are unavailable (we only have dev preds), so we use the dev
    residual of the OTHER half via 2-fold splitting -> honest-ish."""
    S = P[f"score_{tier}"].copy()
    C = np.log(P[f"cost_{tier}"]).copy()
    rng = np.random.default_rng(0)
    half = rng.integers(0, 2, size=len(S))
    for s in only:
        m = subs_dv == s
        for h in (0, 1):
            fit = m & (half != h)
            app = m & (half == h)
            if fit.sum() < 5 or app.sum() == 0:
                continue
            S[app] -= (P[f"score_{tier}"][fit] - dv.score[fit]).mean(0)
            C[app] -= (np.log(P[f"cost_{tier}"][fit]) - np.log(dv.cost[fit])).mean(0)
    return np.clip(S, 0, 1), np.exp(C)


def run(label, mk):
    tot = 0.0
    parts = []
    for t in TIERS:
        S, C = mk(t)
        best = None
        for s in np.arange(0.60, 1.401, 0.005):
            r = tier_result(S, C, dv, t, float(s))
            if r["passed"] and (best is None or r["score"] > best[0]):
                best = (r["score"], float(s))
        tot += TIER_WEIGHT[t] * best[0]
        parts.append(f"{t[:4]}={best[0]:.4f}@{best[1]:.3f}")
    print(f"  {label:52s} {tot:.4f}  " + " ".join(parts))
    return tot


run("deployed (best safety)", lambda t: (P[f"score_{t}"], P[f"cost_{t}"]))
run("+ gsm2/gsm4 subgroup mean correction (true ngen)",
    lambda t: corrected(t, sub_dv_true))
run("+ gsm2/gsm4 subgroup mean correction (predicted)",
    lambda t: corrected(t, sub_dv_pred))
run("+ ALL-family subgroup mean correction (upper ref)",
    lambda t: corrected(t, sub_dv_true, only=tuple(sorted(set(sub_dv_true)))))
