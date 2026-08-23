# SPDX-License-Identifier: Apache-2.0
"""a06 step 3: does the DEPLOYED E43 pipeline already know the sub-family split?

Compares predicted vs realised scores/costs inside each family and inside the
candidate sub-groups found in a06_subfam.py, on dev (held-out predictions).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from a06_features import build  # noqa: E402
from labdata import TIERS, MODEL_IDS  # noqa: E402

d = build("dev")
sp, fam, X, names = d["split"], d["fam"], d["X"], d["names"]
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
S = P["score_premium"]          # tier blends differ slightly; premium used for display
Sf = P["score_fast"]
C = P["cost_premium"]

iq = names.index("n_qmark")
idg = names.index("frac_digit")

GROUPS = {
    "aime & n_qmark==0 (true AIME)": (fam == "aime") & (X[:, iq] == 0),
    "aime & n_qmark>=1 (gsm-like)": (fam == "aime") & (X[:, iq] >= 1),
    "gsm8k & frac_digit>=.08 (dm-like)": (fam == "gsm8k_or_other") & (X[:, idg] >= 0.08),
    "gsm8k & frac_digit<.08": (fam == "gsm8k_or_other") & (X[:, idg] < 0.08),
}
for f in sorted(set(fam)):
    GROUPS[f"FAMILY {f}"] = fam == f

print("Dev, deployed E43 held-out predictions (premium-tier blend)")
print(f"{'group':38s} {'n':>4s} | {'true s (l/m/k)':>22s} | {'pred s':>22s} | "
      f"{'true logc k1':>12s} {'pred logc k1':>12s}")
for g, m in GROUPS.items():
    if m.sum() == 0:
        continue
    ts = sp.score[m].mean(0)
    ps = S[m].mean(0)
    print(f"{g:38s} {m.sum():4d} | {ts[0]:6.3f}{ts[1]:8.3f}{ts[2]:8.3f} | "
          f"{ps[0]:6.3f}{ps[1]:8.3f}{ps[2]:8.3f} | "
          f"{np.log(sp.cost[m,2]).mean():12.3f} {np.log(C[m,2]).mean():12.3f}")

print("\nPer-model correlation of prediction with truth, within family (dev)")
print(f"{'family':18s} {'n':>4s} " + " ".join(f"{m:>12s}" for m in MODEL_IDS) +
      "   corr(gain k1-mid)")
for f in sorted(set(fam)):
    m = fam == f
    cs = []
    for j in range(3):
        a, b = S[m, j], sp.score[m, j]
        cs.append(np.nan if (np.std(a) < 1e-9 or np.std(b) < 1e-9)
                  else np.corrcoef(a, b)[0, 1])
    pg = S[m, 2] - S[m, 1]
    tg = sp.score[m, 2] - sp.score[m, 1]
    cg = np.nan if (np.std(pg) < 1e-9 or np.std(tg) < 1e-9) else np.corrcoef(pg, tg)[0, 1]
    print(f"{f:18s} {m.sum():4d} " + " ".join(f"{c:12.3f}" for c in cs) + f"   {cg:12.3f}")

print("\nPooled dev corr (for reference):",
      [round(float(np.corrcoef(S[:, j], sp.score[:, j])[0, 1]), 3) for j in range(3)])

print("\nWhat the deployed selection actually does per family (premium tier, safety .85)")
from labdata import tier_result  # noqa: E402
r = tier_result(P["score_premium"], P["cost_premium"], sp, "premium", 0.85)
sel = r["sel"]
print(f"{'family':18s} {'n':>4s} {'%light':>7s} {'%mid':>6s} {'%k1':>6s} "
      f"{'score':>7s} {'best-fixed':>10s} {'oracle':>7s}")
for f in sorted(set(fam)):
    m = fam == f
    sc = sp.score[m][np.arange(m.sum()), sel[m]].mean()
    fixed = sp.score[m].mean(0)
    orc = sp.score[m].max(1).mean()
    print(f"{f:18s} {m.sum():4d} {100*np.mean(sel[m]==0):7.1f} {100*np.mean(sel[m]==1):6.1f} "
          f"{100*np.mean(sel[m]==2):6.1f} {sc:7.3f} {fixed.max():10.3f} {orc:7.3f}")
