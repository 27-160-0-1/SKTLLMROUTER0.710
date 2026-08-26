# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a13 / provider view - part 10: upper bound on repairing the family classifier.

The regex classifier drops 98/456 confirmed DeepMind-Mathematics items and ~38
RuleTaker-style items into the fallback bucket 'gsm8k_or_other', whose mean is
then blended into the prediction at weight 0.15 and fed as a one-hot to the meta
GBM.  Measure (a) the residual bias this leaves in the deployed predictions and
(b) an OPTIMISTIC upper bound on fixing it: add the in-sample group-mean residual
back and re-run the allocator.  In-sample => strictly an upper bound.

Run: .venv/Scripts/python.exe experiments/lab/a13_provider_10_famfix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "lab"))
sys.path.insert(0, str(ROOT / "src"))

import labdata  # noqa: E402
from labdata import TIERS, TIER_WEIGHT, tier_result  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

dv = labdata.load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
fam = np.array([classify_family(t) for t in dv.texts])

dm = json.loads((ROOT / "data/sources/deepmind-mathematics-selection.v1.json").read_text(encoding="utf-8"))
dmids = {r["episode_id"] for rows in dm["splits"].values() for r in rows}
is_dm = np.array([e in dmids for e in dv.episode_ids])
import re
RT = re.compile(r"^[A-Z][a-z]+ is [a-z]+\.")
is_rt = np.array([bool(RT.match(t)) for t in dv.texts])

g = fam == "gsm8k_or_other"
groups = {
    "fallback & confirmed dmmath": g & is_dm,
    "fallback & ruletaker-style": g & is_rt & ~is_dm,
    "fallback & rest (true gsm8k)": g & ~is_dm & ~is_rt,
    "regex aime (mixed AIME+gsm8k)": fam == "aime",
    "regex dmmath (pure)": fam == "dmmath",
}

print("=" * 92)
print("A. residual bias of the deployed predictions inside each sub-population")
print("=" * 92)
print(f"{'group':32s} {'n':>4s} {'d s_light':>10s} {'d s_mid':>8s} {'d s_k1':>8s} "
      f"{'d log c_k1':>11s}")
for name, m in groups.items():
    if m.sum() == 0:
        continue
    S, C = P["score_fast"], P["cost_fast"]
    print(f"{name:32s} {m.sum():4d} "
          f"{np.mean(dv.score[m,0]-S[m,0]):10.4f} "
          f"{np.mean(dv.score[m,1]-S[m,1]):8.4f} "
          f"{np.mean(dv.score[m,2]-S[m,2]):8.4f} "
          f"{np.mean(np.log(dv.cost[m,2])-np.log(C[m,2])):11.4f}")


def final(mkS, mkC, tag):
    tot = 0.0
    parts = []
    for t in TIERS:
        r = tier_result(mkS(t), mkC(t), dv, t, SAFE[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]} {r['score']:.4f}/r{r['ratio']:.3f}"
                     f"{'' if r['passed'] else ' BUST'}")
    print(f"{tag:52s} {tot:.4f}   " + "  ".join(parts))
    return tot


ps = lambda t: P[f"score_{t}"]
pc = lambda t: P[f"cost_{t}"]
print()
print("=" * 92)
print("B. optimistic (in-sample) upper bound on repairing the fallback bucket")
print("=" * 92)
base = final(ps, pc, "deployed E43")

sub = {k: v for k, v in groups.items() if k.startswith("fallback")}


def corrected(t, do_score=True, do_cost=True, which=None):
    S = P[f"score_{t}"].copy()
    C = P[f"cost_{t}"].copy()
    for name, m in sub.items():
        if which is not None and name != which:
            continue
        if m.sum() == 0:
            continue
        if do_score:
            for j in range(3):
                S[m, j] += np.mean(dv.score[m, j] - P[f"score_{t}"][m, j])
        if do_cost:
            for j in range(3):
                d = np.mean(np.log(dv.cost[m, j]) - np.log(P[f"cost_{t}"][m, j]))
                C[m, j] *= np.exp(d)
    return S, C


final(lambda t: corrected(t)[0], pc, "  + group-mean SCORE correction (in-sample)")
final(ps, lambda t: corrected(t)[1], "  + group-mean COST correction (in-sample)")
final(lambda t: corrected(t)[0], lambda t: corrected(t)[1],
      "  + BOTH (in-sample upper bound)")
for name in sub:
    final(lambda t, n=name: corrected(t, which=n)[0],
          lambda t, n=name: corrected(t, which=n)[1], f"  only {name}")

print()
print("=" * 92)
print("C. same correction applied to EVERY regex family (upper bound on any")
print("   family-level recalibration whatsoever)")
print("=" * 92)


def fam_corrected(t):
    S = P[f"score_{t}"].copy()
    C = P[f"cost_{t}"].copy()
    for f in sorted(set(fam)):
        m = fam == f
        for j in range(3):
            S[m, j] += np.mean(dv.score[m, j] - P[f"score_{t}"][m, j])
            C[m, j] *= np.exp(np.mean(np.log(dv.cost[m, j]) - np.log(P[f"cost_{t}"][m, j])))
    return S, C


final(lambda t: fam_corrected(t)[0], pc, "  family-mean SCORE recalibration (in-sample)")
final(ps, lambda t: fam_corrected(t)[1], "  family-mean COST recalibration (in-sample)")
final(lambda t: fam_corrected(t)[0], lambda t: fam_corrected(t)[1],
      "  family-mean BOTH (in-sample)")
