# SPDX-License-Identifier: Apache-2.0
"""a03: audit of the regex family classifier that feeds the family-mean blend.

  audit   confusion of similarity.classify_family against data/gold sources
  rule    candidate corrections for the over-broad _AIME rule, scored on gold
  effect  family-mean predictor and the deployed-style family blend, current
          vs corrected family map (dev, ridge+legacy chain)
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lab"))

import labdata  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "a03curve", Path(__file__).with_name("a03-train-failure_curve.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

GOLD = json.loads((ROOT / "data/gold/gold-answers.v1.json").read_text(encoding="utf-8"))

# candidate discriminators: keep "aime" only if one of these fires
CANDS = {
    "latex_backslash": lambda t: "\\" in t,
    "tight_dollar": lambda t: bool(re.search(r"\$[^$\s]{1,24}\$", t)),
    "either": lambda t: "\\" in t or bool(re.search(r"\$[^$\s]{1,24}\$", t)),
    "no_currency": lambda t: not re.search(r"\$\d[\d,.]*\s", t),
}


def audit():
    tr, dv = labdata.load_all()
    rows = []
    for sp in (tr, dv):
        for i, eid in enumerate(sp.episode_ids):
            f = classify_family(sp.texts[i])
            g = GOLD.get(eid)
            rows.append((eid, f, (g or {}).get("source"), sp.texts[i], sp.score[i]))
    n_aime = sum(1 for r in rows if r[1] == "aime")
    cov = [r for r in rows if r[1] == "aime" and r[2]]
    print(f"== 'aime' bucket: {n_aime} of {len(rows)} episodes; "
          f"{len(cov)} have a public-source label in data/gold")
    from collections import Counter
    print("   sources of the covered ones:", Counter(r[2] for r in cov))
    print("== mean score of the 'aime' bucket, split by gold source availability")
    for lab, sel in (("gold-source=gsm8k", [r for r in cov if r[2] == "gsm8k"]),
                     ("no gold match (true AIME)", [r for r in rows if r[1] == "aime" and not r[2]])):
        S = np.asarray([r[4] for r in sel])
        print(f"   {lab:28s} n={len(sel):4d}  mean score {np.round(S.mean(0), 3)}")
    print("== every family, gold-source purity")
    fams = sorted({r[1] for r in rows})
    for f in fams:
        sel = [r for r in rows if r[1] == f]
        c = Counter(r[2] for r in sel if r[2])
        maj = c.most_common(1)[0] if c else ("-", 0)
        print(f"   {f:16s} n={len(sel):4d} covered={sum(1 for r in sel if r[2]):4d} "
              f"majority={maj[0]}({maj[1]})  others={ {k: v for k, v in c.items() if k != maj[0]} }")
    return rows


def rule(rows):
    print("== candidate corrections evaluated on the 'aime' bucket")
    cov = [r for r in rows if r[1] == "aime" and r[2] == "gsm8k"]     # proven NOT aime
    unc = [r for r in rows if r[1] == "aime" and not r[2]]            # presumed AIME
    print(f"   proven-gsm8k {len(cov)}   unmatched(presumed AIME) {len(unc)}")
    for name, fn in CANDS.items():
        fp = sum(1 for r in cov if fn(r[3]))      # wrongly kept as aime
        tp = sum(1 for r in unc if fn(r[3]))      # correctly kept
        print(f"   {name:16s} keeps {tp:3d}/{len(unc)} presumed-AIME, "
              f"wrongly keeps {fp:3d}/{len(cov)} proven-gsm8k")


def corrected_map(texts, rule_name="either"):
    fn = CANDS[rule_name]
    out = []
    for t in texts:
        f = classify_family(t)
        if f == "aime" and not fn(t):
            f = "gsm8k_or_other"
        out.append(f)
    return np.asarray(out)


def effect():
    tr, dv = labdata.load_all()
    Xtr, Xdv, m = _c.load()
    G, Gd = _c.grams(Xtr, Xdv)
    Y, Ydv, Ldv = m["Ytr"], m["Ydv"], m["Ldv"]
    ridge = _c.ridge_dual(G, Gd, Y, np.arange(len(Y)), 10.0)[0]
    base = 0.1 * ridge + 0.9 * Ldv          # the deployed [1] linear ensemble
    for name in ("current", "corrected"):
        ftr = (np.asarray([classify_family(t) for t in tr.texts]) if name == "current"
               else corrected_map(tr.texts))
        fdv = (np.asarray([classify_family(t) for t in dv.texts]) if name == "current"
               else corrected_map(dv.texts))
        fam = np.zeros_like(Ydv)
        for f in set(ftr.tolist()):
            k = ftr == f
            if k.sum() == 0:
                continue
            fam[fdv == f] = Y[k].mean(axis=0)
        cs = [np.corrcoef(fam[:, j], Ydv[:, j])[0, 1] for j in range(3)]
        rm = [float(np.sqrt(np.mean((fam[:, j] - Ydv[:, j]) ** 2))) for j in range(3)]
        print(f"-- family map: {name}  ({len(set(ftr.tolist()))} families) "
              f"family-mean corr {cs[0]:.3f}/{cs[1]:.3f}/{cs[2]:.3f}  "
              f"RMSE {rm[0]:.3f}/{rm[1]:.3f}/{rm[2]:.3f}")
        for w in (0.0, 0.15, 0.3):
            p = (1 - w) * base + w * fam
            f_, s_, per = _c.best_final(p, dv)
            mt = _c.metrics(p, Ydv)
            print(f"     blend w={w:<5g} corr {mt[0][0]:.3f}/{mt[1][0]:.3f}/{mt[2][0]:.3f} "
                  f" final {f_:.4f}  (fast {per['fast']:.4f} bal {per['balanced']:.4f} "
                  f"prem {per['premium']:.4f})")


if __name__ == "__main__":
    from collections import Counter
    what = sys.argv[1] if len(sys.argv) > 1 else "audit"
    if what == "effect":
        effect()
    else:
        rows = audit()
        if what in ("rule", "all"):
            rule(rows)
