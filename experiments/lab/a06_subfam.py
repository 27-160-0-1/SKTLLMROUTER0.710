# SPDX-License-Identifier: Apache-2.0
"""a06 step 2: are the strong within-family signals actually sub-family mixtures?

Splits `aime` by n_qmark and `gsm8k_or_other` by frac_digit and prints the
per-group score/cost profile plus example prompt heads, on train AND dev.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a06_features import build  # noqa: E402

MN = ("light", "mid", "k1")


def profile(tag, d, mask):
    sp = d["split"]
    if mask.sum() == 0:
        print(f"  {tag:34s} n=0")
        return
    s = sp.score[mask].mean(0)
    c = sp.cost[mask].mean(0) / sp.cost[:, 0].mean()
    ot = (sp.otok[mask] / sp.ngen[mask]).mean(0)
    print(f"  {tag:34s} n={mask.sum():4d} s={np.round(s,3)} costratio={np.round(c,1)} "
          f"otok/gen={np.round(ot,0)} ngen={sp.ngen[mask,0].mean():.1f}")


def show(d, name, mask, k=3, width=150):
    idx = np.where(mask)[0][:k]
    for i in idx:
        t = d["split"].texts[i].replace("\n", " | ")
        print(f"      [{name}] {t[:width]}")


def main():
    for nm in ("train", "dev"):
        d = build(nm)
        fam, X, names = d["fam"], d["X"], d["names"]
        iq = names.index("n_qmark")
        idg = names.index("frac_digit")
        ilat = names.index("latex_hits")
        print(f"\n===== {nm} =====")

        m = fam == "aime"
        print(" AIME split by n_qmark:")
        for lo, hi, tag in ((0, 0, "n_qmark==0"), (1, 1, "n_qmark==1"), (2, 99, "n_qmark>=2")):
            mm = m & (X[:, iq] >= lo) & (X[:, iq] <= hi)
            profile(tag, d, mm)
        if nm == "train":
            show(d, "q=0", m & (X[:, iq] == 0))
            show(d, "q>=1", m & (X[:, iq] >= 1))

        g = fam == "gsm8k_or_other"
        print(" gsm8k_or_other split by frac_digit tertile (train-fixed cuts .03/.08):")
        for lo, hi, tag in ((0.0, 0.03, "frac_digit<.03"), (0.03, 0.08, ".03-.08"),
                            (0.08, 9.9, ">=.08")):
            mm = g & (X[:, idg] >= lo) & (X[:, idg] < hi)
            profile(tag, d, mm)
        if nm == "train":
            show(d, "digit<.03", g & (X[:, idg] < 0.03))
            show(d, "digit>=.08", g & (X[:, idg] >= 0.08))

        print(" gsm8k_or_other split by latex_hits>0 (i.e. $...$ present):")
        for cond, tag in (((X[:, ilat] == 0), "latex=0"), ((X[:, ilat] > 0), "latex>0")):
            profile(tag, d, g & cond)
        print(" gsm8k_or_other split by num_generations:")
        for v in (2, 4):
            profile(f"ngen=={v}", d, g & (d["split"].ngen[:, 0] == v))
        print(" AIME split by num_generations:")
        for v in (2, 4):
            profile(f"ngen=={v}", d, m & (d["split"].ngen[:, 0] == v))


if __name__ == "__main__":
    main()
