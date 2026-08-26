# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a08 probe 1: how much of each prompt is family-constant boilerplate?

Measures, per family:
  - n, char length quantiles
  - longest common prefix / suffix across items of that family
  - fraction of median prompt length that the LCP+LCS accounts for
  - the same at token (word) level
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_all  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402


def lcp(strs):
    if not strs:
        return ""
    s0 = min(strs, key=len)
    for i in range(len(s0)):
        c = s0[i]
        if any(s[i] != c for s in strs):
            return s0[:i]
    return s0


def lcs(strs):
    return lcp([s[::-1] for s in strs])[::-1]


def main():
    tr, dv = load_all()
    fams_tr = [classify_family(t) for t in tr.texts]
    fams_dv = [classify_family(t) for t in dv.texts]
    allfams = sorted(set(fams_tr))
    print(f"{'family':16s} {'n_tr':>5s} {'n_dv':>5s} {'p10':>7s} {'p50':>7s} {'p90':>7s} "
          f"{'lcp':>5s} {'lcs':>5s} {'boiler%':>8s}")
    for f in allfams:
        txt = [t for t, ff in zip(tr.texts, fams_tr) if ff == f]
        ntr = len(txt)
        ndv = sum(1 for ff in fams_dv if ff == f)
        L = np.array([len(t) for t in txt])
        p = lcp(txt)
        s = lcs(txt)
        med = float(np.median(L))
        print(f"{f:16s} {ntr:5d} {ndv:5d} {np.percentile(L,10):7.0f} {med:7.0f} "
              f"{np.percentile(L,90):7.0f} {len(p):5d} {len(s):5d} "
              f"{100*(len(p)+len(s))/med:8.1f}")
    print()
    print("=== per-family prefix/suffix samples ===")
    for f in allfams:
        txt = [t for t, ff in zip(tr.texts, fams_tr) if ff == f]
        p, s = lcp(txt), lcs(txt)
        print(f"--- {f} (n={len(txt)})")
        print(f"    LCP[{len(p)}]: {p[:200]!r}")
        print(f"    LCS[{len(s)}]: {s[-200:]!r}")
        print(f"    sample head: {txt[0][:300]!r}")
        print(f"    sample tail: {txt[0][-200:]!r}")


if __name__ == "__main__":
    main()
