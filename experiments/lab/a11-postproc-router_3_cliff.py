# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a11 step 3 - the realised-ratio cliff as a function of the safety scalar,
and the size of the discrete jumps along the Lagrangian path."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, tier_result

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv); idx = np.arange(n)

print("=== realised ratio / score vs safety scalar (deployed predictions, dev) ===")
for t in TIERS:
    ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
    print(f"-- {t} (cap multiplier {TIER_MULT[t]})")
    prev = None
    for s in np.arange(0.80, 1.005, 0.01):
        r = tier_result(ps, pc, dv, t, float(s))
        nk = np.bincount(r["sel"], minlength=3)
        jump = "" if prev is None else f" d_ratio={r['ratio']-prev:+.4f}"
        prev = r["ratio"]
        print(f"   s={s:.2f} score={r['score']:.4f} true_ratio={r['ratio']:.4f} "
              f"{'PASS' if r['passed'] else 'BUST'} sel={nk}{jump}")

print("\n=== how lumpy is the Lagrangian path?  (premium, fine scan of s) ===")
t = "premium"
ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
rows = []
for s in np.arange(0.84, 0.92, 0.0025):
    r = tier_result(ps, pc, dv, t, float(s))
    rows.append((s, r["ratio"], r["score"], int((r["sel"] == 2).sum())))
for s, ra, sc, k1 in rows:
    print(f"   s={s:.4f} true_ratio={ra:.4f} score={sc:.4f} n_k1={k1}")
