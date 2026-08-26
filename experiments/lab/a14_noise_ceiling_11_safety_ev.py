# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a14 step 11 -- EV-optimal safety with the reference allocator, held-out preds."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from labdata import load_split, allocate, TIERS, TIER_MULT, TIER_WEIGHT   # noqa: E402

dv = load_split("dev")
P43 = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
N = len(dv)
SAFE43 = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
E39 = {"fast": 0.98, "balanced": 0.875, "premium": 0.82}


def boot_idx(B, size, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, N, size) for _ in range(B)]


def tier_ev(t, safety, idxs):
    sc, bu = [], 0
    for idx in idxs:
        tc = dv.cost[idx]; ts = dv.score[idx]
        sel = allocate(P43[f"score_{t}"][idx], P43[f"cost_{t}"][idx], tc,
                       TIER_MULT[t], safety)
        ar = np.arange(len(idx))
        ratio = tc[ar, sel].sum() / tc[:, 0].sum()
        ok = ratio <= TIER_MULT[t] + 1e-15
        sc.append(ts[ar, sel].mean() if ok else 0.0)
        bu += 0 if ok else 1
    return float(np.mean(sc)), bu / len(idxs)


def main():
    for size in (880, 1760):
        idxs = boot_idx(400, size, 2026)
        print("=" * 96)
        print(f"EV vs safety, held-out E43 predictions, batch {size}, 400 resamples")
        print("=" * 96)
        best = {}
        for t in TIERS:
            print(f"  {t}")
            rows = []
            for s in np.round(np.arange(0.60, 1.0201, 0.01), 3):
                ev, bu = tier_ev(t, float(s), idxs)
                rows.append((float(s), ev, bu))
            for s, ev, bu in rows:
                if abs(s * 100 % 5) < 1e-6 or abs(s - SAFE43[t]) < 1e-9:
                    mark = " <-- deployed" if abs(s - SAFE43[t]) < 1e-9 else ""
                    print(f"    safety {s:.2f}  EV={ev:.4f}  bust={100*bu:5.1f}%{mark}")
            b = max(rows, key=lambda r: r[1])
            best[t] = b
            print(f"    BEST: safety {b[0]:.2f} EV={b[1]:.4f} bust={100*b[2]:.1f}%")
        tot = sum(TIER_WEIGHT[t] * best[t][1] for t in TIERS)
        dep = sum(TIER_WEIGHT[t] * tier_ev(t, SAFE43[t], idxs)[0] for t in TIERS)
        e39 = sum(TIER_WEIGHT[t] * tier_ev(t, E39[t], idxs)[0] for t in TIERS)
        print(f"  weighted EV: deployed(.98/.87/.85)={dep:.4f}  "
              f"E39-min-regret(.98/.875/.82)={e39:.4f}  "
              f"EV-optimal({'/'.join(f'{best[t][0]:.2f}' for t in TIERS)})={tot:.4f}")
        print(f"  EV gain from re-tuning safety alone: {tot-dep:+.4f}\n")


if __name__ == "__main__":
    main()
