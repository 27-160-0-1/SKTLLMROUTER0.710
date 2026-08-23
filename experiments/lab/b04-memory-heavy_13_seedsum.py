# SPDX-License-Identifier: Apache-2.0
"""b04 - summarise the across-fit dispersion measurement into the numbers the
report needs: bust counts, the safety head-room the fitting noise costs, and the
expectation over a random seed draw vs the averaged head."""
import json, sys
from pathlib import Path
import numpy as np

d = json.loads(Path("reports/lab/b04_seeds.json").read_text())
TIERS = ("fast", "balanced", "premium")
MULT = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
SAF = {"deployed": {"fast": 0.98, "balanced": 0.87, "premium": 0.85},
       "R1": {"fast": 0.960, "balanced": 0.855, "premium": 0.835}}

for name in ("deployed", "R1"):
    ps = d["per_seed"][name]
    seeds = list(ps)
    print(f"\n=== safety {name} "
          f"{'/'.join(f'{SAF[name][t]:.3f}' for t in TIERS)} ===")
    for t in TIERS:
        r = np.array([ps[s][t]["ratio"] for s in seeds])
        maxs = SAF[name][t] * MULT[t] / r
        bust = int((r > MULT[t] + 1e-15).sum())
        ens = np.array([d["ens"][name][k][t]["ratio"] for k in d["ens"][name]])
        emax = SAF[name][t] * MULT[t] / ens
        print(f" {t:9s} single: mean {r.mean():.4f} sd {r.std(ddof=1):.4f} "
              f"min {r.min():.3f} max {r.max():.3f} busts {bust}/{len(r)} | "
              f"max-passing safety worst {maxs.min():.4f} mean {maxs.mean():.4f}")
        print(f" {'':9s} ensembles(2,3,5,8,10): ratios {np.round(ens,4).tolist()} "
              f"sd {ens.std(ddof=1):.4f} | max-passing safety {np.round(emax,4).tolist()}")
    fin = np.array([ps[s]["final"] for s in seeds])
    passing = fin[fin > 0.5]
    ensf = np.array([d["ens"][name][k]["final"] for k in d["ens"][name]])
    print(f" FINAL   single draw: E={fin.mean():.4f} sd={fin.std(ddof=1):.4f}  "
          f"passing-only mean={passing.mean() if len(passing) else float('nan'):.4f} "
          f"({len(passing)}/{len(fin)} passed)")
    print(f"         ensembles: {np.round(ensf,4).tolist()}  mean={ensf.mean():.4f}")
    print(f"         gain of averaging over a random single fit: {ensf.mean()-fin.mean():+.4f}")
