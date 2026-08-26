# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b04 - why the premium realised ratio is bimodal across meta-GBM seeds.

Seeds 11 and 51 land in the two modes (3.80 vs 3.43 light-budgets).  Refit both
and diff the premium picks item by item to see whether the 0.37 gap is spread
over many items or carried by a few very expensive ones - the answer decides
whether seed-averaging can remove it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")
from harness import DEPLOYED_EXP, DEPLOYED_CFG, MULTS

BASE_EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)
SAF = 0.835

lab = lib.MemLab(verbose=False)
fits = {}
for s in (11, 51):
    lab.spec["seeds"] = (s,)
    fits[s] = lab.fit_predict(lab.train_idx, lab.dev_idx, BASE_EXP)
    print(f"seed {s} fitted", flush=True)

idx = lab.dev_idx
tc = lab.true_c[idx]
base = tc[:, 0].sum()
picks = {}
for s, arr in fits.items():
    ps, pc = lab.compose(arr, DEPLOYED_CFG, "premium")
    picks[s] = lab.allocate(ps, pc, MULTS["premium"], SAF)
    r = np.arange(len(idx))
    print(f"seed {s}: ratio {tc[r, picks[s]].sum()/base:.4f}  counts "
          f"{np.bincount(picks[s], minlength=3).tolist()}")

a, b = picks[11], picks[51]
diff = np.where(a != b)[0]
r = np.arange(len(idx))
dcost = (tc[r, a] - tc[r, b]) / base
print(f"\nitems with a different pick: {len(diff)} of {len(idx)}")
o = np.argsort(-np.abs(dcost))
print(f"{'rank':>4} {'item':>5} {'fam':16s} {'pick11':>6} {'pick51':>6} "
      f"{'d cost (light-budgets)':>22} {'cum':>8}")
cum = 0.0
for j, i in enumerate(o[:15]):
    cum += dcost[i]
    print(f"{j+1:4d} {i:5d} {lab.fam_names[idx[i]]:16s} {a[i]:6d} {b[i]:6d} "
          f"{dcost[i]:22.4f} {cum:8.4f}")
print(f"\ntotal ratio gap {dcost.sum():.4f}; top-3 items carry "
      f"{np.sort(np.abs(dcost))[::-1][:3].sum():.4f}, "
      f"top-10 {np.sort(np.abs(dcost))[::-1][:10].sum():.4f}")
big = np.sort(np.abs(dcost))[::-1]
print(f"share of |gap| in the top 1/3/10 items: "
      f"{big[0]/big.sum():.2%} {big[:3].sum()/big.sum():.2%} {big[:10].sum()/big.sum():.2%}")
