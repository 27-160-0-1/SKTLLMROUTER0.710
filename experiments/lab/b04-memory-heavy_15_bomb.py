# SPDX-License-Identifier: Apache-2.0
"""b04 - the heavy tail of single-item k1 cost, in units of the premium budget.

The premium tier's realised budget ratio was found to be bimodal across meta-GBM
fits, with 74 % of the gap carried by ONE item.  This quantifies how many such
items exist and what fraction of the 4.0-light-budget premium cap each can eat.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")

lab = lib.MemLab(verbose=False)
for split, idx in (("train", lab.train_idx), ("dev", lab.dev_idx)):
    tc = lab.true_c[idx]
    base = tc[:, 0].sum()
    share = tc[:, 2] / base            # k1 cost of one item, in light-budgets
    d1 = (tc[:, 1] - tc[:, 0]) / base
    d2 = (tc[:, 2] - tc[:, 1]) / base
    o = np.argsort(-share)
    print(f"\n=== {split} (n={len(idx)}, premium cap = 4.0 light-budgets) ===")
    print(f" k1 cost share of the whole light budget: "
          f"mean {share.mean():.5f}  p50 {np.median(share):.5f}  p99 {np.percentile(share,99):.4f}  "
          f"max {share.max():.4f}")
    print(f" top-10 single items (k1 cost as a fraction of the 4.0 cap):")
    print(f"   {'item':>5} {'family':16s} {'k1/lightbudget':>15} {'% of the 4.0 cap':>17} "
          f"{'d2 upgrade cost':>16} {'true s l/m/k':>18}")
    for i in o[:10]:
        print(f"   {i:5d} {lab.fam_names[idx[i]]:16s} {share[i]:15.4f} {share[i]/4.0*100:16.2f}% "
              f"{d2[i]:16.4f}  {lab.true_s[idx[i]][0]:.2f}/{lab.true_s[idx[i]][1]:.2f}/"
              f"{lab.true_s[idx[i]][2]:.2f}")
    cum = np.cumsum(np.sort(share)[::-1])
    for k in (1, 3, 10, 30):
        print(f"   top-{k:<3d} items hold {cum[k-1]:.3f} light-budgets "
              f"= {cum[k-1]/4.0*100:.1f} % of the premium cap")
    n_big = int((share > 0.05).sum())
    print(f"   items whose k1 cost alone exceeds 1.25 % of the cap (0.05 budgets): {n_big}")
