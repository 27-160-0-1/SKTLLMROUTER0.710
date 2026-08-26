# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""E45 - gain-axis transforms: pair balance and gain shrinkage."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, DEPLOYED_CFG
import gainlab as G

lab = Lab()
cv, arr = G.stage1(lab)

print("=== gain-channel diagnostics (dev, deployed cfg, premium tier) ===")
ps, pc = lab.compose(arr, DEPLOYED_CFG, "premium")
ts = lab.true_s[arr["idx"]]
for nm, a, b in (("d1 mid-light", 0, 1), ("d2 k1-mid", 1, 2), ("d k1-light", 0, 2)):
    p = ps[:, b] - ps[:, a]; t = ts[:, b] - ts[:, a]
    sl = np.polyfit(p, t, 1)[0]
    print(f"  {nm:14s} corr={np.corrcoef(p,t)[0,1]:+.3f} pred_sd={p.std():.4f} "
          f"true_sd={t.std():.4f} slope(true~pred)={sl:.3f} pred_mean={p.mean():+.4f} true_mean={t.mean():+.4f}")

res = []
res.append(G.evaluate(lab, cv, arr, label="baseline"))
print("--- pair balance a2 (a1=1) ---")
for a2 in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0):
    res.append(G.evaluate(lab, cv, arr, transform=G.pair_scale(1.0, a2), label=f"pair a2={a2}"))
print("--- gain shrink toward family mean ---")
for w in (0.15, 0.3, 0.5, 0.7):
    res.append(G.evaluate(lab, cv, arr, transform=G.gain_shrink(w, w), label=f"shrink both w={w}"))
for w in (0.3, 0.6):
    res.append(G.evaluate(lab, cv, arr, transform=G.gain_shrink(w, 0.0), label=f"shrink d1 w={w}"))
    res.append(G.evaluate(lab, cv, arr, transform=G.gain_shrink(0.0, w), label=f"shrink d2 w={w}"))
print("--- gain expansion toward family mean (negative w = expand) ---")
for w in (-0.3, -0.6):
    res.append(G.evaluate(lab, cv, arr, transform=G.gain_shrink(w, w), label=f"expand both w={w}"))

best = max(res, key=lambda r: r["cv_ev"])
print(f"\nBEST by cvEV: {best['label']}  cvEV={best['cv_ev']:.6f} dev={best['dev']:.6f}")
Path("reports/lab/e45_gainaxis.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
