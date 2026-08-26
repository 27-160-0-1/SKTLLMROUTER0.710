# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Can the runaway mid-cost items be identified from the prompt, on train alone?

The fast tier's whole headroom is 0.25L; a single runaway mid generation costs
up to 6.8% of L.  This script looks for a train-only predicate that flags them.
"""
from __future__ import annotations
import importlib.util, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b01lib", HERE / "b01-fast-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP  # noqa
import bench2 as B

lab = Lab()
new = np.array([L.classify_v3(t) for t in lab.texts])
tr, dv = lab.train_idx, lab.dev_idx

# footprint of a mid upgrade, rescaled to an 880-item batch
def footprint(rows):
    tc = lab.true_c[rows]
    return (tc[:, 1] - tc[:, 0]) / tc[:, 0].sum() * (len(rows) / 880.0)


fp = np.zeros(lab.n)
fp[tr] = footprint(tr); fp[dv] = footprint(dv)

print("=== the 12 worst mid upgrades in TRAIN, and the 6 worst in DEV ===")
for nm, rows in (("train", tr), ("dev", dv)):
    o = rows[np.argsort(-fp[rows])][: 12 if nm == "train" else 6]
    for i in o:
        t = lab.texts[i]
        runs = re.findall(r"\d+", t)
        mx = max((len(x) for x in runs), default=0)
        print(f"  {nm:5s} ep{i:5d} {new[i]:14s} fp={fp[i]*100:6.3f}%L otok_mid={lab.otok[i,1]:7.0f} "
              f"otok_l={lab.otok[i,0]:7.0f} len={len(t):6d} maxdigits={mx:3d} "
              f"| {t[:70].replace(chr(10),' ')!r}")

# ---------------------------------------------------------------- predicates
def feats(i):
    t = lab.texts[i]
    runs = [len(x) for x in re.findall(r"\d+", t)]
    return dict(maxdig=max(runs, default=0), ndig=sum(runs),
                lt=len(t), pow_=("**" in t), digfrac=sum(c.isdigit() for c in t) / max(len(t), 1))


F = [feats(i) for i in range(lab.n)]
maxdig = np.array([f["maxdig"] for f in F])
lt = np.array([f["lt"] for f in F])

print("\n=== TRAIN-only: mid-upgrade footprint by max-digit-run bucket ===")
print(f"{'bucket':14s} {'n tr':>6s} {'p50 %L':>8s} {'p99 %L':>8s} {'max %L':>8s} "
      f"{'sum %L':>8s} {'mean d1':>8s} | {'n dv':>5s} {'max %L':>8s} {'mean d1':>8s}")
buckets = [(0, 4), (5, 8), (9, 12), (13, 99)]
for lo, hi in buckets:
    a = tr[(maxdig[tr] >= lo) & (maxdig[tr] <= hi)]
    b = dv[(maxdig[dv] >= lo) & (maxdig[dv] <= hi)]
    d1a = (lab.true_s[a][:, 1] - lab.true_s[a][:, 0]).mean() if len(a) else np.nan
    d1b = (lab.true_s[b][:, 1] - lab.true_s[b][:, 0]).mean() if len(b) else np.nan
    print(f"digits {lo:2d}-{hi:2d}    {len(a):6d} {np.percentile(fp[a],50)*100:8.4f} "
          f"{np.percentile(fp[a],99)*100:8.3f} {fp[a].max()*100:8.3f} {fp[a].sum()*100:8.2f} "
          f"{d1a:8.4f} | {len(b):5d} {fp[b].max()*100:8.3f} {d1b:8.4f}")

print("\n=== TRAIN-only: same, restricted to the math-like families ===")
mathfam = np.isin(new, ["dmmath", "gsm8k_or_other", "aime"])
for lo, hi in buckets:
    a = tr[(maxdig[tr] >= lo) & (maxdig[tr] <= hi) & mathfam[tr]]
    b = dv[(maxdig[dv] >= lo) & (maxdig[dv] <= hi) & mathfam[dv]]
    if not len(a):
        continue
    d1a = (lab.true_s[a][:, 1] - lab.true_s[a][:, 0]).mean()
    d1b = (lab.true_s[b][:, 1] - lab.true_s[b][:, 0]).mean() if len(b) else np.nan
    print(f"digits {lo:2d}-{hi:2d}    {len(a):6d} {np.percentile(fp[a],50)*100:8.4f} "
          f"{np.percentile(fp[a],99)*100:8.3f} {fp[a].max()*100:8.3f} {fp[a].sum()*100:8.2f} "
          f"{d1a:8.4f} | {len(b):5d} "
          f"{(fp[b].max()*100 if len(b) else np.nan):8.3f} {d1b:8.4f}")

# how much of the tail mass would each predicate remove, train and dev
print("\n=== predicate coverage: share of the top-tail footprint removed ===")
preds = {
    "maxdig>=9": maxdig >= 9,
    "maxdig>=11": maxdig >= 11,
    "maxdig>=13": maxdig >= 13,
    "maxdig>=9 & math": (maxdig >= 9) & mathfam,
    "maxdig>=11 & math": (maxdig >= 11) & mathfam,
    "'**' in text": np.array([f["pow_"] for f in F]),
}
for nm, m in preds.items():
    row = []
    for sn, rows in (("train", tr), ("dev", dv)):
        s = m[rows]
        big = fp[rows] > 0.01                    # >1 %L = 4 % of the fast headroom
        row.append((s.mean() * 100, (s & big).sum(), big.sum(),
                    fp[rows][s].sum() / fp[rows].sum() * 100))
    print(f"  {nm:20s} " + " | ".join(
        f"{sn}: {a:5.1f}% items, catches {b}/{c} monsters, {d:5.1f}% of footprint"
        for (sn, _), (a, b, c, d) in zip((("train", 0), ("dev", 0)), row)))
