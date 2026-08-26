# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b03 step 3: level-matched variance decomposition (flattening preserves the family sum)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
ci = cv["idx"]; ts = lab.true_s[ci]; tc = lab.true_c[ci]; m = len(ci)
fam = lab.fam_arr[ci]
MULT = 4.0
ps, pc = lab.compose(cv, DEPLOYED_CFG, "premium")

def flatten(cols):
    """Replace true cost of the given model columns by the family MEAN (level preserved)."""
    out = tc.copy()
    for f in sorted(set(fam)):
        s = fam == f
        for j in cols:
            out[s, j] = tc[s, j].mean()
    return out

VAR = {"none": tc, "k1": flatten([2]), "mid": flatten([1]), "light": flatten([0]),
       "all": flatten([0, 1, 2])}

def run(safety, nboot=1200, seeds=(7, 17, 23)):
    keep = {}
    for s in seeds:
        smp = np.asarray(lab.samples_for(m, s, nboot, 880))
        picks = P.exact_allocate(ps[smp], pc[smp], MULT, safety)
        for name, T in VAR.items():
            Ct = np.take_along_axis(T[smp], picks[:, :, None], axis=2)[:, :, 0]
            C0 = T[smp][:, :, 0]
            keep.setdefault(name, []).append(Ct.sum(axis=1) / C0.sum(axis=1))
        # denominator-only: true numerator, pool-constant denominator
        Ct = np.take_along_axis(tc[smp], picks[:, :, None], axis=2)[:, :, 0]
        keep.setdefault("Dfix", []).append(Ct.sum(axis=1) / tc[smp][:, :, 0].sum(axis=1).mean())
        keep.setdefault("nk1", []).append((picks == 2).sum(axis=1))
        keep.setdefault("sc", []).append(
            np.take_along_axis(ts[smp], picks[:, :, None], axis=2)[:, :, 0].mean(axis=1))
    K = {k: np.concatenate(v) for k, v in keep.items()}
    R = K["none"]
    print(f"\n--- premium safety={safety:.3f}  meanR={R.mean():.4f} sdR={R.std():.4f} "
          f"bust%={100*np.mean(R>MULT):.2f} ---")
    print(f"{'variant':<28}{'meanR':>8}{'sd(R-1)/mean':>14}{'relvar share':>14}{'bust%':>8}")
    base_cv = (R - 1).std() / (R - 1).mean()
    for name in ("none", "k1", "mid", "light", "all", "Dfix"):
        v = K[name]
        cvv = (v - 1).std() / (v - 1).mean()
        print(f"{name:<28}{v.mean():8.4f}{cvv:14.4f}{(cvv/base_cv)**2:14.3f}"
              f"{100*np.mean(v>MULT):8.2f}")
    nk1 = K["nk1"]
    print(f"n_k1 mean={nk1.mean():.1f} sd={nk1.std():.1f} corr(n_k1,R)={np.corrcoef(nk1,R)[0,1]:+.3f}")
    return K

for s in (0.84, 0.90):
    run(s)

# --- how concentrated is the k1 upgrade cost?  which items carry the tail? ---
smp = np.asarray(lab.samples_for(m, 7, 400, 880))
picks = P.exact_allocate(ps[smp], pc[smp], MULT, 0.84)
Ct = np.take_along_axis(tc[smp], picks[:, :, None], axis=2)[:, :, 0]
C0 = tc[smp][:, :, 0]
up = (Ct - C0) / C0.sum(axis=1)[:, None]          # upgrade cost in budget units
isk1 = picks == 2
print("\nupgrade cost in budget units (per bootstrap batch):")
print(f"  total U/D           mean={up.sum(axis=1).mean():.4f}")
print(f"  from k1 picks       mean={np.where(isk1, up, 0).sum(axis=1).mean():.4f}"
      f"  sd={np.where(isk1, up, 0).sum(axis=1).std():.4f}")
print(f"  from mid picks      mean={np.where(picks==1, up, 0).sum(axis=1).mean():.4f}"
      f"  sd={np.where(picks==1, up, 0).sum(axis=1).std():.4f}")
srt = np.sort(np.where(isk1, up, 0), axis=1)[:, ::-1]
tot = np.where(isk1, up, 0).sum(axis=1)
for k in (1, 3, 5, 10, 25):
    print(f"  top-{k:<3d} k1 items carry {100*(srt[:, :k].sum(axis=1)/tot).mean():5.1f}% of k1 upgrade cost")
