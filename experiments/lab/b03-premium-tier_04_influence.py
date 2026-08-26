# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b03 step 4: exact per-item attribution of the premium budget-ratio variance.

Under iid resampling of 880 items, log R = log(mean n_i) - log(mean d_i) with
n_i = true cost of the selected model, d_i = true light cost.  Delta method:
    var(log R) ~ var_i(phi_i)/880,   phi_i = n_i/nbar - d_i/dbar.
phi is an exact additive per-item attribution, so the variance can be split by
selected model / family / cost decile with no simulation.
"""
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
otok = lab.otok[ci]; ngen = lab.ngen[ci]
MULT = 4.0
ps, pc = lab.compose(cv, DEPLOYED_CFG, "premium")

for safety in (0.84, 0.90):
    pick = P.exact_allocate(ps, pc, MULT, safety)
    r = np.arange(m)
    n_i = tc[r, pick]; d_i = tc[:, 0]
    nbar, dbar = n_i.mean(), d_i.mean()
    phi = n_i / nbar - d_i / dbar
    v = phi.var()
    print(f"\n=== safety {safety:.2f}  picks L/M/K = {np.bincount(pick,minlength=3).tolist()} "
          f"R(point)={n_i.sum()/d_i.sum():.4f} ===")
    print(f"  delta-method sd(log R) on 880 = {np.sqrt(v/880):.4f}")
    # bootstrap check
    smp = np.asarray(lab.samples_for(m, 7, 800, 880))
    pk = P.exact_allocate(ps[smp], pc[smp], MULT, safety)
    Ct = np.take_along_axis(tc[smp], pk[:, :, None], axis=2)[:, :, 0]
    Rb = Ct.sum(axis=1) / tc[smp][:, :, 0].sum(axis=1)
    print(f"  bootstrap    sd(log R)        = {np.log(Rb).std():.4f}  (re-allocated each batch)")
    # attribution by selected model
    print(f"  {'group':<22}{'n':>6}{'share var(phi)':>16}{'mean phi':>10}")
    for j, nm in enumerate(("light", "mid", "k1")):
        s = pick == j
        # contribution of a subgroup = sum over that group of (phi_i - phibar)^2 / n
        c = ((phi[s] - phi.mean()) ** 2).sum() / m / v
        print(f"  sel={nm:<18}{s.sum():6d}{c:16.3f}{phi[s].mean():10.3f}")
    print("  --- within the k1-selected set, by true k1 out/gen decile ---")
    k1s = np.where(pick == 2)[0]
    og = otok[k1s, 2] / ngen[k1s, 2]
    q = np.quantile(og, [0, .5, .8, .9, .95, .99, 1.0])
    lab_names = ["p0-50", "p50-80", "p80-90", "p90-95", "p95-99", "p99-100"]
    for a in range(6):
        s = k1s[(og >= q[a]) & (og <= q[a + 1] if a == 5 else og < q[a + 1])]
        if len(s) == 0:
            continue
        c = ((phi[s] - phi.mean()) ** 2).sum() / m / v
        print(f"  out/gen {lab_names[a]:<14}{len(s):6d}{c:16.3f}"
              f"   med out/gen={np.median(otok[s,2]/ngen[s,2]):8.0f}"
              f"   mean s_k1={ts[s,2].mean():.3f}")
    print("  --- by family (k1-selected only) ---")
    for f in sorted(set(fam[k1s])):
        s = k1s[fam[k1s] == f]
        c = ((phi[s] - phi.mean()) ** 2).sum() / m / v
        print(f"  {f:<22}{len(s):6d}{c:16.3f}")
    # winsorise experiment: cap the true k1 cost of selected items at quantile, keep mean
    print("  --- counterfactual: winsorise the realised k1 cost of the selected set ---")
    for qq in (0.99, 0.95, 0.90, 0.80):
        n2 = n_i.copy()
        k = pick == 2
        cap = np.quantile(n_i[k], qq)
        n2[k] = np.minimum(n_i[k], cap)
        n2[k] *= n_i[k].sum() / n2[k].sum()          # preserve the level exactly
        p2 = n2 / n2.mean() - d_i / dbar
        print(f"  cap at q{qq:<5} sd(log R)={np.sqrt(p2.var()/880):.4f}"
              f"  ({100*(1-np.sqrt(p2.var()/v)):.0f}% variance-sd reduction)")
