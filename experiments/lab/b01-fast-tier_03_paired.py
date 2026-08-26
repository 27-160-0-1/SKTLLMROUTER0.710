# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Is the item-level vs family-only frontier difference real?  Paired bootstrap.

For every 880-item resample and every policy, the whole safety grid is evaluated
and the reported statistic is "best realised score whose realised ratio <= 1.25".
Both policies see the same resample, so the difference is paired.  Run on the
honest train-OOF rows AND on dev.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b01lib", HERE / "b01-fast-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP  # noqa
import bench2 as B
import protocol as P

lab = Lab(); MF = 1.25
cv0, arr0 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
new = np.array([L.classify_v3(t) for t in lab.texts])
cvm, dvm = L.fam_matrix(lab, new, cv0, arr0)
ps_cv, pc_cv = lab.compose(cv0, DEPLOYED_CFG, "fast")
ps_dv, pc_dv = lab.compose(arr0, DEPLOYED_CFG, "fast")
GRID = np.arange(0.60, 1.201, 0.01)


def frontier_boot(ps_a, ps_b, pc, rows, nboot=600, seeds=(7, 17, 23)):
    ts = lab.true_s[rows]; tc = lab.true_c[rows]; m = len(rows)
    da, db, wins = [], [], []
    for s in seeds:
        smp = np.asarray(lab.samples_for(m, s, nboot, 880))
        ea, ba, ra = P.safety_curve(ps_a[smp], pc[smp], ts[smp], tc[smp], MF, GRID)
        # per-sample statistic needs the per-sample curve: recompute in chunks
        for chunk in np.array_split(smp, 12):
            out = {}
            for k, psx in (("a", ps_a), ("b", ps_b)):
                best = np.full(len(chunk), -1.0)
                for g in GRID:
                    pk = P.exact_allocate(psx[chunk], pc[chunk], MF, float(g))
                    r = np.arange(len(chunk))[:, None]
                    sc = np.take_along_axis(ts[chunk], pk[:, :, None], 2)[:, :, 0].mean(1)
                    rt = (np.take_along_axis(tc[chunk], pk[:, :, None], 2)[:, :, 0].sum(1)
                          / tc[chunk][:, :, 0].sum(1))
                    ok = rt <= MF + 1e-12
                    best = np.where(ok & (sc > best), sc, best)
                out[k] = best
            da.append(out["a"]); db.append(out["b"]); wins.append(out["a"] > out["b"])
    a = np.concatenate(da); b = np.concatenate(db); d = a - b
    return dict(a=a.mean(), b=b.mean(), diff=d.mean(), sd=d.std(),
                p_a_better=float(np.mean(np.concatenate(wins))),
                lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))


for nm, (psA, psB, pc, rows) in (
        ("TRAIN-OOF", (ps_cv, np.clip(cvm[:, :3], 0, 1), pc_cv, cv0["idx"])),
        ("DEV      ", (ps_dv, np.clip(dvm[:, :3], 0, 1), pc_dv, arr0["idx"]))):
    r = frontier_boot(psA, psB, pc, rows, nboot=200)
    print(f"[{nm}] item={r['a']:.4f} family={r['b']:.4f} diff(item-fam)={r['diff']:+.4f} "
          f"sd={r['sd']:.4f} 95%CI[{r['lo']:+.4f},{r['hi']:+.4f}] P(item better)={r['p_a_better']:.3f}",
          flush=True)

# ------------------------------------------------ per-family picks on dev @1.25
print("\n[dev, safety chosen so realised ratio <= 1.25, per family]")
dev = arr0["idx"]; ts = lab.true_s[dev]; tc = lab.true_c[dev]
fam = new[dev]
rows = {}
for nm, psx in (("item", ps_dv), ("family", np.clip(dvm[:, :3], 0, 1))):
    best = None
    for g in GRID:
        pk = P.exact_allocate(psx, pc_dv, MF, float(g))
        r = np.arange(len(dev))
        rt = tc[r, pk].sum() / tc[:, 0].sum()
        sc = ts[r, pk].mean()
        if rt <= MF + 1e-12 and (best is None or sc > best[0]):
            best = (sc, rt, pk, g)
    rows[nm] = best
    print(f"  {nm:7s} score={best[0]:.4f} ratio={best[1]:.4f} safety={best[3]:.2f} "
          f"counts={np.bincount(best[2],minlength=3).tolist()}")
print(f"  {'family':16s} {'n':>4s} {'lightS':>7s} {'itemUP':>7s} {'famUP':>7s} "
      f"{'itemSc':>7s} {'famSc':>7s} {'itemL':>7s} {'famL':>7s}")
r = np.arange(len(dev))
for f in sorted(set(fam)):
    s = fam == f
    pi = rows["item"][2]; pf = rows["family"][2]
    ci = tc[r, pi][s].sum() / tc[:, 0].sum(); cf = tc[r, pf][s].sum() / tc[:, 0].sum()
    print(f"  {f:16s} {s.sum():4d} {ts[s,0].mean():7.3f} {(pi[s]>0).mean():7.2f} "
          f"{(pf[s]>0).mean():7.2f} {ts[r,pi][s].mean():7.3f} {ts[r,pf][s].mean():7.3f} "
          f"{ci:7.3f} {cf:7.3f}")
