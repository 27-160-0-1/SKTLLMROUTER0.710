# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""The fast tier's binding constraint is the realised budget ratio.  Decompose it.

  ratio - 1 = (sum of TRUE extra cost over the upgraded items) / (TRUE light total)

and the allocator only controls the PREDICTED version of both terms, so

  ratio - 1 = (0.25*safety_eff) * R_extra * R_light
     R_light = pred light total / true light total
     R_extra = true extra / pred extra, over the items actually upgraded.

Everything is measured on the honest train-OOF rows and on dev at matched safety.
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
SETS = {"TRAIN-OOF": cv0, "DEV": arr0}

print("=== 1. ratio decomposition at matched safety ===")
print(f"{'set':10s} {'sfty':>5s} {'ratio':>7s} {'pred ratio':>10s} {'R_light':>8s} "
      f"{'R_extra':>8s} {'n up':>5s} {'true extra/L':>12s} {'pred extra/Lp':>13s}")
DEC = {}
for g in (0.92, 0.96, 0.98):
    for nm, a in SETS.items():
        rows = a["idx"]; ps, pc = lab.compose(a, DEPLOYED_CFG, "fast")
        tc = lab.true_c[rows]
        pk = P.exact_allocate(ps, pc, MF, g)
        r = np.arange(len(rows))
        Ct = tc[:, 0].sum(); Cp = pc[:, 0].sum()
        te = tc[r, pk] - tc[:, 0]
        pe = pc[r, pk] - pc[:, 0]
        ratio = tc[r, pk].sum() / Ct
        pratio = pc[r, pk].sum() / Cp
        DEC[(nm, g)] = (pk, te, pe, Ct, Cp)
        print(f"{nm:10s} {g:5.2f} {ratio:7.4f} {pratio:10.4f} {Cp/Ct:8.4f} "
              f"{te.sum()/max(pe.sum(),1e-12):8.4f} {(pk>0).sum():5d} "
              f"{te.sum()/Ct:12.4f} {pe.sum()/Cp:13.4f}")

print("\n=== 2. where the extra-cost under-prediction lives (safety 0.96) ===")
for nm in SETS:
    pk, te, pe, Ct, Cp = DEC[(nm, 0.96)]
    rows = SETS[nm]["idx"]; fam = new[rows]
    print(f"-- {nm}: total true extra {te.sum()/Ct:.4f}L  pred extra {pe.sum()/Ct:.4f}L(true-L units)"
          f"  gap {(te.sum()-pe.sum())/Ct:+.4f}L")
    print(f"   {'family':16s} {'n up':>5s} {'true/L':>8s} {'pred/L':>8s} {'gap/L':>8s} {'x':>6s}")
    tot = []
    for f in sorted(set(fam)):
        s = (fam == f) & (pk > 0)
        if s.sum() == 0:
            continue
        tot.append((te[s].sum() - pe[s].sum()) / Ct)
        print(f"   {f:16s} {s.sum():5d} {te[s].sum()/Ct:8.4f} {pe[s].sum()/Ct:8.4f} "
              f"{(te[s].sum()-pe[s].sum())/Ct:+8.4f} {te[s].sum()/max(pe[s].sum(),1e-12):6.2f}")

print("\n=== 3. item concentration of the gap (safety 0.96) ===")
for nm in SETS:
    pk, te, pe, Ct, Cp = DEC[(nm, 0.96)]
    rows = SETS[nm]["idx"]
    gap = (te - pe) / Ct
    o = np.argsort(-gap)
    hd = 0.25 - te.sum() / Ct    # remaining headroom in L units
    print(f"-- {nm}: headroom left {hd:+.4f}L ; top-1 {gap[o[0]]:+.4f}L "
          f"top-3 {gap[o[:3]].sum():+.4f}L top-10 {gap[o[:10]].sum():+.4f}L "
          f"(share of positive gap: {gap[o[:10]].sum()/gap[gap>0].sum()*100:.0f}%)")
    for k in o[:6]:
        i = rows[k]
        print(f"     ep={i:5d} fam={new[i]:14s} pick={pk[k]} true_extra={te[k]/Ct*100:6.3f}%L "
              f"pred_extra={pe[k]/Ct*100:6.3f}%L  x{te[k]/max(pe[k],1e-12):7.1f} "
              f"otok_mid={lab.otok[i,1]:8.0f} len={len(lab.texts[i]):6d}")

print("\n=== 4. bootstrap bust curve, both row sets ===")
GRID = np.arange(0.86, 1.021, 0.01)
for nm, a in SETS.items():
    rows = a["idx"]; ps, pc = lab.compose(a, DEPLOYED_CFG, "fast")
    ev, bu, raw = L.fast_ev(lab, ps, pc, rows, GRID, nboot=400)
    print(f"-- {nm}")
    print("   safety " + "".join(f"{g:7.2f}" for g in GRID))
    print("   bust%  " + "".join(f"{b*100:7.1f}" for b in bu))
    print("   EV     " + "".join(f"{e:7.4f}" for e in ev))
    print("   raw    " + "".join(f"{e:7.4f}" for e in raw))
