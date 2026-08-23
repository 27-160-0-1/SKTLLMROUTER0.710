# SPDX-License-Identifier: Apache-2.0
"""Variance decomposition of the fast tier's realised budget ratio, and the
break-even bust probability of the safety insurance."""
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

print("=== variance share of the realised ratio numerator (multinomial resampling) ===")
print("share_i = e_i^2 / sum_j e_j^2, e = true extra cost of the item the allocator upgraded")
for nm, a in (("TRAIN-OOF", cv0), ("DEV", arr0)):
    ps, pc = lab.compose(a, DEPLOYED_CFG, "fast")
    rows = a["idx"]; tc = lab.true_c[rows]; r = np.arange(len(rows))
    for g in (0.91, 0.96):
        pk = P.exact_allocate(ps, pc, MF, g)
        e = tc[r, pk] - tc[:, 0]
        v = e ** 2
        o = np.argsort(-v)
        tot = v.sum()
        fam = new[rows]
        byfam = {f: v[fam == f].sum() / tot for f in np.unique(fam) if v[fam == f].sum() > 0}
        top = ", ".join(f"{k}={v_*100:.0f}%" for k, v_ in
                        sorted(byfam.items(), key=lambda kv: -kv[1])[:4])
        print(f"  {nm:10s} sf={g:.2f}  top1={v[o[0]]/tot*100:5.1f}%  top3={v[o[:3]].sum()/tot*100:5.1f}%"
              f"  top10={v[o[:10]].sum()/tot*100:5.1f}%  |  by family: {top}")

print("\n=== break-even: when does lowering the fast safety pay? ===")
ps, pc = lab.compose(arr0, DEPLOYED_CFG, "fast")
rows = arr0["idx"]; ts = lab.true_s[rows]; tc = lab.true_c[rows]; r = np.arange(len(rows))
for g in (0.915, 0.94, 0.96, 0.98):
    pk = P.exact_allocate(ps, pc, MF, g)
    print(f"  dev safety {g:.3f}: score={ts[r,pk].mean():.4f} ratio={tc[r,pk].sum()/tc[:,0].sum():.4f}")
base = P.exact_allocate(ps, pc, MF, 0.915)
lo = ts[r, base].mean()
for g in (0.94, 0.96, 0.98):
    pk = P.exact_allocate(ps, pc, MF, g)
    hi = ts[r, pk].mean()
    print(f"  safety {g:.2f} beats safety 0.915 only if its bust probability < "
          f"{(1 - lo / hi) * 100:.2f}%")

print("\n=== bootstrap bust of the two candidate operating points, both row sets ===")
G = np.array([0.915, 0.94, 0.96, 0.98])
for nm, a in (("TRAIN-OOF", cv0), ("DEV", arr0)):
    ps, pc = lab.compose(a, DEPLOYED_CFG, "fast")
    ev, bu, raw = L.fast_ev(lab, ps, pc, a["idx"], G, nboot=600)
    print(f"  {nm:10s} " + "  ".join(f"sf={g:.3f}: bust={b*100:5.1f}% EV={e:.4f}"
                                     for g, b, e in zip(G, bu, ev)))
