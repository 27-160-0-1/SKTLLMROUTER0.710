# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""The fast-tier policy I would ship, measured end to end.

  * k1 banned in the fast tier (measured worth exactly 0.0000 on both row sets)
  * score = item-level, family-only, or a 50/50 shrink toward the repaired
    9-family posterior
  * variance-form cost re-transformation sweep (a09's C4) restricted to fast
  * safety chosen on TRAIN only, by the one-runaway stress criterion of _09
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b01lib", HERE / "b01-fast-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP, DEPLOYED_SAFETY  # noqa
import bench2 as B
import protocol as P

lab = Lab(); MF = 1.25
cv0, arr0 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
new = np.array([L.classify_v3(t) for t in lab.texts])
cvm, dvm = L.fam_matrix(lab, new, cv0, arr0)
SETS = (("TRAIN-OOF", cv0, cvm), ("DEV", arr0, dvm))

# variance-form sigma per model, estimated on TRAIN-OOF residuals only
psc, pcc = lab.compose(cv0, DEPLOYED_CFG, "fast")
SIG = np.array([np.std(np.log(pcc[:, m] / lab.true_c[cv0["idx"]][:, m])) for m in range(3)])
print(f"[train-OOF log-cost residual sd] {SIG.round(4).tolist()}")


def build(a, M, kind, ban_k1=True, kappa=0.0):
    ps, pc = lab.compose(a, DEPLOYED_CFG, "fast")
    fs = np.clip(M[:, :3], 0.0, 1.0)
    if kind == "item":
        s = ps
    elif kind == "family":
        s = fs
    else:
        w = float(kind.split("=")[1]); s = (1 - w) * ps + w * fs
    c = pc * np.exp(kappa * SIG ** 2)[None, :]
    s = s.copy()
    if ban_k1:
        s[:, 2] = -1e9
    return s, c


def row(a, M, kind, grid, **kw):
    rows = a["idx"]; ts = lab.true_s[rows]; tc = lab.true_c[rows]; r = np.arange(len(rows))
    s, c = build(a, M, kind, **kw)
    out = []
    for g in grid:
        pk = P.exact_allocate(s, c, MF, float(g))
        out.append((ts[r, pk].mean(), tc[r, pk].sum() / tc[:, 0].sum()))
    return out


GS = (0.88, 0.90, 0.91, 0.92, 0.94, 0.96, 0.98)
print("\n=== score / realised ratio by safety (k1 banned) ===")
print(f"{'set':10s} {'policy':12s} " + "".join(f"{g:>16.2f}" for g in GS))
for nm, a, M in SETS:
    for kind in ("item", "shrink=0.5", "family"):
        o = row(a, M, kind, GS)
        print(f"{nm:10s} {kind:12s} " + "".join(f"{x:8.4f}/r{y:.3f}" for x, y in o))

print("\n=== variance-form kappa on the fast cost (item-level scores, k1 banned) ===")
for nm, a, M in SETS:
    for k in (0.0, 0.25, 0.5, 1.0):
        o = row(a, M, "item", GS, kappa=k)
        print(f"{nm:10s} kappa={k:<5} " + "".join(f"{x:8.4f}/r{y:.3f}" for x, y in o))

# ------------------------------------------------------ stress-optimal safety
print("\n=== TRAIN-only stress criterion: EV with one 6.5%L runaway injected ===")


def stress_ev(a, kind, M, grid, z=0.065, seeds=(7, 17, 23), nboot=300, kappa=0.0):
    rows = a["idx"]; ts = lab.true_s[rows]; tc = lab.true_c[rows]; m = len(rows)
    s, c = build(a, M, kind, kappa=kappa)
    ev = np.zeros(len(grid)); bu = np.zeros(len(grid))
    for sd in seeds:
        smp = np.asarray(lab.samples_for(m, sd, nboot, 880))
        TC = tc[smp].copy()
        if z > 0:
            fam = new[rows][smp]
            mathy = np.isin(fam, ["dmmath", "gsm8k_or_other", "code", "aime"])
            base = TC[:, :, 0].sum(axis=1)
            pen = np.where(mathy, c[smp][:, :, 1] - c[smp][:, :, 0], 1e18)
            j = np.argmin(pen, axis=1); bi = np.arange(len(j))
            TC[bi, j, 1] = TC[bi, j, 0] + z * base
        e, b, _r = P.safety_curve(s[smp], c[smp], ts[smp], TC, MF, grid)
        ev += e / len(seeds); bu += b / len(seeds)
    return ev, bu


G2 = np.arange(0.86, 1.001, 0.005)
best = {}
for kind in ("item", "shrink=0.5", "family"):
    ev, bu = stress_ev(cv0, kind, cvm, G2)
    ev0, bu0 = stress_ev(cv0, kind, cvm, G2, z=0.0)
    i = int(np.argmax(ev)); i0 = int(np.argmax(ev0))
    best[kind] = float(G2[i])
    print(f"  {kind:12s} stress-optimal safety={G2[i]:.3f} (EV {ev[i]:.4f}, bust {bu[i]*100:.1f}%) | "
          f"unstressed optimum {G2[i0]:.3f} (EV {ev0[i0]:.4f}); "
          f"unstressed EV at the stress optimum = {ev0[i]:.4f}")

print("\n=== end-to-end bench2 (fast policy applied, balanced/premium untouched) ===")
base = B.run(lab, cv0, arr0, DEPLOYED_CFG, label="baseline legacy-OOF", nboot=400)
FIX = dict(base["safety"])


def mk(kind, ban=True, kappa=0.0):
    def tf(lab_, arr, ps, pc, tier):
        if tier != "fast":
            return ps, pc
        M = cvm if len(arr["idx"]) == len(cvm) else dvm
        return build(arr, M, kind, ban_k1=ban, kappa=kappa)
    return tf


for kind in ("item", "shrink=0.5", "family"):
    for sf in (best[kind], 0.91):
        fs = dict(FIX, fast=sf)
        r = B.run(lab, cv0, arr0, DEPLOYED_CFG, transform=mk(kind), fixed_safety=fs,
                  label=f"fast={kind} k1ban @sf{sf:.3f}", nboot=400)
print("\n(reference) deployed safety triple:")
B.run(lab, cv0, arr0, DEPLOYED_CFG, fixed_safety=DEPLOYED_SAFETY, label="baseline @E43 .98/.87/.85",
      nboot=400)
