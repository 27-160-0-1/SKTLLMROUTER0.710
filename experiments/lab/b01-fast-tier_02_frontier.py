# SPDX-License-Identifier: Apache-2.0
"""Safety-free comparison: the score-vs-realised-budget frontier of each fast policy.

Comparing policies at their own EV-optimal safety confounds "better picks" with
"different risk appetite".  Here every policy is swept over the whole safety grid
and reported as (realised budget ratio -> realised score), on BOTH the honest
train-OOF rows and dev.  A policy dominates iff its curve is above another's at
the same realised ratio.
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

lab = Lab()
MF = 1.25
cv0, arr0 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")

old = lab.fam_arr
new = np.array([L.classify_v3(t) for t in lab.texts])
print(f"[partition] v3 moves {(old != new).sum()}/{len(old)} items")
FM = {"deployed9": L.fam_matrix(lab, old, cv0, arr0),
      "repaired9": L.fam_matrix(lab, new, cv0, arr0)}

ps_cv, pc_cv = lab.compose(cv0, DEPLOYED_CFG, "fast")
ps_dv, pc_dv = lab.compose(arr0, DEPLOYED_CFG, "fast")


def policy(kind, w=1.0, key="repaired9", cost_mode="item"):
    if kind == "item":
        return (ps_cv, pc_cv), (ps_dv, pc_dv)
    cvm, dvm = FM[key]
    out = []
    for M, ps, pc in ((cvm, ps_cv, pc_cv), (dvm, ps_dv, pc_dv)):
        fs = np.clip(M[:, :3], 0.0, 1.0)
        s = (1 - w) * ps + w * fs
        c = pc
        if cost_mode == "fam":
            c = np.exp(M[:, 3:6]).copy()
            c[:, 1] = np.maximum(c[:, 1], c[:, 0] * (1 + 1e-12))
            c[:, 2] = np.maximum(c[:, 2], c[:, 1] * (1 + 1e-12))
        out.append((s, c))
    return out[0], out[1]


GRID = np.arange(0.60, 1.201, 0.005)


def curve(ps, pc, rows):
    ts = lab.true_s[rows]; tc = lab.true_c[rows]
    sc, rt = [], []
    for s in GRID:
        pk = P.exact_allocate(ps, pc, MF, float(s))
        r = np.arange(len(rows))
        sc.append(ts[r, pk].mean()); rt.append(tc[r, pk].sum() / tc[:, 0].sum())
    return np.array(sc), np.array(rt)


TGT = (1.10, 1.15, 1.20, 1.25)


def at_ratio(sc, rt, targets=TGT):
    return [sc[rt <= t + 1e-12].max() if (rt <= t + 1e-12).any() else np.nan for t in targets]


POLS = [("item-level", dict(kind="item")),
        ("family9 deployed regex", dict(kind="fam", key="deployed9")),
        ("family9 repaired regex", dict(kind="fam", key="repaired9")),
        ("family9 rep + family cost", dict(kind="fam", key="repaired9", cost_mode="fam")),
        ("shrink w=0.5 -> repaired9", dict(kind="fam", key="repaired9", w=0.5)),
        ("shrink w=0.75 -> repaired9", dict(kind="fam", key="repaired9", w=0.75)),
        ("shrink w=0.9 -> repaired9", dict(kind="fam", key="repaired9", w=0.9)),
        ]

hdr = "".join(f"{t:>8.2f}" for t in TGT)
print(f"\n{'policy':28s} | TRAIN-OOF score at realised ratio<= | DEV score at realised ratio<=")
print(f"{'':28s} | {hdr}   | {hdr}")
res = {}
for nm, kw in POLS:
    (a, b), (c, d) = policy(**kw)
    s1, r1 = curve(a, b, cv0["idx"]); s2, r2 = curve(c, d, arr0["idx"])
    res[nm] = (s1, r1, s2, r2)
    print(f"{nm:28s} | " + "".join(f"{x:8.4f}" for x in at_ratio(s1, r1)) + "   | "
          + "".join(f"{x:8.4f}" for x in at_ratio(s2, r2)), flush=True)

s1, r1 = curve(lab.true_s[cv0["idx"]], lab.true_c[cv0["idx"]], cv0["idx"])
s2, r2 = curve(lab.true_s[arr0["idx"]], lab.true_c[arr0["idx"]], arr0["idx"])
print(f"{'ORACLE realised':28s} | " + "".join(f"{x:8.4f}" for x in at_ratio(s1, r1)) + "   | "
      + "".join(f"{x:8.4f}" for x in at_ratio(s2, r2)))
p = L.eb_posterior(lab)
s1, r1 = curve(p[cv0["idx"]], lab.true_c[cv0["idx"]], cv0["idx"])
s2, r2 = curve(p[arr0["idx"]], lab.true_c[arr0["idx"]], arr0["idx"])
print(f"{'ORACLE latent-p (realised)':28s} | " + "".join(f"{x:8.4f}" for x in at_ratio(s1, r1))
      + "   | " + "".join(f"{x:8.4f}" for x in at_ratio(s2, r2)))
print(f"{'ALL LIGHT':28s} | " + f"{lab.true_s[cv0['idx']][:,0].mean():8.4f}" * 4 + "   | "
      + f"{lab.true_s[arr0['idx']][:,0].mean():8.4f}" * 4)

np.savez(HERE.parents[1] / "reports/lab/b01_frontier.npz", grid=GRID,
         **{f"{k}|{i}": v for k, tup in res.items() for i, v in enumerate(tup)})

print("\n[safety -> realised ratio] (cap 1.25)   TRAIN-OOF / DEV")
gs = np.arange(0.88, 1.021, 0.02)
print(f"{'policy':28s} " + "".join(f"{g:>15.2f}" for g in gs))
for nm in res:
    s1, r1, s2, r2 = res[nm]
    row = "".join(f"{r1[np.argmin(np.abs(GRID-g))]:7.4f}/{r2[np.argmin(np.abs(GRID-g))]:7.4f}"
                  for g in gs)
    print(f"{nm:28s} " + row)
