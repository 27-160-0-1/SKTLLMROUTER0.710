# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b02 step 7 - family-posterior policy, mid-ranking anatomy, k1 caps, and the
candidate balanced policy.
"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b02lib", HERE / "b02-balanced-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, MULTS, TIERS, W, DEPLOYED_CFG  # noqa
import protocol as P  # noqa

MULT = MULTS["balanced"]
GRID = np.arange(0.60, 1.201, 0.005)
lab = Lab()
cv, arr = L.load_stage("base")
ci = cv["idx"]; di = arr["idx"]; m = len(ci)
ALL = np.concatenate([np.asarray(lab.samples_for(m, s, 400, 880)) for s in (7, 17, 23)])
eb = np.load("reports/lab/b02_eb.npz")["eb"]


def ev_of(ps, pc, grid=GRID):
    ev, bu, raw = P.safety_curve(ps[ALL], pc[ALL], lab.true_s[ci][ALL], lab.true_c[ci][ALL],
                                 MULT, grid)
    gi = int(np.argmax(ev))
    return float(ev[gi]), float(grid[gi]), float(bu[gi])


def dev_of(ps, pc, safety):
    pick = lab.allocate(ps, pc, MULT, safety)
    r = np.arange(len(di))
    return (float(lab.true_s[di][r, pick].mean()),
            float(lab.true_c[di][r, pick].sum() / lab.true_c[di][:, 0].sum()),
            np.bincount(pick, minlength=3).tolist())


ps0c, pc0c = lab.compose(cv, DEPLOYED_CFG, "balanced")
ps0d, pc0d = lab.compose(arr, DEPLOYED_CFG, "balanced")
e0, s0, b0 = ev_of(ps0c, pc0c)
d0 = dev_of(ps0d, pc0d, s0)
print(f"baseline  EV={e0:.6f}@s{s0:.3f} bust={b0*100:.2f}%  dev={d0[0]:.6f} r={d0[1]:.3f} {d0[2]}")

# ---------------------------------------------------------------- (A) family
print("\n=== (A) family-posterior score policies (train-only family means) ===")
tr = lab.train_idx


def fam_means(fit_rows):
    g = lab.true_s[fit_rows].mean(axis=0)
    out = {}
    for f in lab.FAMILIES:
        rr = fit_rows[lab.fam_arr[fit_rows] == f]
        out[f] = lab.true_s[rr].mean(axis=0) if len(rr) >= 8 else g
    return out


# honest: for the OOF rows use 10-fold family means; for dev use train means
fold_of = np.random.default_rng(123).integers(0, 10, size=len(tr))
fam_cv = np.zeros((len(ci), 3))
pos = {int(v): k for k, v in enumerate(ci)}
for f in range(10):
    fm = fam_means(tr[fold_of != f])
    for i in tr[fold_of == f]:
        fam_cv[pos[int(i)]] = fm[lab.fam_arr[i]]
fm_tr = fam_means(tr)
fam_dev = np.array([fm_tr[f] for f in lab.fam_arr[di]])

for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    psc = np.clip((1 - w) * ps0c + w * fam_cv, 0, 1)
    psd = np.clip((1 - w) * ps0d + w * fam_dev, 0, 1)
    e, s, b = ev_of(psc, pc0c)
    d = dev_of(psd, pc0d, s)
    print(f"  score = (1-w)*deployed + w*family, w={w:4.2f}: EV={e:.6f} ({e-e0:+.4f}) "
          f"@s{s:.3f} bust={b*100:.2f}%  dev={d[0]:.6f} ({d[0]-d0[0]:+.4f}) r={d[1]:.3f} {d[2]}")

# ------------------------------------------------------- (B) mid-rank anatomy
print("\n=== (B) anatomy of the mid-upgrade ranking (OOF rows) ===")
tcv = lab.true_c[ci]
for tier in TIERS:
    ps, pc = lab.compose(cv, DEPLOYED_CFG, tier)
    d1s = ps[:, 1] - ps[:, 0]; d1c = pc[:, 1] - pc[:, 0]
    t1c = tcv[:, 1] - tcv[:, 0]
    e_pred = d1s / np.maximum(d1c, 1e-12)
    e_flat = d1s / np.maximum(pc[:, 0] * (pc[:, 1] / pc[:, 0]).mean() - pc[:, 0], 1e-12)
    e_true = d1s / np.maximum(t1c, 1e-12)
    print(f"  {tier:9s} sd(log d1s|>0)={np.log(np.maximum(d1s,1e-9))[d1s>0].std():.3f} "
          f"sd(log d1c)={np.log(np.maximum(d1c,1e-12)).std():.3f} "
          f"sd(log c1/c0 true)={np.log(tcv[:,1]/tcv[:,0]).std():.3f} "
          f"sd(log c2/c0 true)={np.log(tcv[:,2]/tcv[:,0]).std():.3f}")
    print(f"            spearman(e_pred, e_true)={spearmanr(e_pred, e_true).statistic:.3f} "
          f"spearman(e_pred, e_flat)={spearmanr(e_pred, e_flat).statistic:.3f} "
          f"spearman(e_flat, e_true)={spearmanr(e_flat, e_true).statistic:.3f}")

print("\n  cost-blind variants at BALANCED (mid cost = global mean multiple of light):")
for name, pcv, pcd in [
    ("deployed pred cost", pc0c, pc0d),
    ("flat mid mult", None, None),
]:
    if pcv is None:
        r1 = (lab.true_c[tr][:, 1] / lab.true_c[tr][:, 0]).mean()
        r2 = (lab.true_c[tr][:, 2] / lab.true_c[tr][:, 0]).mean()
        pcv = np.column_stack([pc0c[:, 0], pc0c[:, 0] * r1, pc0c[:, 0] * r2])
        pcd = np.column_stack([pc0d[:, 0], pc0d[:, 0] * r1, pc0d[:, 0] * r2])
        name += f" ({r1:.2f}/{r2:.1f})"
    e, s, b = ev_of(ps0c, pcv)
    d = dev_of(ps0d, pcd, s)
    print(f"   {name:34s} EV={e:.6f} ({e-e0:+.4f}) @s{s:.3f} bust={b*100:.2f}% "
          f"dev={d[0]:.6f} ({d[0]-d0[0]:+.4f}) {d[2]}")

# ------------------------------------------------------------ (C) k1 controls
print("\n=== (C) explicit k1 controls at BALANCED ===")


def cap_k1(ps, pc, nmax):
    """Forbid k1 for all but the nmax best chord-efficiency items."""
    e = (ps[:, 2] - ps[:, 0]) / np.maximum(pc[:, 2] - pc[:, 0], 1e-12)
    keep = np.argsort(-e)[:nmax]
    ps = ps.copy()
    mask = np.ones(len(ps), bool); mask[keep] = False
    ps[mask, 2] = ps[mask, 1]           # k1 gain zeroed -> never bought
    return ps


for frac in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 1.0):
    nc = int(round(frac * m)); nd = int(round(frac * len(di)))
    psc = cap_k1(ps0c, pc0c, nc); psd = cap_k1(ps0d, pc0d, nd)
    e, s, b = ev_of(psc, pc0c)
    d = dev_of(psd, pc0d, s)
    print(f"  k1 allowed for top {frac*100:5.1f}% : EV={e:.6f} ({e-e0:+.4f}) @s{s:.3f} "
          f"bust={b*100:.2f}% dev={d[0]:.6f} ({d[0]-d0[0]:+.4f}) r={d[1]:.3f} {d[2]}")

print("\n  veto by predicted k1 price ratio c2/c0 (drop the most expensive q):")
for q in (1.0, 0.95, 0.9, 0.8, 0.6):
    def veto(ps, pc):
        ratio = pc[:, 2] / pc[:, 0]
        thr = np.quantile(ratio, q)
        ps = ps.copy(); bad = ratio > thr
        ps[bad, 2] = ps[bad, 1]
        return ps
    psc = veto(ps0c, pc0c); psd = veto(ps0d, pc0d)
    e, s, b = ev_of(psc, pc0c)
    d = dev_of(psd, pc0d, s)
    print(f"   keep q<={q:4.2f}: EV={e:.6f} ({e-e0:+.4f}) @s{s:.3f} bust={b*100:.2f}% "
          f"dev={d[0]:.6f} ({d[0]-d0[0]:+.4f}) {d[2]}")

# ------------------------------------------------------------ (D) candidates
print("\n=== (D) candidate balanced policies ===")
CANDS = [
    ("deployed", dict()),
    ("kappa2=1.35", dict(k2=1.35)),
    ("kappa2=1.50", dict(k2=1.50)),
    ("kappa2=1.50 + fam w=.25", dict(k2=1.50, w=0.25)),
    ("kappa2=1.50 + k1cap 3%", dict(k2=1.50, cap=0.03)),
    ("kappa2=1.50 + blend_bal 1.0", dict(k2=1.50, blend=1.0)),
]
for name, kw in CANDS:
    cfg = dict(DEPLOYED_CFG)
    if "blend" in kw:
        cfg["blend_balanced"] = kw["blend"]
    psc, pcc = lab.compose(cv, cfg, "balanced")
    psd, pcd = lab.compose(arr, cfg, "balanced")
    if "w" in kw:
        psc = np.clip((1 - kw["w"]) * psc + kw["w"] * fam_cv, 0, 1)
        psd = np.clip((1 - kw["w"]) * psd + kw["w"] * fam_dev, 0, 1)
    if "k2" in kw:
        pcc = pcc.copy(); pcc[:, 2] *= kw["k2"]
        pcd = pcd.copy(); pcd[:, 2] *= kw["k2"]
    if "cap" in kw:
        psc = cap_k1(psc, pcc, int(round(kw["cap"] * m)))
        psd = cap_k1(psd, pcd, int(round(kw["cap"] * len(di))))
    e, s, b = ev_of(psc, pcc)
    d = dev_of(psd, pcd, s)
    print(f"  {name:28s} EV={e:.6f} ({e-e0:+.4f}) @s{s:.3f} bust={b*100:.2f}% "
          f"dev={d[0]:.6f} ({d[0]-d0[0]:+.4f}) r={d[1]:.3f} L/M/K={d[2]}")
