# SPDX-License-Identifier: Apache-2.0
"""b02 step 8 - matched-realised-budget comparison (risk dimension removed).

For every policy sweep the safety scalar on a fine grid, record (realised ratio,
realised score) and, for a grid of budget targets, take the best score whose
realised ratio is <= the target.  Two evaluation sets: the Train OOF rows (the
honest selection set) and Dev (held out).  Also re-checks the family blend on a
DEV bootstrap, because its Train-OOF EV and its Dev point disagree.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b02lib", HERE / "b02-balanced-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, MULTS, DEPLOYED_CFG  # noqa
import protocol as P  # noqa

MULT = MULTS["balanced"]
lab = Lab()
cv, arr = L.load_stage("base")
ci = cv["idx"]; di = arr["idx"]; m = len(ci)
tr = lab.train_idx

ps_c, pc_c = lab.compose(cv, DEPLOYED_CFG, "balanced")
ps_d, pc_d = lab.compose(arr, DEPLOYED_CFG, "balanced")


def fam_means(fit_rows):
    g = lab.true_s[fit_rows].mean(axis=0)
    return {f: (lab.true_s[fit_rows[lab.fam_arr[fit_rows] == f]].mean(axis=0)
                if (lab.fam_arr[fit_rows] == f).sum() >= 8 else g) for f in lab.FAMILIES}


fold_of = np.random.default_rng(123).integers(0, 10, size=len(tr))
pos = {int(v): k for k, v in enumerate(ci)}
fam_cv = np.zeros((len(ci), 3))
for f in range(10):
    fm = fam_means(tr[fold_of != f])
    for i in tr[fold_of == f]:
        fam_cv[pos[int(i)]] = fm[lab.fam_arr[i]]
fm_tr = fam_means(tr)
fam_dev = np.array([fm_tr[f] for f in lab.fam_arr[di]])

POLICIES = {}
POLICIES["deployed"] = (ps_c, pc_c, ps_d, pc_d)
for k2 in (1.5,):
    a = pc_c.copy(); a[:, 2] *= k2
    b = pc_d.copy(); b[:, 2] *= k2
    POLICIES[f"kappa2={k2}"] = (ps_c, a, ps_d, b)
for w in (0.25, 0.5, 1.0):
    POLICIES[f"fam blend w={w}"] = (np.clip((1 - w) * ps_c + w * fam_cv, 0, 1), pc_c,
                                    np.clip((1 - w) * ps_d + w * fam_dev, 0, 1), pc_d)
# no-k1 policy (pure "spread on mid")
nk_c = ps_c.copy(); nk_c[:, 2] = nk_c[:, 1]
nk_d = ps_d.copy(); nk_d[:, 2] = nk_d[:, 1]
POLICIES["no k1 at all"] = (nk_c, pc_c, nk_d, pc_d)
# mid-forbidden policy (pure "concentrate on k1")
nm_c = ps_c.copy(); nm_c[:, 1] = nm_c[:, 0]
nm_d = ps_d.copy(); nm_d[:, 1] = nm_d[:, 0]
POLICIES["no mid (k1 only)"] = (nm_c, pc_c, nm_d, pc_d)

SAFETY = np.arange(0.30, 1.601, 0.0025)
TARGETS = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.95, 2.0]


def matched(ps, pc, idx, label):
    ts = lab.true_s[idx]; tc = lab.true_c[idx]
    picks = P.exact_allocate(np.repeat(ps[None], len(SAFETY), axis=0),
                             np.repeat(pc[None], len(SAFETY), axis=0), MULT, 1.0) \
        if False else None
    rows = []
    for s in SAFETY:
        p = lab.allocate(ps, pc, MULT, float(s))
        r = np.arange(len(idx))
        rows.append((tc[r, p].sum() / tc[:, 0].sum(), ts[r, p].mean(),
                     int((p == 1).sum()), int((p == 2).sum())))
    rows = np.array([(a, b, c, d) for a, b, c, d in rows])
    out = []
    for T in TARGETS:
        ok = rows[rows[:, 0] <= T + 1e-12]
        if len(ok) == 0:
            out.append((np.nan, 0, 0)); continue
        j = int(np.argmax(ok[:, 1]))
        out.append((ok[j, 1], int(ok[j, 2]), int(ok[j, 3])))
    return out


for setname, idx, sel in (("TRAIN-OOF", ci, 0), ("DEV", di, 2)):
    print(f"\n=== matched realised budget, {setname} : best score at ratio <= T ===")
    print(f"  {'policy':18s} " + " ".join(f"{T:>15.2f}" for T in TARGETS))
    base = None
    for name, tup in POLICIES.items():
        ps, pc = tup[sel], tup[sel + 1]
        res = matched(ps, pc, idx, name)
        if base is None:
            base = [r[0] for r in res]
            print(f"  {name:18s} " + " ".join(f"{r[0]:.4f}(M{r[1]:3d}K{r[2]:3d})" for r in res))
        else:
            print(f"  {name:18s} " + " ".join(
                f"{r[0]-b:+.4f}(M{r[1]:3d}K{r[2]:3d})" for r, b in zip(res, base)))

print("\n=== family blend judged on a DEV bootstrap (sanity check of the "
      "Train-OOF/Dev disagreement) ===")
GRID = np.arange(0.60, 1.201, 0.005)
md = len(di)
SD = np.concatenate([np.asarray(lab.samples_for(md, s, 400, 880)) for s in (7, 17, 23)])
SC = np.concatenate([np.asarray(lab.samples_for(m, s, 400, 880)) for s in (7, 17, 23)])
print(f"  {'policy':18s} {'TrainOOF EV':>12s} {'DEV EV':>10s} {'DEV raw':>9s} {'DEVbust%':>9s}")
for name, tup in POLICIES.items():
    e1, b1, r1 = P.safety_curve(tup[0][SC], tup[1][SC], lab.true_s[ci][SC], lab.true_c[ci][SC],
                                MULT, GRID)
    e2, b2, r2 = P.safety_curve(tup[2][SD], tup[3][SD], lab.true_s[di][SD], lab.true_c[di][SD],
                                MULT, GRID)
    i1 = int(np.argmax(e1)); i2 = int(np.argmax(e2))
    print(f"  {name:18s} {e1[i1]:12.6f} {e2[i2]:10.6f} {r2[i2]:9.6f} {b2[i2]*100:9.2f}")
