# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Are the TRAIN rows intrinsically harder, or is the CV model just weaker?

Same 880-row fit set, two hold-out row sets (the other train half, and dev).
If AUC(d1) is equally low on both, the CV/dev disagreement is a row-set property;
if it is low only on the train half, it is a fit-size / fold artefact.
"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b01lib", HERE / "b01-fast-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP  # noqa
import protocol as P

lab = Lab()
EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)
new = np.array([L.classify_v3(t) for t in lab.texts])


def auc(x, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(x); n1 = y.sum(); n0 = (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def wf_auc(x, y, fam):
    num = den = 0.0
    for f in np.unique(fam):
        s = fam == f
        yy = y[s]
        if yy.all() or (~yy).all():
            continue
        w = yy.sum() * (~yy).sum()
        num += auc(x[s], yy) * w; den += w
    return num / den if den else np.nan


def report(tag, arr):
    rows = arr["idx"]
    ps, pc = lab.compose(arr, DEPLOYED_CFG, "fast")
    ts = lab.true_s[rows]
    d1 = ps[:, 1] - ps[:, 0]
    y1 = ts[:, 1] > ts[:, 0]
    fam = new[rows]
    # frontier value at ratio<=1.25, item vs family-constant
    tc = lab.true_c[rows]
    tg = lab.targets
    fitm = arr.get("_fit")
    fs = np.zeros((len(rows), 3))
    gm = tg[fitm].mean(axis=0)
    for f in np.unique(new[fitm]):
        sel = fitm[new[fitm] == f]
        m = tg[sel].mean(axis=0) if len(sel) >= 8 else gm
        fs[new[rows] == f] = np.clip(m[:3], 0, 1)
    fs[np.all(fs == 0, axis=1)] = np.clip(gm[:3], 0, 1)
    best = {}
    for nm, psx in (("item", ps), ("fam", fs)):
        b = -1
        for g in np.arange(0.60, 1.201, 0.01):
            pk = P.exact_allocate(psx, pc, 1.25, float(g))
            r = np.arange(len(rows))
            if tc[r, pk].sum() / tc[:, 0].sum() <= 1.25 + 1e-12:
                b = max(b, ts[r, pk].mean())
        best[nm] = b
    print(f"{tag:34s} n={len(rows):5d} AUCd1_wf={wf_auc(d1,y1,fam):.4f} "
          f"AUCd1_pool={auc(d1,y1):.4f} corr_sl={np.corrcoef(ps[:,0],ts[:,0])[0,1]:.4f} "
          f"front_item={best['item']:.4f} front_fam={best['fam']:.4f} "
          f"diff={best['item']-best['fam']:+.4f}", flush=True)


tr = lab.train_idx
for rep in range(3):
    rng = np.random.default_rng(100 + rep)
    perm = rng.permutation(tr)
    A, Bh = perm[:880], perm[880:]
    t0 = time.perf_counter()
    a1 = lab.fit_predict(A, Bh, EXP); a1["_fit"] = A
    a2 = lab.fit_predict(A, lab.dev_idx, EXP); a2["_fit"] = A
    report(f"seed{rep} fit=train880 -> train880", a1)
    report(f"seed{rep} fit=train880 -> dev880  ", a2)
    print(f"   ({time.perf_counter()-t0:.0f}s)", flush=True)

# full-train model, both row sets (dev is the only honest hold-out here)
a3 = lab.fit_predict(tr, lab.dev_idx, EXP); a3["_fit"] = tr
report("fit=train1760 -> dev880", a3)
