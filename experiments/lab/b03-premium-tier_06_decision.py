# SPDX-License-Identifier: Apache-2.0
"""b03 step 6: the premium decision axis - what does the k1 spend buy, and how good
is the available 'k1 beats mid' signal?"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P
from sklearn.metrics import roc_auc_score

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")

def waste(a, safety, name):
    idx = a["idx"]; ts = lab.true_s[idx]; tc = lab.true_c[idx]
    ps, pc = lab.compose(a, DEPLOYED_CFG, "premium")
    pick = P.exact_allocate(ps, pc, 4.0, safety)
    r = np.arange(len(idx)); D = tc[:, 0].sum()
    k1 = pick == 2
    d2t = ts[:, 2] - ts[:, 1]
    up = (tc[r, pick] - tc[:, 0]) / D
    print(f"\n--- {name} n={len(idx)} safety={safety} picks={np.bincount(pick,minlength=3).tolist()} "
          f"ratio={tc[r,pick].sum()/D:.3f} score={ts[r,pick].mean():.4f} ---")
    tot_k1_cost = up[k1].sum()
    print(f"  k1 spend = {tot_k1_cost:.3f} budget units ({100*tot_k1_cost/4.0:.1f}% of the cap)"
          f"  score bought = {d2t[k1].sum()/len(idx):+.4f}")
    for nm, sel in (("k1 <= mid (bought nothing)", k1 & (d2t <= 0)),
                    ("k1 <  mid (regression)", k1 & (d2t < 0)),
                    ("mid already 1.0", k1 & (ts[:, 1] >= 1.0)),
                    ("mid == 0 (rescue attempt)", k1 & (ts[:, 1] <= 0)),
                    ("   of which k1 rescued", k1 & (ts[:, 1] <= 0) & (ts[:, 2] > 0))):
        print(f"  {nm:<28} n={int(sel.sum()):4d}  cost={up[sel].sum():.3f} "
              f"({100*up[sel].sum()/4.0:4.1f}% of cap)  dscore={d2t[sel].sum()/len(idx):+.4f}")
    return ps, pc, pick

ps_cv, pc_cv, pick_cv = waste(cv, 0.840, "TRAIN-OOF pool")
ps_dv, pc_dv, pick_dv = waste(arr, 0.840, "DEV held-out")

# ---- quality of the available signals on the DECISION axis -----------------
for nm, a in (("cv(OOF)", cv), ("dev", arr)):
    idx = a["idx"]; ts = lab.true_s[idx]
    ps, pc = lab.compose(a, DEPLOYED_CFG, "premium")
    d2p = ps[:, 2] - ps[:, 1]
    d2t = ts[:, 2] - ts[:, 1]
    y = (d2t > 0).astype(int)
    ylo = (d2t < 0).astype(int)
    print(f"\n[{nm}] corr(d2 pred, d2 true) = {np.corrcoef(d2p, d2t)[0,1]:.3f}"
          f"   corr(s_k1) = {np.corrcoef(ps[:,2], ts[:,2])[0,1]:.3f}")
    print(f"        AUC(d2pred -> k1>mid)   = {roc_auc_score(y, d2p):.3f}   base rate {y.mean():.3f}")
    print(f"        AUC(d2pred -> k1<mid)   = {roc_auc_score(ylo, -d2p):.3f}  base rate {ylo.mean():.3f}")
    # within family
    fam = lab.fam_arr[idx]
    aucs = []
    for f in sorted(set(fam)):
        s = fam == f
        if y[s].min() != y[s].max() and s.sum() > 30:
            aucs.append((f, int(s.sum()), roc_auc_score(y[s], d2p[s]),
                         float(np.corrcoef(d2p[s], d2t[s])[0, 1])))
    for f, n, au, co in aucs:
        print(f"        {f:<18} n={n:4d} AUC={au:.3f} corr={co:+.3f}")
    # the efficiency axis the allocator actually ranks on
    eff_p = d2p / np.maximum(pc[:, 2] - pc[:, 1], 1e-9)
    eff_t = d2t / np.maximum(lab.true_c[idx][:, 2] - lab.true_c[idx][:, 1], 1e-9)
    print(f"        spearman(eff pred, eff true) = "
          f"{np.corrcoef(np.argsort(np.argsort(eff_p)), np.argsort(np.argsort(eff_t)))[0,1]:.3f}")
