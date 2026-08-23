# SPDX-License-Identifier: Apache-2.0
"""Why do TRAIN-OOF and DEV disagree about item-level vs family-only?

Measures the decision-relevant statistic (within-family AUC of the mid-light gain,
a05's P1) and the cost-side calibration separately on the two row sets.
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

lab = Lab()
cv0, arr0 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
new = np.array([L.classify_v3(t) for t in lab.texts])
p = L.eb_posterior(lab)


def auc(x, y):
    """AUC of score x for binary y; ties get 0.5."""
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan, 0
    from scipy.stats import rankdata
    r = rankdata(x)
    n1 = y.sum(); n0 = (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), int(min(n0, n1))


def pooled_within_family_auc(x, lab_bin, fam):
    """Concordant-pair AUC pooled over families (a05's within-family statistic)."""
    num = den = 0.0
    for f in np.unique(fam):
        s = fam == f
        yy = lab_bin[s]
        if yy.all() or (~yy).all():
            continue
        a, _ = auc(x[s], yy)
        w = yy.sum() * (~yy).sum()
        num += a * w; den += w
    return num / den if den else np.nan


rows_sets = {"TRAIN-OOF": (cv0, cv0["idx"]), "DEV": (arr0, arr0["idx"])}
print(f"{'set':10s} {'n':>5s} {'AUC d1 pooled':>14s} {'AUC d1 within-fam':>18s} "
      f"{'AUC d2 within-fam':>18s} {'rho(d1,EBd1)':>13s}")
for nm, (a, rows) in rows_sets.items():
    ps, pc = lab.compose(a, DEPLOYED_CFG, "fast")
    d1 = ps[:, 1] - ps[:, 0]; d2 = ps[:, 2] - ps[:, 1]
    ts = lab.true_s[rows]
    y1 = ts[:, 1] > ts[:, 0]; y2 = ts[:, 2] > ts[:, 1]
    fam = new[rows]
    from scipy.stats import spearmanr
    eb1 = p[rows][:, 1] - p[rows][:, 0]
    print(f"{nm:10s} {len(rows):5d} {auc(d1,y1)[0]:14.4f} "
          f"{pooled_within_family_auc(d1,y1,fam):18.4f} "
          f"{pooled_within_family_auc(d2,y2,fam):18.4f} "
          f"{spearmanr(d1, eb1).statistic:13.4f}")

print(f"\n{'set':10s} {'corr s_l':>9s} {'corr s_m':>9s} {'sd log c_l err':>15s} "
      f"{'mean log(pred/true) c_l':>24s} {'sum ratio c_l':>14s}")
for nm, (a, rows) in rows_sets.items():
    ps, pc = lab.compose(a, DEPLOYED_CFG, "fast")
    ts = lab.true_s[rows]; tc = lab.true_c[rows]
    e = np.log(pc[:, 0] / tc[:, 0])
    print(f"{nm:10s} {np.corrcoef(ps[:,0],ts[:,0])[0,1]:9.4f} "
          f"{np.corrcoef(ps[:,1],ts[:,1])[0,1]:9.4f} {e.std():15.4f} {e.mean():24.4f} "
          f"{pc[:,0].sum()/tc[:,0].sum():14.4f}")

# per-family mid-light gain, train vs dev, in realised and latent-p units
print("\n[family-level d1 = mean(s_mid - s_light)]  train / dev, realised and EB")
tr = lab.train_idx; dv = lab.dev_idx
print(f"{'family':16s} {'n tr':>5s} {'n dv':>5s} {'d1 tr':>8s} {'d1 dv':>8s} "
      f"{'EB d1 tr':>9s} {'EB d1 dv':>9s} {'dcost/L tr':>11s} {'dcost/L dv':>11s}")
for f in sorted(set(new)):
    a = tr[new[tr] == f]; b = dv[new[dv] == f]
    if len(a) < 5 or len(b) < 5:
        continue
    dta = lab.true_s[a][:, 1] - lab.true_s[a][:, 0]
    dtb = lab.true_s[b][:, 1] - lab.true_s[b][:, 0]
    ea = p[a][:, 1] - p[a][:, 0]; eb = p[b][:, 1] - p[b][:, 0]
    ca = (lab.true_c[a][:, 1] - lab.true_c[a][:, 0]).sum() / lab.true_c[tr][:, 0].sum()
    cb = (lab.true_c[b][:, 1] - lab.true_c[b][:, 0]).sum() / lab.true_c[dv][:, 0].sum()
    print(f"{f:16s} {len(a):5d} {len(b):5d} {dta.mean():8.4f} {dtb.mean():8.4f} "
          f"{ea.mean():9.4f} {eb.mean():9.4f} {ca:11.4f} {cb:11.4f}")
