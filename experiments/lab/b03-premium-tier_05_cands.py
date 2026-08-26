# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b03 step 5: C6 rescue / C5 kappa2 / C4 variance-form, measured under bench2."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import importlib
lib = importlib.import_module("b03-premium-tier_lib")

lab = Lab()
EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)
cv, arr = B.stage(lab, EXP, tag="legoof")
ci = cv["idx"]; di = arr["idx"]
fam_cv = lab.fam_arr[ci]; fam_dv = lab.fam_arr[di]
inner = lib.inner_folds(len(ci))

# ---- pre-compute, per tier, the cross-fit rescue predictions ---------------
RESCUE_CV, RESCUE_DV = {}, {}
SIG_CV, SIG_DV = {}, {}
for t in TIERS:
    pcv, ccv = lab.compose(cv, DEPLOYED_CFG, t)
    pdv, cdv = lab.compose(arr, DEPLOYED_CFG, t)
    r_cv = np.zeros(len(ci))
    for k in range(lib.FOLDS):
        co = lib.fit_rescue(pcv[inner != k, 1], lab.true_s[ci][inner != k, 2], fam_cv[inner != k])
        r_cv[inner == k] = lib.apply_rescue(co, pcv[inner == k, 1], fam_cv[inner == k])
    co_all = lib.fit_rescue(pcv[:, 1], lab.true_s[ci][:, 2], fam_cv)
    RESCUE_CV[t] = r_cv
    RESCUE_DV[t] = lib.apply_rescue(co_all, pdv[:, 1], fam_dv)
    s_cv = np.zeros((len(ci), 3))
    for k in range(lib.FOLDS):
        sg = lib.fit_sigma2(lab, ci[inner != k], ccv[inner != k])
        s_cv[inner == k] = lib.apply_sigma2(sg, fam_cv[inner == k], 1.0)
    sg_all = lib.fit_sigma2(lab, ci, ccv)
    SIG_CV[t] = s_cv
    SIG_DV[t] = lib.apply_sigma2(sg_all, fam_dv, 1.0)

print("rescue corr with true s_k1  cv=%.3f dev=%.3f" % (
    np.corrcoef(RESCUE_CV["premium"], lab.true_s[ci][:, 2])[0, 1],
    np.corrcoef(RESCUE_DV["premium"], lab.true_s[di][:, 2])[0, 1]))


def mk(alpha=0.0, kappa=0.0, k2=1.0, k1m=1.0, tiers=("premium",)):
    def tr(lab_, a, ps, pc, tier):
        if tier not in tiers:
            return ps, pc
        ps = ps.copy(); pc = pc.copy()
        if alpha > 0:
            R = RESCUE_CV[tier] if len(a["idx"]) == len(ci) else RESCUE_DV[tier]
            ps[:, 2] = (1 - alpha) * ps[:, 2] + alpha * R
        if kappa > 0:
            S = SIG_CV[tier] if len(a["idx"]) == len(ci) else SIG_DV[tier]
            pc = pc * (S ** kappa)
        pc[:, 1] *= k1m; pc[:, 2] *= k2
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        return np.clip(ps, 0, 1), pc
    return tr


base = B.run(lab, cv, arr, DEPLOYED_CFG, label="A0 legoof base")
res = {"A0": base}
for a in (0.5, 0.75, 1.0):
    res[f"resc{a}"] = B.run(lab, cv, arr, DEPLOYED_CFG, transform=mk(alpha=a),
                            label=f"C6 rescue prem alpha={a}")
for k in (1.24, 1.5, 2.0):
    res[f"k2_{k}"] = B.run(lab, cv, arr, DEPLOYED_CFG, transform=mk(k2=k),
                           label=f"C5 kappa2={k} prem")
for kp in (0.5, 1.0, 1.5):
    res[f"var{kp}"] = B.run(lab, cv, arr, DEPLOYED_CFG, transform=mk(kappa=kp),
                            label=f"C4 var-form kappa={kp} prem")
res["combo"] = B.run(lab, cv, arr, DEPLOYED_CFG, transform=mk(alpha=1.0, k2=1.5),
                     label="C6(1.0)+C5(1.5) prem")
print("\npremium-tier detail:")
for k, r in res.items():
    d = r["det"]["premium"]; t = r["dev_tiers"]["premium"]
    print(f"  {k:<12} premEV={d['ev']:.4f} bust={d['bust']*100:5.2f}% raw={d['raw']:.4f}"
          f"  safety={r['safety']['premium']:.3f}  devPrem={t['score']:.4f}/r{t['ratio']:.3f}"
          f"  EV={r['EV']:.6f} dev={r['dev']:.6f}")
