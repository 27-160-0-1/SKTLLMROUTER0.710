# SPDX-License-Identifier: Apache-2.0
"""b05 step 0: reproduce the two reference points and time one fit_predict."""
import sys, time, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_EXP, DEPLOYED_CFG, DEPLOYED_SAFETY
import bench2 as B

lab = Lab()
t0 = time.perf_counter()
cv0, arr0 = B.stage(lab, DEPLOYED_EXP, tag="base")
cv1, arr1 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
print(f"[b05] stages loaded in {time.perf_counter()-t0:.1f}s", flush=True)

r0 = B.run(lab, cv0, arr0, DEPLOYED_CFG, label="baseline")
r1 = B.run(lab, cv1, arr1, DEPLOYED_CFG, label="legacy-OOF meta (C1)")

# gain-axis diagnostics on the OOF rows and on dev
def gain_diag(lab, cv, cfg, tag):
    idx = cv["idx"]
    ts = lab.true_s[idx]
    d1 = ts[:, 1] - ts[:, 0]; d2 = ts[:, 2] - ts[:, 1]
    fam = lab.fam_arr[idx]
    out = {}
    for t in ("fast",):
        ps, _pc = lab.compose(cv, cfg, t)
        g1 = ps[:, 1] - ps[:, 0]; g2 = ps[:, 2] - ps[:, 1]
        out["corr_d1"] = float(np.corrcoef(g1, d1)[0, 1])
        out["corr_d2"] = float(np.corrcoef(g2, d2)[0, 1])
        # pooled within-family AUC of sign
        for nm, g, d in (("auc1", g1, d1), ("auc2", g2, d2)):
            num = den = 0.0
            for f in np.unique(fam):
                m = fam == f
                pos = g[m][d[m] > 0]; neg = g[m][d[m] < 0]
                if len(pos) == 0 or len(neg) == 0:
                    continue
                w = len(pos) * len(neg)
                a = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
                num += a * w; den += w
            out[nm] = num / den
    print(f"[{tag}] corr(d1)={out['corr_d1']:.4f} corr(d2)={out['corr_d2']:.4f} "
          f"wf-AUC1={out['auc1']:.4f} wf-AUC2={out['auc2']:.4f}", flush=True)
    return out

gain_diag(lab, cv0, DEPLOYED_CFG, "OOF base")
gain_diag(lab, cv1, DEPLOYED_CFG, "OOF legoof")
gain_diag(lab, arr0, DEPLOYED_CFG, "DEV base")
gain_diag(lab, arr1, DEPLOYED_CFG, "DEV legoof")

t0 = time.perf_counter()
_ = lab.fit_predict(lab.train_idx[:1584], lab.train_idx[1584:], dict(DEPLOYED_EXP, legacy_oof_meta=True))
print(f"[b05] one fit_predict (1584 fit / 176 hold) = {time.perf_counter()-t0:.1f}s", flush=True)
