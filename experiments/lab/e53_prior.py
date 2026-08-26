# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""E53 - the real external difficulty prior, measured under bench2 (stressed safety).

Arms: no prior; full prior; prior masked to 70/50/30% coverage (a private set
drawn from the same benchmarks will not be 100% reproducible - dmmath is not
reproducible at all and ruletaker/longdoc only partly).
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, DEPLOYED_EXP, DEPLOYED_CFG
import bench2 as B
import priorfeat as PF

CORE = dict(DEPLOYED_EXP, legacy_oof_meta=True, meta_seeds=(11, 23, 37, 53, 71))
lab = Lab()
entries, means, glob = PF.load_table(["local-llm/labels_axlight.jsonl"])
F = PF.build_features(lab.texts, lab.fam_arr, entries, means, glob)
cov = PF.coverage_report(F, lab.fam_arr)
print(f"table {len(entries)} prompts; coverage: " + " ".join(f"{k}={v:.2f}" for k, v in sorted(cov.items())), flush=True)
print(f"  train {F[lab.train_idx,0].mean():.3f}  dev {F[lab.dev_idx,0].mean():.3f}", flush=True)
m = F[:, 0] > 0
if m.sum() > 20:
    for j, nm in enumerate(("light", "mid", "k1")):
        print(f"  corr(prior, true {nm}) on covered = {np.corrcoef(F[m,1], lab.true_s[m,j])[0,1]:.3f}")
    d1 = lab.true_s[m,1]-lab.true_s[m,0]; d2 = lab.true_s[m,2]-lab.true_s[m,1]
    print(f"  corr(prior, d1) = {np.corrcoef(F[m,1], d1)[0,1]:+.3f}   corr(prior, d2) = {np.corrcoef(F[m,1], d2)[0,1]:+.3f}")
    print(f"  corr(outlen, d2) = {np.corrcoef(F[m,5], d2)[0,1]:+.3f}")

res = []
lab.set_extra_features(None)
cv, arr = B.stage(lab, CORE, tag="core5")
res.append(B.run(lab, cv, arr, DEPLOYED_CFG, label="no prior"))
for frac in (1.0, 0.7, 0.5, 0.3):
    G = F if frac == 1.0 else PF.mask_coverage(F, frac, seed=3)
    lab.set_extra_features(G)
    cv, arr = B.stage(lab, CORE, tag=f"prior{int(frac*100)}", force=True)
    res.append(B.run(lab, cv, arr, DEPLOYED_CFG, label=f"prior coverage {frac:.0%}"))
lab.set_extra_features(None)
Path("reports/lab/e53_prior.json").write_text(json.dumps(
    [{k: v for k, v in r.items() if k != "curves"} for r in res], indent=2, default=float), encoding="utf-8")
