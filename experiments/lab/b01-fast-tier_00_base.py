# SPDX-License-Identifier: Apache-2.0
"""Fast-tier baseline, oracle, latent-p oracle, and what k1 would buy."""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b01lib", HERE / "b01-fast-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP, DEPLOYED_SAFETY, MULTS  # noqa
import bench2 as B
import protocol as P

np.set_printoptions(precision=4, suppress=True)
lab = Lab()
MF = 1.25

# ---------------------------------------------------------------- 0. stages
cv0, arr0 = B.stage(lab, DEPLOYED_EXP, tag="base")
cv1, arr1 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
print("[stages] base + legoof loaded", flush=True)

for nm, (cv, arr) in (("baseline", (cv0, arr0)), ("legacy-OOF", (cv1, arr1))):
    r = B.run(lab, cv, arr, DEPLOYED_CFG, label=f"{nm}", nboot=400)
    d = r["dev_tiers"]["fast"]; e = r["det"]["fast"]
    print(f"   FAST {nm:12s} safety={r['safety']['fast']:.3f} EVfast={e['ev']:.6f} "
          f"bust={e['bust']*100:.1f}% devfast={d['score']:.6f} r={d['ratio']:.4f} "
          f"margin={d['margin']*100:.2f}%", flush=True)

# ------------------------------------------------- 1. dev picks at the E43 point
dev = lab.dev_idx
ps, pc = lab.compose(arr1, DEPLOYED_CFG, "fast")
st = L.fast_stats(lab, ps, pc, dev, DEPLOYED_SAFETY["fast"])
print(f"\n[dev @ safety .98 legoof] counts={st['counts']} score={st['score']:.4f} "
      f"ratio={st['ratio']:.4f} passed={st['passed']}")

# ---------------------------------------------------------------- 2. oracles
ts = lab.true_s[dev]; tc = lab.true_c[dev]
oc = L.fast_stats(lab, ts, tc, dev, 1.0)
print(f"[dev realised-score oracle] counts={oc['counts']} score={oc['score']:.4f} "
      f"ratio={oc['ratio']:.4f}")

p = L.eb_posterior(lab)
pdev = p[dev]
# latent-p oracle: allocate on p, evaluate on p
pick = P.exact_allocate(pdev, tc, MF, 1.0)
r = np.arange(len(dev))
print(f"[dev latent-p oracle    ] counts={np.bincount(pick,minlength=3).tolist()} "
      f"p-score={pdev[r,pick].mean():.4f} realised={ts[r,pick].mean():.4f} "
      f"ratio={tc[r,pick].sum()/tc[:,0].sum():.4f}")
# deployed allocation evaluated on p
print(f"[dev deployed on latent p] p-score={pdev[r,st['pick']].mean():.4f}  "
      f"all-light p={pdev[:,0].mean():.4f} realised all-light={ts[:,0].mean():.4f}")

# same on the OOF train rows (bigger sample, honest)
ci = cv1["idx"]
tsc = lab.true_s[ci]; tcc = lab.true_c[ci]
occ = L.fast_stats(lab, tsc, tcc, ci, 1.0)
pc_ = p[ci]
pk = P.exact_allocate(pc_, tcc, MF, 1.0)
rr = np.arange(len(ci))
print(f"[train realised oracle] counts={occ['counts']} score={occ['score']:.4f}")
print(f"[train latent-p oracle] counts={np.bincount(pk,minlength=3).tolist()} "
      f"p={pc_[rr,pk].mean():.4f} realised={tsc[rr,pk].mean():.4f} "
      f"all-light p={pc_[:,0].mean():.4f}")

# ------------------------------------------------ 3. should fast ever pick k1?
# oracle restricted to {light, mid}
big = np.full_like(tc, 0.0); big[:] = tc
tc_nok1 = tc.copy(); ts_nok1 = ts.copy()
ts_nok1[:, 2] = -1e9                      # forbid k1
o2 = L.fast_stats(lab, ts_nok1, tc_nok1, dev, 1.0)
print(f"\n[dev oracle, k1 banned ] counts={o2['counts']} score={o2['score']:.4f} "
      f"ratio={o2['ratio']:.4f}  (vs {oc['score']:.4f} with k1)")
p2 = pdev.copy(); p2[:, 2] = -1e9
pk2 = P.exact_allocate(p2, tc, MF, 1.0)
print(f"[dev latent-p oracle, k1 banned] counts={np.bincount(pk2,minlength=3).tolist()} "
      f"p={pdev[r,pk2].mean():.4f}")

# which items does the oracle send to k1 in fast?
k1i = np.where(oc["pick"] == 2)[0]
print(f"\n[oracle k1 picks in fast] n={len(k1i)} families="
      f"{np.unique(lab.fam_arr[dev][k1i], return_counts=True)}")
gain = ts[k1i, 2] - ts[k1i, 0]
cost = (tc[k1i, 2] - tc[k1i, 0]) / tc[:, 0].sum()
print(f"   mean realised gain over light={gain.mean():.3f}  total extra cost="
      f"{cost.sum():.4f}L  eff={gain.sum()/len(dev)/cost.sum():.3f} score/L")
