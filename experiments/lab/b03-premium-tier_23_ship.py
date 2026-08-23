# SPDX-License-Identifier: Apache-2.0
"""b03 step 23: the shippable premium policy, priced on three independent risk pools."""
import sys, pickle, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P

lab = Lab()
cvB, arrB = B.stage(lab, DEPLOYED_EXP, tag="base")
cv1, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
cv2 = pickle.loads(Path("reports/lab/b03_cv_seed777.pkl").read_bytes())
POOLS = {"trainOOF-s123": cv1, "trainOOF-s777": cv2, "dev": arr}

def tier_ev(x, tier, g, cfg=DEPLOYED_CFG):
    idx = x["idx"]; ts = lab.true_s[idx]; tc = lab.true_c[idx]; n = len(idx)
    ps, pc = lab.compose(x, cfg, tier)
    ev = bu = 0.0
    for s in (7, 17, 23):
        smp = np.asarray(lab.samples_for(n, s, 400, 880))
        e, b, _ = P.safety_curve(ps[smp], pc[smp], ts[smp], tc[smp], MULTS[tier], np.array([g]))
        ev += e[0] / 3; bu += b[0] / 3
    return ev, bu

print("PREMIUM: EV / bust% on three pools, plus the single dev realisation")
print(f"{'safety':>7}" + "".join(f"{p:>22}" for p in POOLS) + f"{'devScore':>10}{'devRatio':>10}{'margin%':>9}")
rows = {}
for g in (0.70, 0.73, 0.76, 0.79, 0.82, 0.84, 0.87):
    cells = []
    for p, x in POOLS.items():
        e, b = tier_ev(x, "premium", g)
        cells.append(f"{e:14.4f}/{100*b:6.2f}")
    ps, pc = lab.compose(arr, DEPLOYED_CFG, "premium")
    pk = P.exact_allocate(ps, pc, 4.0, g); r = np.arange(len(arr["idx"]))
    rt = lab.true_c[arr["idx"]][r, pk].sum() / lab.true_c[arr["idx"]][:, 0].sum()
    sc = lab.true_s[arr["idx"]][r, pk].mean()
    rows[g] = dict(dev=float(sc), ratio=float(rt))
    print(f"{g:7.2f}" + "".join(cells) + f"{sc:10.4f}{rt:10.3f}{100*(4/rt-1):9.2f}")

print("\nFAST (not my tier - flagged): same table")
print(f"{'safety':>7}" + "".join(f"{p:>22}" for p in POOLS) + f"{'devScore':>10}{'devRatio':>10}{'margin%':>9}")
for g in (0.90, 0.92, 0.94, 0.95, 0.96, 0.98):
    cells = []
    for p, x in POOLS.items():
        e, b = tier_ev(x, "fast", g)
        cells.append(f"{e:14.4f}/{100*b:6.2f}")
    ps, pc = lab.compose(arr, DEPLOYED_CFG, "fast")
    pk = P.exact_allocate(ps, pc, 1.25, g); r = np.arange(len(arr["idx"]))
    rt = lab.true_c[arr["idx"]][r, pk].sum() / lab.true_c[arr["idx"]][:, 0].sum()
    sc = lab.true_s[arr["idx"]][r, pk].mean()
    print(f"{g:7.2f}" + "".join(cells) + f"{sc:10.4f}{rt:10.3f}{100*(1.25/rt-1):9.2f}")

print("\n=== final packages (bench2, fixed safety) ===")
for nm, tri in (("C1 @ bench2 argmax (BRIEF2 §5)", {"fast": .960, "balanced": .825, "premium": .840}),
                ("C1 @ b03 risk-priced", {"fast": .940, "balanced": .820, "premium": .760}),
                ("C1 @ b03 premium-only change", {"fast": .960, "balanced": .825, "premium": .760}),
                ("base (no C1) @ bench2 argmax", {"fast": .960, "balanced": .840, "premium": .735})):
    src = (cvB, arrB) if nm.startswith("base") else (cv1, arr)
    r = B.run(lab, src[0], src[1], DEPLOYED_CFG, label=nm, fixed_safety=tri)
    # dev-pool risk of the same triple
    dp = {t: tier_ev(src[1], t, tri[t]) for t in TIERS}
    print(f"      dev-pool EV={sum(W[t]*dp[t][0] for t in TIERS):.6f}"
          f"  bust%={'/'.join(f'{100*dp[t][1]:.1f}' for t in TIERS)}")
