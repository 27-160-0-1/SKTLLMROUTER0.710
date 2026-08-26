# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b03 step 10: mid-success gate, hard cost ceiling, family-level vs item-level policy."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, W, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
ci, di = cv["idx"], arr["idx"]
FAMS = sorted(set(lab.fam_arr))

# ---- (C) mid-success gate: forbid k1 when the mid score is predicted high ----
def gate(thr, tiers=("premium",)):
    def tr(lab_, a, ps, pc, tier):
        if tier not in tiers:
            return ps, pc
        ps = ps.copy()
        ps[ps[:, 1] >= thr, 2] = 0.0
        return ps, pc
    return tr

# ---- (E) hard ceiling on the predicted k1 cost (C8-style), in light-batch units ----
def ceiling(frac, tiers=("premium",)):
    def tr(lab_, a, ps, pc, tier):
        if tier not in tiers:
            return ps, pc
        ps = ps.copy()
        share = pc[:, 2] / pc[:, 0].sum()
        ps[share > frac, 2] = 0.0
        return ps, pc
    return tr

# ---- (D) family-level k1 policy: only the top-q fraction of each family may use k1 ----
def famtop(q_by_fam, tiers=("premium",)):
    def tr(lab_, a, ps, pc, tier):
        if tier not in tiers:
            return ps, pc
        ps = ps.copy()
        fam = lab.fam_arr[a["idx"]]
        eff = (ps[:, 2] - ps[:, 1]) / np.maximum(pc[:, 2] - pc[:, 1], 1e-12)
        for f in FAMS:
            s = np.where(fam == f)[0]
            if len(s) == 0:
                continue
            q = q_by_fam.get(f, 1.0)
            if q >= 1.0:
                continue
            k = int(np.floor(q * len(s)))
            order = s[np.argsort(-eff[s])]
            ps[order[k:], 2] = 0.0
        return ps, pc
    return tr

base = B.run(lab, cv, arr, DEPLOYED_CFG, label="A0 legoof base")
for t in (0.75, 0.85, 0.90, 0.95):
    B.run(lab, cv, arr, DEPLOYED_CFG, transform=gate(t), label=f"gate: no k1 if s_mid_hat>={t}")
for f in (0.02, 0.01, 0.005, 0.002):
    B.run(lab, cv, arr, DEPLOYED_CFG, transform=ceiling(f), label=f"ceiling: k1 cost <= {f} of L")

# family-level: veto k1 entirely for the durable-negative families, then quotas
NEG = {"ruletaker": 0.0, "longdoc": 0.0, "hrmcr": 0.0}
B.run(lab, cv, arr, DEPLOYED_CFG, transform=famtop(NEG), label="family veto: rt/ld/hrmcr no k1")
for q in (0.05, 0.10, 0.20, 0.35):
    qq = dict(NEG); qq.update({f: q for f in ("belebele", "truthfulqa")})
    B.run(lab, cv, arr, DEPLOYED_CFG, transform=famtop(qq),
          label=f"+ belebele/truthfulqa k1 quota {q:.0%}")

# ---- family-level k1 economics: premium vs balanced menu -------------------
print("\n--- realised k1 usage and value by family (dev, safety at EV-opt) ---")
sfty = {"fast": 0.960, "balanced": 0.825, "premium": 0.840}
for t in TIERS:
    ps, pc = lab.compose(arr, DEPLOYED_CFG, t)
    pk = P.exact_allocate(ps, pc, {"fast": 1.25, "balanced": 2.0, "premium": 4.0}[t], sfty[t])
    fam = lab.fam_arr[di]; tc = lab.true_c[di]; ts = lab.true_s[di]
    D = tc[:, 0].sum(); r = np.arange(len(di))
    up = (tc[r, pk] - tc[:, 0]) / D
    print(f"  {t}: picks {np.bincount(pk,minlength=3).tolist()} ratio {tc[r,pk].sum()/D:.3f}")
    for f in FAMS:
        s = fam == f
        nk = int((pk[s] == 2).sum()); nm = int((pk[s] == 1).sum())
        if nk == 0 and t != "premium":
            continue
        d2 = (ts[s][pk[s] == 2, 2] - ts[s][pk[s] == 2, 1]).sum() / len(di) if nk else 0.0
        ck = up[s][pk[s] == 2].sum() if nk else 0.0
        print(f"     {f:<16} n={int(s.sum()):4d} k1={nk:3d} mid={nm:4d}"
              f"  k1cost={ck:.3f}u  dscore={d2:+.4f}"
              f"  eff={d2/ck if ck>1e-9 else float('nan'):7.4f}")
