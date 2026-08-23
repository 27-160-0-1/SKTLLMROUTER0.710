# SPDX-License-Identifier: Apache-2.0
"""b03 step 12: premium-tier headroom ladder (per-column cost oracles, gain oracles)
and shift stress of the safety choice.  All evaluated with the honest EV protocol."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
ci, di = cv["idx"], arr["idx"]
ts, tc = lab.true_s[ci], lab.true_c[ci]
m = len(ci); MULT = 4.0
ps0, pc0 = lab.compose(cv, DEPLOYED_CFG, "premium")
GRID = np.arange(0.50, 1.601, 0.005)
TUNE, EVAL = (7, 17), (101, 103, 107)

def evaluate(ps, pc, label, pool=None):
    ev = np.zeros(len(GRID))
    T, C = (ts, tc) if pool is None else pool
    for s in TUNE:
        smp = np.asarray(lab.samples_for(m, s, 400, 880))
        e, _b, _r = P.safety_curve(ps[smp], pc[smp], T[smp], C[smp], MULT, GRID)
        ev += e / len(TUNE)
    sf = float(GRID[int(np.argmax(ev))])
    out = []
    for s in EVAL:
        smp = np.asarray(lab.samples_for(m, s, 400, 880))
        pk = P.exact_allocate(ps[smp], pc[smp], MULT, sf)
        Cc = np.take_along_axis(C[smp], pk[:, :, None], axis=2)[:, :, 0]
        Ss = np.take_along_axis(T[smp], pk[:, :, None], axis=2)[:, :, 0]
        R = Cc.sum(axis=1) / C[smp][:, :, 0].sum(axis=1)
        out.append(np.where(R > MULT, 0.0, Ss.mean(axis=1)))
    x = np.concatenate(out)
    print(f"  {label:<38} safety={sf:5.3f} premEV={x.mean():.4f} bust={100*np.mean(x==0):5.2f}%")
    return x

print("=== premium cost-side oracle ladder (score predictions untouched) ===")
base = evaluate(ps0, pc0, "base")
for j, nm in ((0, "light"), (1, "mid"), (2, "k1")):
    pc = pc0.copy(); pc[:, j] = tc[:, j]
    pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
    pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
    evaluate(ps0, pc, f"true {nm} cost only")
evaluate(ps0, tc.copy(), "true cost (all three)")
# k1 cost: keep the family level, give the item-level truth (and vice versa)
fam = lab.fam_arr[ci]
lvl = pc0.copy()
for f in sorted(set(fam)):
    s = fam == f
    lvl[s, 2] = pc0[s, 2] * (tc[s, 2].sum() / pc0[s, 2].sum())
evaluate(ps0, lvl, "k1 cost: family level matched only")

print("\n=== premium score-side oracle ladder (cost predictions untouched) ===")
d1t = ts[:, 1] - ts[:, 0]; d2t = ts[:, 2] - ts[:, 1]
for nm, ps in (("true d2 (k1-mid gain)", np.column_stack([ps0[:, 0], ps0[:, 1], ps0[:, 1] + d2t])),
               ("true d1 (mid-light gain)", np.column_stack([ps0[:, 0], ps0[:, 0] + d1t,
                                                             ps0[:, 0] + d1t + (ps0[:, 2] - ps0[:, 1])])),
               ("true score (all)", ts.copy())):
    evaluate(np.clip(ps, 0, 1), pc0, nm)
# partial d2 knowledge: blend
for lam in (0.25, 0.5):
    ps = ps0.copy(); ps[:, 2] = np.clip(ps[:, 1] + (1 - lam) * (ps0[:, 2] - ps0[:, 1]) + lam * d2t, 0, 1)
    evaluate(ps, pc0, f"d2 blended toward truth lam={lam}")

print("\n=== shift stress: EV of a fixed safety under re-weighted pools ===")
scen = {"nominal": np.ones(m)}
w = np.ones(m); w[np.isin(fam, ["longdoc"])] = 0.5; scen["half longdoc (budget shrinks)"] = w
w = np.ones(m); w[np.isin(fam, ["code", "dmmath", "aime"])] = 3.0; scen["reasoning x3"] = w
w = np.ones(m); w[np.isin(fam, ["ruletaker", "longdoc"])] = 3.0; scen["ruletaker/longdoc x3"] = w
og = lab.otok[ci][:, 2] / lab.ngen[ci][:, 2]
w = 1.0 + 2.0 * (og > np.quantile(og, 0.8)); scen["long-think x3"] = w
print(f"{'scenario':<32}" + "".join(f"{g:>8.2f}" for g in (0.75, 0.80, 0.84, 0.88, 0.92)))
for nm, w in scen.items():
    p = w / w.sum()
    row = []
    for g in (0.75, 0.80, 0.84, 0.88, 0.92):
        vals = []
        for s in (7, 17, 23):
            rng = np.random.default_rng(s)
            smp = rng.choice(m, size=(400, 880), p=p)
            pk = P.exact_allocate(ps0[smp], pc0[smp], MULT, g)
            Cc = np.take_along_axis(tc[smp], pk[:, :, None], axis=2)[:, :, 0]
            Ss = np.take_along_axis(ts[smp], pk[:, :, None], axis=2)[:, :, 0]
            R = Cc.sum(axis=1) / tc[smp][:, :, 0].sum(axis=1)
            vals.append(np.where(R > MULT, 0.0, Ss.mean(axis=1)))
        v = np.concatenate(vals)
        row.append(f"{v.mean():8.4f}")
    print(f"{nm:<32}" + "".join(row))
print("\nbust% under the same scenarios")
for nm, w in scen.items():
    p = w / w.sum(); row = []
    for g in (0.75, 0.80, 0.84, 0.88, 0.92):
        vals = []
        for s in (7, 17, 23):
            rng = np.random.default_rng(s)
            smp = rng.choice(m, size=(400, 880), p=p)
            pk = P.exact_allocate(ps0[smp], pc0[smp], MULT, g)
            Cc = np.take_along_axis(tc[smp], pk[:, :, None], axis=2)[:, :, 0]
            R = Cc.sum(axis=1) / tc[smp][:, :, 0].sum(axis=1)
            vals.append(R > MULT)
        row.append(f"{100*np.concatenate(vals).mean():8.2f}")
    print(f"{nm:<32}" + "".join(row))
