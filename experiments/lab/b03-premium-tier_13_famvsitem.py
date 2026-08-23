# SPDX-License-Identifier: Apache-2.0
"""b03 step 13: family-level k1 budget split vs item-level allocation; per-family k1 veto."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P

lab = Lab()
cv, arr = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")
ci, di = cv["idx"], arr["idx"]
ts, tc = lab.true_s[ci], lab.true_c[ci]
m = len(ci); MULT = 4.0
fam_cv, fam_dv = lab.fam_arr[ci], lab.fam_arr[di]
FAMS = sorted(set(lab.fam_arr))
ps0, pc0 = lab.compose(cv, DEPLOYED_CFG, "premium")
GRID = np.arange(0.50, 1.401, 0.005)
TUNE, EVAL = (7, 17), (101, 103, 107)

def ev_of(picks_fn, label):
    """picks_fn(ps, pc, cap_frac, smp) -> picks; here we only need safety-swept allocators."""
    pass

# ---------- family-level policy: k1 for whole families, ordered by train efficiency ----
# train-only family menu, in BATCH light units (the budget denominator)
Lbar = tc[:, 0].sum()
menu = []
for f in FAMS:
    s = fam_cv == f
    dsc = (ts[s, 2] - ts[s, 1]).sum() / m                      # score per item of the batch
    dc = (tc[s, 2] - tc[s, 1]).sum() / Lbar                    # cost in batch-light units
    menu.append((f, dsc / dc if dc > 0 else -1e9, dsc, dc, int(s.sum())))
menu.sort(key=lambda r: -r[1])
print("train-only family menu for the mid->k1 upgrade (batch-normalised)")
print(f"{'family':<18}{'eff':>9}{'dscore':>10}{'dcost(u)':>10}{'n':>6}")
for f, e, d, c, n in menu:
    print(f"{f:<18}{e:9.4f}{d:+10.4f}{c:10.4f}{n:6d}")

def fam_alloc(ps, pc, safety, order, item_inside=True):
    """Spend the k1 budget family by family in `order`; inside a family either take
    every item (pure family policy) or the best items by predicted efficiency."""
    B_, mm, _ = ps.shape
    # start from the deployed light/mid solve with k1 removed
    ps2 = ps.copy(); ps2[:, :, 2] = -1e9
    picks = P.exact_allocate(ps2, pc, MULT, safety)
    cap = pc[:, :, 0].sum(axis=1) * max(1.0, MULT * safety)
    spent = np.take_along_axis(pc, picks[:, :, None], axis=2)[:, :, 0].sum(axis=1)
    fam = np.asarray(order[1])
    for f in order[0]:
        sel = fam == f
        if not sel.any():
            continue
        idxs = np.where(sel)[0]
        if item_inside:
            eff = (ps[:, :, 2] - ps[:, :, 1]) / np.maximum(pc[:, :, 2] - pc[:, :, 1], 1e-12)
            for b in range(B_):
                o = idxs[np.argsort(-eff[b, idxs])]
                dc = pc[b, o, 2] - np.take_along_axis(pc[b, o], picks[b, o][:, None], axis=1)[:, 0]
                cum = spent[b] + np.cumsum(dc)
                k = int(np.searchsorted(cum, cap[b], side="right"))
                picks[b, o[:k]] = 2
                spent[b] = cum[k - 1] if k else spent[b]
        else:
            for b in range(B_):
                dc = pc[b, idxs, 2] - np.take_along_axis(pc[b, idxs], picks[b, idxs][:, None], axis=1)[:, 0]
                if spent[b] + dc.sum() <= cap[b]:
                    picks[b, idxs] = 2; spent[b] += dc.sum()
    return picks

def score_policy(fn, label, grid=np.arange(0.50, 1.30, 0.02)):
    best = (-1, None)
    for g in grid:
        vals = []
        for s in TUNE:
            smp = np.asarray(lab.samples_for(m, s, 150, 880))
            pk = fn(ps0[smp], pc0[smp], float(g), (list(x[0] for x in menu), fam_cv[smp[0]]))
            # note: family vector must follow the sample
            vals.append(None)
        break
    return None

# simpler: evaluate family policies deterministically on OOF pool and on dev, plus a
# bootstrap EV using a fixed safety grid
def run_fam(order_names, item_inside, safeties):
    out = []
    for g in safeties:
        vals = []
        for s in EVAL:
            smp = np.asarray(lab.samples_for(m, s, 150, 880))
            PS = ps0[smp]; PC = pc0[smp]
            picks = np.zeros(PS.shape[:2], dtype=int)
            for b in range(PS.shape[0]):
                pk = fam_alloc(PS[b:b+1], PC[b:b+1], g, (order_names, fam_cv[smp[b]]), item_inside)
                picks[b] = pk[0]
            C = np.take_along_axis(tc[smp], picks[:, :, None], axis=2)[:, :, 0]
            S = np.take_along_axis(ts[smp], picks[:, :, None], axis=2)[:, :, 0]
            R = C.sum(axis=1) / tc[smp][:, :, 0].sum(axis=1)
            vals.append(np.where(R > MULT, 0.0, S.mean(axis=1)))
        v = np.concatenate(vals)
        out.append((g, v.mean(), np.mean(v == 0)))
    return out

order_names = [x[0] for x in menu]
print("\nfamily-level k1 budget split (families taken whole, best-efficiency order)")
for g, e, b in run_fam(order_names, False, (0.75, 0.84, 0.95)):
    print(f"  safety={g:.2f}  premEV={e:.4f} bust={100*b:.2f}%")
print("family-budget order + item-level ordering INSIDE each family")
for g, e, b in run_fam(order_names, True, (0.75, 0.84, 0.95)):
    print(f"  safety={g:.2f}  premEV={e:.4f} bust={100*b:.2f}%")
print("reference: deployed item-level Lagrangian")
for g in (0.75, 0.84, 0.95):
    vals = []
    for s in EVAL:
        smp = np.asarray(lab.samples_for(m, s, 150, 880))
        pk = P.exact_allocate(ps0[smp], pc0[smp], MULT, g)
        C = np.take_along_axis(tc[smp], pk[:, :, None], axis=2)[:, :, 0]
        S = np.take_along_axis(ts[smp], pk[:, :, None], axis=2)[:, :, 0]
        R = C.sum(axis=1) / tc[smp][:, :, 0].sum(axis=1)
        vals.append(np.where(R > MULT, 0.0, S.mean(axis=1)))
    v = np.concatenate(vals)
    print(f"  safety={g:.2f}  premEV={v.mean():.4f} bust={100*np.mean(v==0):.2f}%")

# ---------- per-family k1 veto, one family at a time ----------------------
print("\nper-family k1 veto (premium only), honest bench2:")
def veto(fs):
    def tr(lab_, a, ps, pc, tier):
        if tier != "premium":
            return ps, pc
        ps = ps.copy()
        f = lab.fam_arr[a["idx"]]
        ps[np.isin(f, fs), 2] = 0.0
        return ps, pc
    return tr
B.run(lab, cv, arr, DEPLOYED_CFG, label="A0 legoof base")
for f in ("code", "dmmath", "belebele", "truthfulqa", "gsm8k_or_other"):
    B.run(lab, cv, arr, DEPLOYED_CFG, transform=veto([f]), label=f"no k1 on {f} (premium)")
