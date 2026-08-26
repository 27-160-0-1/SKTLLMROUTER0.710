# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a05 step 2: WHERE the loss lives -- per family, per realised-score pattern,
wasted-budget accounting, and a policy-level (re-solved) Shapley of the gap."""
from __future__ import annotations
import sys, itertools
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, TIERS, TIER_MULT, TIER_WEIGHT, tier_result, MODEL_IDS
ROOT = Path(__file__).resolve().parents[2]

dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
C = np.load(ROOT / "experiments/lab/a05-selection-failure_cache.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
N = len(dv); IDX = np.arange(N)
FAM = C["fam"]; PHAT = C["phat"]
FAMS = sorted(set(FAM.tolist()))
L = dv.cost[:, 0].sum()
SHORT = ["L", "M", "K"]

# ---------------------------------------------------------------- policy-level Shapley
print("=== policy-level counterfactual ladder (each cell re-solves the allocation honestly) ===")
Sopt = {0: (lambda t: P[f"score_{t}"]), 1: (lambda t: dv.score)}
Copt = {0: (lambda t: P[f"cost_{t}"]), 1: (lambda t: dv.cost)}
Fopt = {0: (lambda t: SAFE[t]), 1: (lambda t: 1.0)}
vals = {}
for bs, bc, bf in itertools.product((0, 1), repeat=3):
    tot = 0.0
    for t in TIERS:
        r = tier_result(Sopt[bs](t), Copt[bc](t), dv, t, Fopt[bf](t))
        tot += TIER_WEIGHT[t] * r["tier_score"]
    vals[(bs, bc, bf)] = tot
    print(f"  score={'TRUE' if bs else 'pred'} cost={'TRUE' if bc else 'pred'} "
          f"safety={'1.0 ' if bf else 'dep '}  final={tot:.4f}")
perms = list(itertools.permutations(range(3)))
sh = np.zeros(3)
for perm in perms:
    st = [0, 0, 0]; prev = vals[tuple(st)]
    for pl in perm:
        st[pl] = 1
        sh[pl] += (vals[tuple(st)] - prev) / len(perms)
        prev = vals[tuple(st)]
nm = ["true score", "true cost", "safety=1.0"]
print("  policy-level Shapley of the 0.1018 gap:",
      "  ".join(f"{nm[i]}={sh[i]:+.4f}" for i in range(3)), f" (sum {sh.sum():+.4f})")

# ---------------------------------------------------------------- per-family
print("\n=== per-family selection and loss (weighted over tiers) ===")
hdr = f"{'family':14s} {'n':>4s} {'costK/L':>8s} | {'dep L/M/K':>14s} {'orc L/M/K':>14s} | " \
      f"{'realised loss':>13s} {'EB loss':>9s} {'dep cost/L':>10s} {'orc cost/L':>10s}"
print(hdr)
rows = []
for f_ in FAMS:
    m = FAM == f_
    dep_cnt = np.zeros(3); orc_cnt = np.zeros(3)
    rl = el = dc = oc = 0.0
    for t in TIERS:
        sd = C[f"sel_d_{t}"]; so = C[f"sel_o_{t}"]
        w = TIER_WEIGHT[t]
        dep_cnt += w * np.bincount(sd[m], minlength=3)
        orc_cnt += w * np.bincount(so[m], minlength=3)
        rl += w * (dv.score[m, so[m]] - dv.score[m, sd[m]]).sum() / N
        el += w * (PHAT[m, so[m]] - PHAT[m, sd[m]]).sum() / N
        dc += w * dv.cost[m, sd[m]].sum() / L
        oc += w * dv.cost[m, so[m]].sum() / L
    rows.append((f_, m.sum(), rl, el, dc, oc))
    print(f"{f_:14s} {m.sum():4d} {dv.cost[m,2].sum()/L:8.3f} | "
          f"{dep_cnt[0]:4.0f}/{dep_cnt[1]:4.0f}/{dep_cnt[2]:4.0f} "
          f"{orc_cnt[0]:4.0f}/{orc_cnt[1]:4.0f}/{orc_cnt[2]:4.0f} | "
          f"{rl:+13.4f} {el:+9.4f} {dc:10.3f} {oc:10.3f}")
print(f"{'TOTAL':14s} {N:4d} {dv.cost[:,2].sum()/L:8.3f} | {'':14s} {'':14s} | "
      f"{sum(r[2] for r in rows):+13.4f} {sum(r[3] for r in rows):+9.4f} "
      f"{sum(r[4] for r in rows):10.3f} {sum(r[5] for r in rows):10.3f}")

# ---------------------------------------------------------------- realised-score patterns
print("\n=== realised-score patterns: what the deployed router did ===")
sL, sM, sK = dv.score[:, 0], dv.score[:, 1], dv.score[:, 2]
pats = {
    "A light already 1.0":                 sL >= 1.0,
    "B light 0, mid 1.0":                  (sL <= 0) & (sM >= 1.0),
    "C light 0, k1 1.0":                   (sL <= 0) & (sK >= 1.0),
    "D light 0, mid 0, k1 0 (hopeless)":   (sL <= 0) & (sM <= 0) & (sK <= 0),
    "E light==mid==k1 (no gain anywhere)": (sL == sM) & (sM == sK),
    "F mid <= light (mid useless)":        sM <= sL,
    "G k1 <= light (k1 useless)":          sK <= sL,
}
for nmp, msk in pats.items():
    line = f"  {nmp:36s} n={msk.sum():4d} "
    for t in TIERS:
        sd = C[f"sel_d_{t}"]
        up = (sd[msk] > 0).mean()
        spend = (dv.cost[msk, sd[msk]] - dv.cost[msk, 0]).sum() / L
        line += f"| {t[:4]} up={up:5.1%} spend={spend:6.3f}L "
    print(line)

# ---------------------------------------------------------------- wasted budget
print("\n=== wasted-budget accounting (deployed allocation) ===")
print(f"{'tier':9s} {'upgrades':>8s} {'spend/L':>8s} | {'no-change up':>12s} {'wasted/L':>9s} "
      f"{'harmful up':>10s} {'harm/L':>8s} {'harm score':>10s} | {'useful up':>9s} {'gain':>7s}")
waste = {}
for t in TIERS:
    sd = C[f"sel_d_{t}"]
    up = sd > 0
    spend = (dv.cost[IDX, sd] - dv.cost[:, 0]).sum() / L
    ds = dv.score[IDX, sd] - dv.score[:, 0]
    nochg = up & (ds == 0)
    harm = up & (ds < 0)
    useful = up & (ds > 0)
    w_no = (dv.cost[nochg, sd[nochg]] - dv.cost[nochg, 0]).sum() / L
    w_hm = (dv.cost[harm, sd[harm]] - dv.cost[harm, 0]).sum() / L
    print(f"{t:9s} {up.sum():8d} {spend:8.3f} | {nochg.sum():12d} {w_no:9.3f} "
          f"{harm.sum():10d} {w_hm:8.3f} {ds[harm].sum()/N:+10.4f} | "
          f"{useful.sum():9d} {ds[useful].sum()/N:+7.4f}")
    waste[t] = dict(sel=sd, nochg=nochg, harm=harm, freed=w_no + w_hm)

# same table but in EB-expected terms (a "no-change" upgrade defined by E[p])
print("\n  ... same, judged on EB expected p (|dp|<0.02 = no change):")
for t in TIERS:
    sd = C[f"sel_d_{t}"]
    up = sd > 0
    dp = PHAT[IDX, sd] - PHAT[:, 0]
    nochg = up & (np.abs(dp) < 0.02)
    harm = up & (dp <= -0.02)
    w_no = (dv.cost[nochg, sd[nochg]] - dv.cost[nochg, 0]).sum() / L
    w_hm = (dv.cost[harm, sd[harm]] - dv.cost[harm, 0]).sum() / L
    print(f"  {t:9s} no-change n={nochg.sum():4d} budget={w_no:6.3f}L | "
          f"harmful n={harm.sum():4d} budget={w_hm:6.3f}L dE[p]={dp[harm].sum()/N:+.4f} | "
          f"useful n={(up&(dp>=0.02)).sum():4d} dE[p]={dp[up&(dp>=0.02)].sum()/N:+.4f}")

# ---------------------------------------------------------------- what could the freed budget buy
print("\n=== if the wasted budget were re-spent optimally elsewhere (oracle-greedy upper bound) ===")
for t in TIERS:
    sd = C[f"sel_d_{t}"].copy()
    nochg, harm = waste[t]["nochg"], waste[t]["harm"]
    kill = nochg | harm
    sd2 = sd.copy(); sd2[kill] = 0
    freed = (dv.cost[IDX, sd] - dv.cost[IDX, sd2]).sum() / L
    lost = (dv.score[IDX, sd] - dv.score[IDX, sd2]).sum() / N     # >=0 only from harmful (negative)
    # greedy: among items NOT killed, buy the best true-efficiency upgrade with the freed budget
    cands = []
    for i in range(N):
        if kill[i]:
            continue
        for j in range(3):
            if j <= sd[i]:
                continue
            dcst = dv.cost[i, j] - dv.cost[i, sd[i]]
            dsc = dv.score[i, j] - dv.score[i, sd[i]]
            dp = PHAT[i, j] - PHAT[i, sd[i]]
            if dcst > 0 and dsc > 0:
                cands.append((dsc / dcst, i, j, dcst, dsc, dp))
    cands.sort(reverse=True)
    budget = freed * L
    gain = gain_p = 0.0
    used = set(); spent = 0.0
    for eff, i, j, dcst, dsc, dp in cands:
        if i in used or dcst > budget - spent:
            continue
        used.add(i); spent += dcst; gain += dsc; gain_p += dp
    print(f"  {t:9s} freed={freed:6.3f}L (from {int(kill.sum())} items, realised score change "
          f"{-lost:+.4f}) -> greedy re-spend buys {len(used):3d} upgrades, "
          f"realised +{gain/N:.4f}, EB-expected +{gain_p/N:.4f}  [net realised {(-lost+gain/N):+.4f}]")

# ---------------------------------------------------------------- biggest single-item losses
print("\n=== top-20 single-item weighted losses (realised) ===")
tot_loss = np.zeros(N)
tot_eloss = np.zeros(N)
for t in TIERS:
    sd = C[f"sel_d_{t}"]; so = C[f"sel_o_{t}"]
    tot_loss += TIER_WEIGHT[t] * (dv.score[IDX, so] - dv.score[IDX, sd])
    tot_eloss += TIER_WEIGHT[t] * (PHAT[IDX, so] - PHAT[IDX, sd])
order = np.argsort(-tot_loss)[:20]
print(f"{'i':>4s} {'family':14s} {'true s L/M/K':>16s} {'pred s L/M/K':>18s} "
      f"{'costratio M/K':>14s} {'dep f/b/p':>10s} {'wloss':>7s} {'wEBloss':>8s}")
for i in order:
    ts = "/".join(f"{dv.score[i,j]:.2f}" for j in range(3))
    ps_ = "/".join(f"{P['score_premium'][i,j]:.2f}" for j in range(3))
    cr = f"{dv.cost[i,1]/dv.cost[i,0]:.1f}/{dv.cost[i,2]/dv.cost[i,0]:.1f}"
    dep = "".join(SHORT[C[f'sel_d_{t}'][i]] for t in TIERS)
    print(f"{i:4d} {FAM[i]:14s} {ts:>16s} {ps_:>18s} {cr:>14s} {dep:>10s} "
          f"{tot_loss[i]:+7.3f} {tot_eloss[i]:+8.3f}")
