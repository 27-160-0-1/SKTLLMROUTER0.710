# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a10 P1 — the denominator problem.

The realised budget ratio is exactly

    R = sum_i w_i * r_i,sel      w_i = c_i0 / sum_j c_j0   (normalised light cost)
                                 r_im = c_im / c_i0        (per-item cost multiplier)

and the allocator's constraint is the same expression built from *predicted*
w and r.  This script measures which of the two factors carries the error,
how well the current parameterisation (r_hat = exp(log c_m - log c_0))
already cancels the shared term, and the distribution of the correction
factor K = realised_ratio / predicted_ratio, which is what the safety scalar
must cover.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_MULT, TIER_WEIGHT, tier_result, allocate
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
tr = load_split("train")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}   # E43 deployed
n = len(dv)
idx = np.arange(n)

print("=" * 78)
print("(0) exact cost algebra:  c_m == rate_m * (I + 4*O_m) / 1e6 ?")
RATE_IN = np.array([1.0, 2.127, 6.565])
RATE_OUT = np.array([4.0, 8.509, 26.260])
print("    r_out/r_in per model:", np.round(RATE_OUT / RATE_IN, 6))
for sp in (tr, dv):
    T = sp.itok + 4.0 * sp.otok                      # 'effective tokens' with the 4x convention
    c_hat = RATE_IN[None, :] * T / 1e6
    err = np.abs(c_hat - sp.cost) / np.maximum(sp.cost, 1e-12)
    print(f"    {sp.name}: max rel err of c = rate_in*(I+4O)/1e6 -> {err.max():.2e} "
          f"(mid uses 4.0005, so tiny)")

print()
print("=" * 78)
print("(1) variance decomposition of the realised ratio, per tier")
print("    R = sum_i w_i r_i,sel ; compare the pieces predicted vs true")
fam_dv = np.array([classify_family(t) for t in dv.texts])


def decomp(t):
    ps, pc = P[f"score_{t}"], P[f"cost_{t}"]
    sel = allocate(ps, pc, dv.cost, TIER_MULT[t], SAFE[t])
    w_t = dv.cost[:, 0] / dv.cost[:, 0].sum()
    w_p = pc[:, 0] / pc[:, 0].sum()
    r_t = dv.cost[idx, sel] / dv.cost[:, 0]
    r_p = pc[idx, sel] / pc[:, 0]
    R_true = float((w_t * r_t).sum())
    R_pred = float((w_p * r_p).sum())
    R_wtrue = float((w_t * r_p).sum())      # only w corrected
    R_rtrue = float((w_p * r_t).sum())      # only r corrected
    return dict(sel=sel, R_true=R_true, R_pred=R_pred, R_wtrue=R_wtrue, R_rtrue=R_rtrue,
                w_t=w_t, w_p=w_p, r_t=r_t, r_p=r_p)


print(f"{'tier':9s} {'R_pred':>8s} {'R_true':>8s} {'K=Rt/Rp':>8s} | "
      f"{'w true only':>11s} {'r true only':>11s} | share of log K from w / r")
for t in TIERS:
    d = decomp(t)
    lk = np.log(d["R_true"] / d["R_pred"])
    lw = np.log(d["R_wtrue"] / d["R_pred"])
    lr = np.log(d["R_rtrue"] / d["R_pred"])
    print(f"{t:9s} {d['R_pred']:8.4f} {d['R_true']:8.4f} {np.exp(lk):8.4f} | "
          f"{d['R_wtrue']:11.4f} {d['R_rtrue']:11.4f} | "
          f"{100*lw/lk:5.1f}% / {100*lr/lk:5.1f}%")

print()
print("=" * 78)
print("(2) per-item error of the two parameterisations (premium preds)")
pc = P["cost_premium"]
lc_err = np.log(pc) - np.log(dv.cost)
print("    absolute log-cost error sd per model:", np.round(lc_err.std(0), 4))
print("    corr of the log-cost residuals across models:")
C = np.corrcoef(lc_err.T)
for j, m in enumerate(MODEL_IDS):
    print(f"      {m:11s} " + " ".join(f"{C[j,k]:+.3f}" for k in range(3)))
for j, m in enumerate(MODEL_IDS[1:], start=1):
    r_t = dv.cost[:, j] / dv.cost[:, 0]
    r_p = pc[:, j] / pc[:, 0]
    e = np.log(r_p) - np.log(r_t)
    print(f"    ratio r_{m}/light   log-err  mean={e.mean():+.4f} sd={e.std():.4f}"
          f"   (vs abs {lc_err[:,j].std():.4f});  var reduction {1-e.var()/lc_err[:,j].var():+.3f}")
w_t = dv.cost[:, 0] / dv.cost[:, 0].sum()
w_p = pc[:, 0] / pc[:, 0].sum()
ew = np.log(w_p) - np.log(w_t)
print(f"    weight w (normalised light) log-err mean={ew.mean():+.4f} sd={ew.std():.4f}"
      f"   (vs abs light {lc_err[:,0].std():.4f})")

print()
print("=" * 78)
print("(3) how much would a PERFECT r (per-item ratio) or PERFECT w buy?")
print("    substitute the true quantity into the allocator's cost matrix,")
print("    keeping the other one predicted, then re-tune safety on dev (oracle-tuned).")


def build_cost(t, use_true_w=False, use_true_r=False):
    pcm = P[f"cost_{t}"].copy()
    w = dv.cost[:, 0] if use_true_w else pcm[:, 0]
    if use_true_r:
        r = dv.cost / dv.cost[:, :1]
    else:
        r = pcm / pcm[:, :1]
    return w[:, None] * r


def best_safety(t, Cm, S=None, grid=np.arange(0.40, 1.601, 0.005)):
    S = P[f"score_{t}"] if S is None else S
    best = None
    for sf in grid:
        r = tier_result(S, Cm, dv, t, float(sf))
        if r["passed"] and (best is None or r["score"] > best[1]):
            best = (float(sf), r["score"], r["ratio"])
    return best


rows = [("deployed  (w_hat, r_hat)", False, False),
        ("true w only", True, False),
        ("true r only", False, True),
        ("true both  (= true cost)", True, True)]
for label, uw, ur in rows:
    tot_dep = 0.0
    tot_best = 0.0
    parts = []
    for t in TIERS:
        Cm = build_cost(t, uw, ur)
        rr = tier_result(P[f"score_{t}"], Cm, dv, t, SAFE[t])
        tot_dep += TIER_WEIGHT[t] * rr["tier_score"]
        b = best_safety(t, Cm)
        tot_best += TIER_WEIGHT[t] * b[1]
        parts.append(f"{t[:4]}={b[1]:.4f}@{b[0]:.3f}")
    print(f"  {label:26s} at deployed safety={tot_dep:.4f}   oracle-safety={tot_best:.4f}   "
          + " ".join(parts))

print()
print("=" * 78)
print("(4) bootstrap distribution of K = realised/predicted ratio (880 draws w/ replacement)")
print("    the safety scalar must satisfy  safety <= 1/quantile_hi(K)")
rng = np.random.default_rng(7)
B = 400
for t in TIERS:
    ps, pcm = P[f"score_{t}"], P[f"cost_{t}"]
    Ks = np.empty(B)
    for b in range(B):
        s = rng.integers(0, n, n)
        sel = allocate(ps[s], pcm[s], dv.cost[s], TIER_MULT[t], SAFE[t])
        Rp = pcm[s][np.arange(n), sel].sum() / pcm[s][:, 0].sum()
        Rt = dv.cost[s][np.arange(n), sel].sum() / dv.cost[s][:, 0].sum()
        Ks[b] = Rt / Rp
    print(f"  {t:9s} K: mean={Ks.mean():.4f} sd={Ks.std():.4f} "
          f"q50={np.percentile(Ks,50):.4f} q95={np.percentile(Ks,95):.4f} "
          f"q99={np.percentile(Ks,99):.4f} max={Ks.max():.4f}  "
          f"=> max safety 1/q99 = {1/np.percentile(Ks,99):.4f} (deployed {SAFE[t]})")

print()
print("=" * 78)
print("(5) which items does the allocator upgrade, by family (premium & fast)")
for t in ("fast", "premium"):
    sel = allocate(P[f"score_{t}"], P[f"cost_{t}"], dv.cost, TIER_MULT[t], SAFE[t])
    print(f"  --- {t}: counts per family x model, and realised score/cost effect")
    print(f"      {'family':14s} {'n':>4s} {'light':>5s} {'mid':>4s} {'k1':>4s} "
          f"{'d_score':>8s} {'d_cost_share':>12s}")
    for f in sorted(set(fam_dv)):
        m = fam_dv == f
        cnt = [int(((sel == j) & m).sum()) for j in range(3)]
        ds = float((dv.score[idx, sel][m] - dv.score[m, 0]).sum() / n)
        dc = float((dv.cost[idx, sel][m] - dv.cost[m, 0]).sum() / dv.cost[:, 0].sum())
        print(f"      {f:14s} {int(m.sum()):4d} {cnt[0]:5d} {cnt[1]:4d} {cnt[2]:4d} "
              f"{ds:+8.4f} {dc:+12.4f}")
