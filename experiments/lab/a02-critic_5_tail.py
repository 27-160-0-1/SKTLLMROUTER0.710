# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a02-critic #5.  The budget risk is NOT 'broad cost error' -- it is a handful of
episodes, and the premium bust boundary is one episode wide.

Also: the noise floor of the project's decision rule is a compute choice.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, TIER_MULT, tier_result
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
N = len(dv)
fam = np.array([classify_family(t) for t in dv.texts])
L = dv.cost[:, 0].sum()

print("=" * 100)
print("0.  Which constants are actually deployed?  The BRIEF is self-inconsistent.")
print("=" * 100)
for name, sfd in (("BRIEF section 3 table / diag*.py  (.98/.89/.88)", {"fast": .98, "balanced": .89, "premium": .88}),
                  ("BRIEF section 5 architecture box (.98/.87/.85)", {"fast": .98, "balanced": .87, "premium": .85}),
                  ("E39 min-regret insurance         (.985/.875/.81)", {"fast": .985, "balanced": .875, "premium": .81})):
    tot = 0.0
    parts = []
    for t in TIERS:
        r = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, sfd[t])
        tot += TIER_WEIGHT[t] * r["tier_score"]
        parts.append(f"{t[:4]}={r['score']:.4f}/r{r['ratio']:.3f}{'' if r['passed'] else '!'}")
    print(f"  {name:48s} final={tot:.4f}   " + " ".join(parts))

print()
print("=" * 100)
print("1.  Concentration of the premium budget in single episodes")
print("=" * 100)
c_k1 = dv.cost[:, 2]
share = c_k1 / L
o = np.argsort(-share)
print(f"  light-baseline total L = {L:.3f} credits; premium cap = 4L = {4*L:.3f}")
print(f"  {'rank':>4s} {'family':10s} {'true k1 cost':>13s} {'as % of the 4L cap':>20s} {'pred k1 cost':>13s} {'pred/true':>10s}")
for i in o[:12]:
    print(f"  {list(o).index(i)+1:4d} {fam[i]:10s} {c_k1[i]:13.4f} {share[i]/4*100:19.2f}% "
          f"{P['cost_premium'][i,2]:13.4f} {P['cost_premium'][i,2]/c_k1[i]:10.2f}")
print(f"\n  top 1 episode  = {share[o[0]]/4*100:.2f}% of the premium cap")
print(f"  top 5 episodes = {share[o[:5]].sum()/4*100:.2f}% of the premium cap")
print(f"  top 20 episodes= {share[o[:20]].sum()/4*100:.2f}% of the premium cap")
print(f"  the deployed premium run leaves {(4.0-3.654)*L:.3f} credits of slack "
      f"= {(4.0-3.654)/share[o[0]]:.1f} of the single most expensive episode")

print("\n  how many episodes cost more (in k1) than the remaining slack at safety .88?")
slack = (4.0 - 3.654) * L
print(f"  slack = {slack:.3f} credits; episodes with true k1 cost > slack: "
      f"{int((c_k1 > slack).sum())} of {N}")

print()
print("=" * 100)
print("2.  Tail-cap: forbid k1 for episodes whose PREDICTED k1 cost exceeds a quantile,")
print("    then re-tune safety.  (E08 rejected per-item caps in 2026-08-13, before the")
print("    current cost model, on the grounds that 'budget variance comes from broad")
print("    cost error, not a few items'.  Section 1 above contradicts that premise.)")
print("=" * 100)


def run_cap(q, tier, sgrid):
    """cap: predicted k1 cost above quantile q -> k1 made ineligible (price = +inf)."""
    S = P[f"score_{tier}"].copy()
    C = P[f"cost_{tier}"].copy()
    if q < 1.0:
        thr = np.quantile(C[:, 2], q)
        ban = C[:, 2] > thr
        C = C.copy()
        C[ban, 2] = 1e9
    best = None
    for sf in sgrid:
        r = tier_result(S, C, dv, tier, float(sf))
        if r["passed"] and (best is None or r["score"] > best[0]):
            best = (r["score"], float(sf), r["ratio"], int((r["sel"] == 2).sum()))
    return best


SG = np.arange(0.70, 1.401, 0.005)
print(f"  {'q':>6s} " + "  ".join(f"{t:>26s}" for t in TIERS) + "   weighted")
for q in (1.0, 0.995, 0.99, 0.98, 0.97, 0.95, 0.90):
    row = []
    tot = 0.0
    for t in TIERS:
        b = run_cap(q, t, SG)
        row.append(f"{b[0]:.4f}@sf{b[1]:.3f} r{b[2]:.2f} k1={b[3]:3d}")
        tot += TIER_WEIGHT[t] * b[0]
    print(f"  {q:6.3f} " + "  ".join(f"{r:>26s}" for r in row) + f"   {tot:.4f}")
print("  (safety re-tuned per row on dev = oracle-tuned; the comparison across rows is")
print("   what matters, and the honest cross-fitted version is in section 3.)")

print("\n  smoothness of the realised-ratio curve (premium), max jump per 0.005 safety step:")
for q in (1.0, 0.99, 0.98, 0.95):
    C = P["cost_premium"].copy()
    if q < 1.0:
        C[C[:, 2] > np.quantile(C[:, 2], q), 2] = 1e9
    rr = np.array([tier_result(P["score_premium"], C, dv, "premium", float(sf))["ratio"]
                   for sf in np.arange(0.74, 1.20, 0.005)])
    print(f"    q={q:.3f}  max jump {np.abs(np.diff(rr)).max():.4f}   mean {np.abs(np.diff(rr)).mean():.4f}")

print()
print("=" * 100)
print("3.  Honest cross-fitted value of the tail cap (half-split of dev, 8 splits x 2)")
print("=" * 100)


def alloc(S, C, mult, safety):
    Lp = C[:, 0].sum()
    cap = Lp * max(1.0, mult * safety)

    def choose(pen):
        return (S - pen * C / Lp).argmax(1)
    sel = choose(0.0)
    tot = C[np.arange(len(sel)), sel].sum()
    if tot > cap:
        lo, hi = 0.0, 1.0
        sel = choose(hi)
        while C[np.arange(len(sel)), sel].sum() > cap and hi < 2 ** 40:
            lo, hi = hi, hi * 2
            sel = choose(hi)
        for _ in range(40):
            mid = (lo + hi) / 2
            cand = choose(mid)
            if C[np.arange(len(cand)), cand].sum() <= cap:
                hi, sel = mid, cand
            else:
                lo = mid
        if C[np.arange(len(sel)), sel].sum() > cap:
            sel = np.zeros(len(S), dtype=int)
    return sel


def eval_pool(pool, q, safety, tier):
    S = P[f"score_{tier}"][pool]
    C = P[f"cost_{tier}"][pool].copy()
    if q < 1.0:
        C[C[:, 2] > np.quantile(P[f"cost_{tier}"][:, 2], q), 2] = 1e9
    sel = alloc(S, C, TIER_MULT[tier], safety)
    ts, tc = dv.score[pool], dv.cost[pool]
    ar = np.arange(len(pool))
    ratio = tc[ar, sel].sum() / tc[:, 0].sum()
    ok = ratio <= TIER_MULT[tier] + 1e-15
    return (ts[ar, sel].mean() if ok else 0.0), ratio


def boot_ev_pool(pool, q, safety, tier, idx):
    S = P[f"score_{tier}"].copy()
    C = P[f"cost_{tier}"].copy()
    if q < 1.0:
        C[C[:, 2] > np.quantile(C[:, 2], q), 2] = 1e9
    vals = []
    for b in range(idx.shape[0]):
        p2 = pool[idx[b]]
        sel = alloc(S[p2], C[p2], TIER_MULT[tier], safety)
        ar = np.arange(len(p2))
        ts, tc = dv.score[p2], dv.cost[p2]
        ratio = tc[ar, sel].sum() / tc[:, 0].sum()
        vals.append(ts[ar, sel].mean() if ratio <= TIER_MULT[tier] + 1e-15 else 0.0)
    return float(np.mean(vals))


def strat_split(rng):
    a, b = [], []
    for f in set(fam):
        ii = np.where(fam == f)[0]
        rng.shuffle(ii)
        h = len(ii) // 2
        a.extend(ii[:h]); b.extend(ii[h:])
    return np.array(sorted(a)), np.array(sorted(b))


QS = (1.0, 0.99, 0.98, 0.95)
SG2 = np.arange(0.74, 1.30, 0.01)
NB = 60
out = {q: [] for q in QS}
for sp in range(8):
    rng = np.random.default_rng(500 + sp)
    A, B = strat_split(rng)
    for fitp, evp in ((A, B), (B, A)):
        bidx = np.random.default_rng(9 + sp).integers(0, len(fitp), size=(NB, len(fitp)))
        for q in QS:
            tot = 0.0
            for t in TIERS:
                sf = max(SG2, key=lambda s: boot_ev_pool(fitp, q, float(s), t, bidx))
                v, _ = eval_pool(evp, q, sf, t)
                tot += TIER_WEIGHT[t] * v
            out[q].append(tot)
print(f"  {'q (k1 ban above pred-cost quantile)':38s} {'held-out half':>13s} {'delta':>8s}")
base = np.mean(out[1.0])
for q in QS:
    v = np.array(out[q])
    print(f"  {q:38.3f} {v.mean():13.4f} {v.mean()-base:+8.4f}"
          f"   (paired wins {int((v - np.array(out[1.0]) > 0).sum())}/{len(v)},"
          f" paired sd {np.std(v-np.array(out[1.0]),ddof=1)/np.sqrt(len(v)):.4f})")

print()
print("=" * 100)
print("4.  The noise floor is a compute choice: paired sd vs number of bootstrap resamples")
print("=" * 100)


def boot_ev_full(idx, cmul, safety):
    tot = 0.0
    for t in TIERS:
        S = P[f"score_{t}"]
        C = P[f"cost_{t}"] * np.array([1.0, 1.0, cmul])[None, :]
        vals = []
        for b in range(idx.shape[0]):
            p2 = idx[b]
            sel = alloc(S[p2], C[p2], TIER_MULT[t], safety[t])
            ar = np.arange(len(p2))
            ts, tc = dv.score[p2], dv.cost[p2]
            ratio = tc[ar, sel].sum() / tc[:, 0].sum()
            vals.append(ts[ar, sel].mean() if ratio <= TIER_MULT[t] + 1e-15 else 0.0)
        tot += TIER_WEIGHT[t] * np.mean(vals)
    return tot


SAFE = {"fast": .98, "balanced": .89, "premium": .88}
print(f"  {'B (resamples)':>14s} {'sd of paired delta over 6 seeds':>32s}  {'sqrt-law prediction':>20s}")
ref = None
for B in (100, 400, 1600):
    ds = []
    for sd in (7, 17, 23, 31, 43, 57):
        idx = np.random.default_rng(sd).integers(0, N, size=(B, N))
        ds.append(boot_ev_full(idx, 1.01, SAFE) - boot_ev_full(idx, 1.00, SAFE))
    s = np.std(ds, ddof=1)
    if ref is None:
        ref = (B, s)
    print(f"  {B:14d} {s:32.5f}  {ref[1]*np.sqrt(ref[0]/B):20.5f}")
