# SPDX-License-Identifier: Apache-2.0
"""a02-critic #4.  Is the project's own decision rule calibrated correctly?

(a) resolution of the safety knob: how finely can the realised budget ratio
    actually be steered?  (the knob that decides bust / no-bust)
(b) paired vs unpaired noise of the 880-bootstrap EV estimator -- the quantity
    the "gains < 0.0005 are noise / 3 seeds" rule is calibrated against
(c) winner's-curse arithmetic for E43's coordinate-descent sweep
(d) sign test over the rejected small-positive experiments recorded in
    EXPERIMENT_LOG.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from math import erf, sqrt, log

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, TIER_MULT, tier_result

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
N = len(dv)

print("=" * 100)
print("(a)  Resolution of the safety knob on the honest held-out dev predictions")
print("=" * 100)
for t in TIERS:
    print(f"\n  {t}  (cap = {TIER_MULT[t]}x light)")
    prev = None
    rows = []
    for sf in np.arange(0.74, 1.021, 0.005):
        r = tier_result(P[f"score_{t}"], P[f"cost_{t}"], dv, t, float(sf))
        rows.append((float(sf), r["ratio"], r["score"], r["passed"],
                     int((r["sel"] == 2).sum()), int((r["sel"] == 1).sum())))
    # print only the rows around the deployed value and the cliff
    for sf, ratio, sc, ok, nk1, nmid in rows:
        flag = ""
        if abs(sf - SAFE[t]) < 1e-9:
            flag = "  <= DEPLOYED"
        if prev is not None and (prev[3] and not ok):
            flag += "   *** BUST CLIFF ***"
        if abs(sf - SAFE[t]) < 0.031 or (prev is not None and prev[3] != ok):
            print(f"    safety={sf:.3f} realised_ratio={ratio:.3f} score={sc:.4f} "
                  f"pass={str(ok):5s} n_k1={nk1:3d} n_mid={nmid:3d}{flag}")
        prev = (sf, ratio, sc, ok)
    rr = np.array([x[1] for x in rows])
    print(f"    max |d ratio| for one 0.005 safety step: {np.abs(np.diff(rr)).max():.4f}"
          f"   (mean {np.abs(np.diff(rr)).mean():.4f})")
    # where is the cliff?
    ok = np.array([x[3] for x in rows])
    sfs = np.array([x[0] for x in rows])
    if ok.any() and (~ok).any():
        cliff = sfs[ok][-1]
        print(f"    largest safety that still passes on THIS dev sample: {cliff:.3f}"
              f"   (deployed {SAFE[t]:.3f}, margin {cliff-SAFE[t]:+.3f})")

print()
print("=" * 100)
print("(b)  Paired vs unpaired noise of the 880-item bootstrap EV estimator")
print("=" * 100)


def alloc_batch(S, C, cap):
    B, m, _ = S.shape
    L = C[:, :, 0].sum(1)

    def choose(pen):
        return (S - pen[:, None, None] * C / L[:, None, None]).argmax(2)

    pen0 = np.zeros(B)
    sel = choose(pen0)
    tot = np.take_along_axis(C, sel[:, :, None], 2)[:, :, 0].sum(1)
    over = tot > cap
    if not over.any():
        return sel
    low, high = np.zeros(B), np.ones(B)
    for _ in range(24):
        s2 = choose(high)
        t2 = np.take_along_axis(C, s2[:, :, None], 2)[:, :, 0].sum(1)
        bad = (t2 > cap) & over
        if not bad.any():
            break
        low = np.where(bad, high, low)
        high = np.where(bad, high * 2.0, high)
    for _ in range(30):
        mid = (low + high) / 2.0
        cand = choose(mid)
        ct = np.take_along_axis(C, cand[:, :, None], 2)[:, :, 0].sum(1)
        okm = ct <= cap
        high = np.where(okm, mid, high)
        low = np.where(okm, low, mid)
    sf_ = choose(high)
    ct = np.take_along_axis(C, sf_[:, :, None], 2)[:, :, 0].sum(1)
    sel = np.where(over[:, None], sf_, sel)
    tot = np.where(over, ct, tot)
    give = tot > cap
    if give.any():
        sel = np.where(give[:, None], 0, sel)
    return sel


def boot_ev(idx, transform=None, safety=SAFE, want_bust=False):
    tot = 0.0
    bust = {}
    for t in TIERS:
        S, C = P[f"score_{t}"].copy(), P[f"cost_{t}"].copy()
        if transform is not None:
            S, C = transform(S, C)
        Sb, Cb = S[idx], C[idx]
        ts, tc = dv.score[idx], dv.cost[idx]
        cap = Cb[:, :, 0].sum(1) * max(1.0, TIER_MULT[t] * safety[t])
        sel = alloc_batch(Sb, Cb, cap)
        got = np.take_along_axis(ts, sel[:, :, None], 2)[:, :, 0].mean(1)
        cst = np.take_along_axis(tc, sel[:, :, None], 2)[:, :, 0].sum(1)
        ok = (cst / tc[:, :, 0].sum(1)) <= TIER_MULT[t] + 1e-15
        tot = tot + TIER_WEIGHT[t] * got * ok
        bust[t] = 1.0 - ok.mean()
    return (tot, bust) if want_bust else tot


def shrink_to_fam(S, w):
    import sys as _s
    return S


VARIANTS = {
    "baseline": None,
    "gain x1.05 (uniform gain rescale)": lambda S, C: (S[:, [0]] + 1.05 * (S - S[:, [0]]), C),
    "gain x1.50 (uniform gain rescale)": lambda S, C: (S[:, [0]] + 1.50 * (S - S[:, [0]]), C),
    "k1 gain +0.002 only": lambda S, C: (S + np.array([0.0, 0.0, 0.002])[None, :], C),
    "mid gain +0.002 only": lambda S, C: (S + np.array([0.0, 0.002, 0.0])[None, :], C),
    "k1 price +1%": lambda S, C: (S, C * np.array([1.0, 1.0, 1.01])[None, :]),
    "k1 price +10%": lambda S, C: (S, C * np.array([1.0, 1.0, 1.10])[None, :]),
}
NB = 400
seeds = [7, 17, 23, 31, 43, 57]
res = {k: [] for k in VARIANTS}
for sd in seeds:
    rng = np.random.default_rng(sd)
    idx = rng.integers(0, N, size=(NB, N))
    for k_, tf in VARIANTS.items():
        res[k_].append(boot_ev(idx, tf))

print(f"  {NB} resamples of 880, {len(seeds)} bootstrap seeds\n")
print(f"  {'variant':16s} {'EV mean':>9s} {'sd ACROSS seeds':>16s} "
      f"{'paired delta vs base':>21s} {'sd of paired delta':>19s}")
base_seed_means = np.array([v.mean() for v in res["baseline"]])
for k_, arr in res.items():
    sm = np.array([v.mean() for v in arr])
    d = sm - base_seed_means
    print(f"  {k_:16s} {sm.mean():9.4f} {sm.std(ddof=1):16.4f} "
          f"{d.mean():+21.4f} {d.std(ddof=1):19.4f}")
sig_ref = (np.array([v.mean() for v in res["k1 gain +0.002 only"]]) - base_seed_means).std(ddof=1)
print(f"\n  ratio  sd(across seeds, unpaired) / sd(paired delta for a small change) = "
      f"{base_seed_means.std(ddof=1) / max(1e-9, sig_ref):.1f}x")
print("  => a rule that compares a PAIRED mean difference against the UNPAIRED")
print("     seed spread is that many times too strict.")
print("  NOTE: 'gain xC' is EXACTLY a no-op.  util_j = s_j - pen*c_j/L and the bisection")
print("  solves for pen, so replacing (s_j - s_0) by g*(s_j - s_0) is identical to")
print("  replacing pen by pen/g -> the selection is invariant.  Any hyper-parameter that")
print("  only rescales all gains uniformly (GAIN_ALPHA, RANK_BETA, tier blend, in their")
print("  scale component) cannot change a single decision while the budget binds.")

# within-seed Monte-Carlo error of the EV mean itself
v0 = res["baseline"][0]
print(f"\n  within-one-seed MC standard error of EV (400 resamples) = {v0.std(ddof=1)/np.sqrt(NB):.4f}")
print(f"  spread of the 880-item bootstrap distribution itself      = {v0.std(ddof=1):.4f}")

print("\n  per-tier BUST probability of the deployed constants on 880-item resamples of dev:")
rng = np.random.default_rng(7)
idx = rng.integers(0, N, size=(2000, N))
_, bust = boot_ev(idx, None, SAFE, want_bust=True)
for t in TIERS:
    print(f"    {t:9s} P(over budget) = {bust[t]*100:.1f}%")
print("  (E39 reported 0.0% / 0.2% / 0.2% -- measured on the CV prediction set over 2,640,")
print("   not on this held-out prediction set.)")

print()
print("=" * 100)
print("(c)  Winner's curse arithmetic for E43")
print("=" * 100)
sig = sig_ref
print(f"  measured paired sd of a single EV comparison, per seed: sigma = {sig:.5f}")
print(f"  E43 evaluated: 11 'expensive' combos, then 2 rounds x 8 constants x ~5 values")
print(f"  = about {11 + 2*8*5} candidate evaluations, all on the same folds.")


def emax(K):
    # expected max of K iid N(0,1), Blom approximation
    return (2 * log(K)) ** 0.5 - (log(log(K)) + log(4 * np.pi)) / (2 * (2 * log(K)) ** 0.5)


for K in (11, 40, 91, 200):
    print(f"    K={K:4d}  E[max of K iid N(0,1)] = {emax(K):.2f}"
          f"  -> expected upward bias of the winner if all candidates were equal: "
          f"{emax(K)*sig:+.4f}")
print(f"\n  E43 reported CV gain +0.0040 (3 seeds x 400) and held-out gain +0.0019.")
print(f"  shrinkage 0.0021 = {0.0021/0.0040*100:.0f}% of the CV gain, consistent with a")
print(f"  winner's-curse bias of roughly {emax(91)*sig:.4f} on top of a real ~0.0019.")

print()
print("=" * 100)
print("(d)  Sign test over the small-positive results that the project REJECTED")
print("=" * 100)
# values transcribed from EXPERIMENT_LOG.md; the arithmetic below is computed here.
rejected = [
    ("E13 DM-module prior",      +0.0004, 1, "coverage 10.8% only"),
    ("E15 word kNN scale .15",   +0.0002, 1, "char kNN subsumes"),
    ("E23b comp1024+ordinal",    +0.0003, 1, "artifact size"),
    ("E36 avg(T k.5, P)",        +0.0007, 1, "'same size as E27' -> no 3-seed run"),
    ("E37 conformal cost a=.5",  +0.0006, 1, "monotone decay for larger alpha"),
    ("E40 K4 ridge-only",        +0.0007, 3, "non-unimodal in K; vs UNPAIRED seed sd"),
    ("E31 ratio+smear, fast",    +0.0011, 1, "premium lost -0.0066 in the same config"),
    ("E41 aux ridge+GBM score",  +0.0005, 1, "single seed"),
    ("E42 lookup features",      +0.0000, 3, "mean 0.0000 - correctly rejected"),
]
adopted_small = [("E27 rank head b=.25", +0.00063, 3), ("E21 ordinal", +0.0013, 3),
                 ("E20 kNN k16", +0.0008, 2)]
pos = [r for r in rejected if r[1] > 0.0002]
print(f"  rejected experiments with a positive point estimate > +0.0002: {len(pos)}/{len(rejected)}")
tot = sum(r[1] for r in pos)
print(f"  sum of their point estimates: {tot:+.4f}")
p_all_pos = 0.5 ** len(pos)
print(f"  if every one were pure noise, P(all {len(pos)} positive) = {p_all_pos:.4f}")
print(f"  smallest ADOPTED effect: E27 rank head, paired 3-seed mean {adopted_small[0][1]:+.5f}")
print(f"  largest  REJECTED effect: E31 fast-tier {max(r[1] for r in rejected):+.5f}"
      f"  and E40/E36 at +0.0007")
print("  => the accept/reject boundary is not a threshold, it is a coin flip at ~+0.0006.")
