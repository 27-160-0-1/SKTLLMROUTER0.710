# SPDX-License-Identifier: Apache-2.0
"""a10 P3 — what should be fed to the allocator?

Four preprocessing designs, all evaluated with the SAME deployed score/cost
predictions (reports/lab/dev_preds_e43.npz) so that only the allocator input
changes:

  D1  LEDGER/UTILITY DECOUPLING.  The Lagrangian uses pred_cost twice:
      (a) inside the utility, where only the *ordering* of s/c matters, and
      (b) in the budget ledger, where only the *sum* matters.  Feed a
      bias-corrected (Duan-smeared, per family x model) cost to the ledger and
      the raw cost to the utility.  E32 did the opposite (decision-only
      inflation); E36-F smeared BOTH.  This cell is untested.
  D2  FAMILY CLAMP.  Hard-clamp families whose upgrade is never worth it.
  D3  TIER-RESTRICTED ACTION SET.  fast can only ever afford {light,mid}.
  D4  TWO-LEVEL (family budget then within-family) allocation.

Plus a variance decomposition of K = realised_ratio / predicted_ratio.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_MULT, TIER_WEIGHT
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
tr = load_split("train")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
n = len(dv)
famdv = np.array([classify_family(t) for t in dv.texts])
famtr = np.array([classify_family(t) for t in tr.texts])
fams = sorted(set(famtr) | set(famdv))
fid = np.array([fams.index(f) for f in famdv])


# ------------------------------------------------------------------ allocator
def allocate2(pred_score, util_cost, ledger_cost, mult, safety, mask=None):
    """Lagrangian bisection with SEPARATE utility cost and ledger cost.

    mask: optional (n,3) bool of allowed actions (False = forbidden).
    With ledger_cost is util_cost and mask None this is exactly labdata.allocate.
    """
    N = len(pred_score)
    ar = np.arange(N)
    L = ledger_cost[:, 0].sum()
    cap = L * max(1.0, mult * safety)
    U0 = pred_score.copy()
    if mask is not None:
        U0 = np.where(mask, U0, -1e18)

    def choose(pen):
        util = U0 - pen * util_cost / util_cost[:, 0].sum()
        return util.argmax(axis=1)

    sel = choose(0.0)
    tot = ledger_cost[ar, sel].sum()
    if tot > cap:
        low, high = 0.0, 1.0
        sel = choose(high)
        tot = ledger_cost[ar, sel].sum()
        while tot > cap and high < 2 ** 60:
            low, high = high, high * 2.0
            sel = choose(high)
            tot = ledger_cost[ar, sel].sum()
        for _ in range(40):
            mid = (low + high) / 2.0
            cand = choose(mid)
            ct = ledger_cost[ar, cand].sum()
            if ct <= cap:
                high, sel, tot = mid, cand, ct
            else:
                low = mid
    if tot > cap:
        sel = np.zeros(N, dtype=int)
    return sel


def score_sel(sel, split, rows, tier):
    ar = np.arange(len(rows))
    tc = split.cost[rows]
    ts = split.score[rows]
    ratio = tc[ar, sel].sum() / tc[:, 0].sum()
    ok = ratio <= TIER_MULT[tier] + 1e-15
    sc = ts[ar, sel].mean()
    return float(sc), float(ratio), bool(ok)


def dev_final(make_inputs, safety=None):
    """make_inputs(tier) -> (score, util_cost, ledger_cost, mask)"""
    safety = SAFE if safety is None else safety
    tot, parts = 0.0, []
    rows = np.arange(n)
    for t in TIERS:
        s, uc, lc, mk = make_inputs(t)
        sel = allocate2(s, uc, lc, TIER_MULT[t], safety[t], mk)
        sc, ra, ok = score_sel(sel, dv, rows, t)
        tot += TIER_WEIGHT[t] * (sc if ok else 0.0)
        parts.append(f"{t[:4]}={sc:.4f}/r{ra:.3f}{'' if ok else '!BUST'}")
    return tot, parts


def boot_ev(make_inputs, safety, B=300, seed=7):
    """880-size bootstrap EV, the project's decision criterion."""
    rng = np.random.default_rng(seed)
    acc = {t: [] for t in TIERS}
    bust = {t: 0 for t in TIERS}
    for b in range(B):
        rows = rng.integers(0, n, n)
        for t in TIERS:
            s, uc, lc, mk = make_inputs(t)
            sel = allocate2(s[rows], uc[rows], lc[rows], TIER_MULT[t], safety[t],
                            None if mk is None else mk[rows])
            sc, ra, ok = score_sel(sel, dv, rows, t)
            acc[t].append(sc if ok else 0.0)
            bust[t] += 0 if ok else 1
    ev = {t: float(np.mean(acc[t])) for t in TIERS}
    ev["final"] = sum(TIER_WEIGHT[t] * ev[t] for t in TIERS)
    ev["bust"] = {t: bust[t] / B for t in TIERS}
    return ev


def best_safety_ev(make_inputs, grid=np.arange(0.60, 1.201, 0.01), B=200, seed=7,
                   tiers=TIERS):
    """per-tier safety maximising bootstrap EV."""
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, n, n) for _ in range(B)]
    out = {}
    for t in tiers:
        s, uc, lc, mk = make_inputs(t)
        best = None
        for sf in grid:
            vals = []
            nb = 0
            for rows in boots:
                sel = allocate2(s[rows], uc[rows], lc[rows], TIER_MULT[t], float(sf),
                                None if mk is None else mk[rows])
                sc, ra, ok = score_sel(sel, dv, rows, t)
                vals.append(sc if ok else 0.0)
                nb += 0 if ok else 1
            ev = float(np.mean(vals))
            if best is None or ev > best[1]:
                best = (float(sf), ev, nb / B)
        out[t] = best
    return out


BASE = lambda t: (P[f"score_{t}"], P[f"cost_{t}"], P[f"cost_{t}"], None)

print("=" * 80)
print("BASELINE (deployed E43 inputs, safety .98/.87/.85)")
tot, parts = dev_final(BASE)
print(f"  dev final = {tot:.4f}   " + " ".join(parts))
ev = boot_ev(BASE, SAFE)
print(f"  bootstrap EV = {ev['final']:.4f}  "
      f"(fast {ev['fast']:.4f} bal {ev['balanced']:.4f} prem {ev['premium']:.4f})  "
      f"bust {ev['bust']}")

print()
print("=" * 80)
print("(A) variance decomposition of K = realised/predicted budget ratio")
print("    K = [sum c_sel / sum chat_sel] / [sum c_0 / sum chat_0]")
rng = np.random.default_rng(11)
for t in TIERS:
    s, uc, lc, _ = BASE(t)
    sel0 = allocate2(s, uc, lc, TIER_MULT[t], SAFE[t])
    # family-level calibration factors of the DEPLOYED head at the deployed selection
    num_f = np.zeros(len(fams)); den_f = np.zeros(len(fams))
    nump = np.zeros(len(fams)); denp = np.zeros(len(fams))
    for k in range(len(fams)):
        m = fid == k
        num_f[k] = dv.cost[np.arange(n), sel0][m].sum()
        nump[k] = uc[np.arange(n), sel0][m].sum()
        den_f[k] = dv.cost[m, 0].sum()
        denp[k] = uc[m, 0].sum()
    print(f"  {t}: per-family (true/pred) of the SELECTED cost, and of the light cost")
    for k, f in enumerate(fams):
        print(f"      {f:14s} sel {num_f[k]/max(nump[k],1e-12):6.3f}   "
              f"light {den_f[k]/max(denp[k],1e-12):6.3f}   "
              f"share of premium ledger {nump[k]/nump.sum():5.3f}")
    break  # only fast for brevity; premium printed below
for t in ("premium",):
    s, uc, lc, _ = BASE(t)
    sel0 = allocate2(s, uc, lc, TIER_MULT[t], SAFE[t])
    print(f"  {t}: same table")
    for k, f in enumerate(fams):
        m = fid == k
        a = dv.cost[np.arange(n), sel0][m].sum(); b = uc[np.arange(n), sel0][m].sum()
        c = dv.cost[m, 0].sum(); d = uc[m, 0].sum()
        print(f"      {f:14s} sel {a/b:6.3f}   light {c/d:6.3f}   share {b/uc[np.arange(n),sel0].sum():5.3f}")

print()
print("  concentration: fraction of the true premium spend from the top-k items")
s, uc, lc, _ = BASE("premium")
sel0 = allocate2(s, uc, lc, TIER_MULT["premium"], SAFE["premium"])
cc = np.sort(dv.cost[np.arange(n), sel0])[::-1]
tot_c = cc.sum()
for k in (1, 5, 10, 25, 50, 100):
    print(f"      top {k:3d}: {cc[:k].sum()/tot_c:6.3f}")

print()
print("=" * 80)
print("(D1) LEDGER/UTILITY DECOUPLING — Duan smearing on the ledger only")
print("     factor[f,m] = mean(true_c / pred_c) over a fitting set; ledger uses")
print("     pred_c * factor, utility keeps raw pred_c.")


def smear_factors(rows_fit, tier, mode="family"):
    pc = P[f"cost_{tier}"]
    fac = np.ones((len(fams), 3))
    if mode == "global":
        g = dv.cost[rows_fit].sum(0) / pc[rows_fit].sum(0)
        return np.tile(g, (len(fams), 1))
    for k in range(len(fams)):
        m = rows_fit[fid[rows_fit] == k]
        if len(m) >= 8:
            fac[k] = dv.cost[m].sum(0) / pc[m].sum(0)
        else:
            fac[k] = dv.cost[rows_fit].sum(0) / pc[rows_fit].sum(0)
    return fac


allrows = np.arange(n)
half = np.arange(n) % 2


def mk_smear(mode, crossfit):
    def f(t):
        pc = P[f"cost_{t}"]
        led = pc.copy()
        if crossfit:
            for h in (0, 1):
                fit = allrows[half == 1 - h]
                fac = smear_factors(fit, t, mode)
                app = allrows[half == h]
                led[app] = pc[app] * fac[fid[app]]
        else:
            fac = smear_factors(allrows, t, mode)
            led = pc * fac[fid]
        return P[f"score_{t}"], pc, led, None
    return f


for mode in ("global", "family"):
    for cf in (True, False):
        mk = mk_smear(mode, cf)
        bs = best_safety_ev(mk, B=200)
        evd = {t: bs[t][0] for t in TIERS}
        tot, parts = dev_final(mk, evd)
        fev = sum(TIER_WEIGHT[t] * bs[t][1] for t in TIERS)
        tag = "cross-fit" if cf else "in-sample"
        print(f"  smear={mode:6s} {tag}: EV={fev:.4f}  safety="
              + "/".join(f"{bs[t][0]:.2f}" for t in TIERS)
              + "  bust=" + "/".join(f"{bs[t][2]:.3f}" for t in TIERS)
              + f"   devfinal@thatsafety={tot:.4f}")
bs0 = best_safety_ev(BASE, B=200)
print("  baseline EV-optimal safety      : EV="
      f"{sum(TIER_WEIGHT[t]*bs0[t][1] for t in TIERS):.4f}  safety="
      + "/".join(f"{bs0[t][0]:.2f}" for t in TIERS)
      + "  bust=" + "/".join(f"{bs0[t][2]:.3f}" for t in TIERS))

print()
print("=" * 80)
print("(D2) FAMILY CLAMP — forbid upgrades for selected families")
for clamp in (["longdoc"], ["hrmcr"], ["longdoc", "hrmcr"], ["longdoc", "hrmcr", "aime"],
              ["longdoc", "hrmcr", "code"]):
    def mk(t, clamp=clamp):
        mask = np.ones((n, 3), bool)
        for f in clamp:
            mask[fid == fams.index(f), 1:] = False
        return P[f"score_{t}"], P[f"cost_{t}"], P[f"cost_{t}"], mask
    tot, parts = dev_final(mk)
    e = boot_ev(mk, SAFE, B=200)
    print(f"  clamp {str(clamp):34s} dev={tot:.4f} EV={e['final']:.4f} "
          f"(fast {e['fast']:.4f} bal {e['balanced']:.4f} prem {e['premium']:.4f})")

print()
print("=" * 80)
print("(D3) TIER-RESTRICTED ACTION SET")
for name, allow in (("fast: {light,mid} only", {"fast": [0, 1]}),
                    ("fast+bal: {light,mid}", {"fast": [0, 1], "balanced": [0, 1]}),
                    ("premium: {light,mid,k1} (base)", {})):
    def mk(t, allow=allow):
        mask = np.ones((n, 3), bool)
        if t in allow:
            mask[:] = False
            mask[:, allow[t]] = True
        return P[f"score_{t}"], P[f"cost_{t}"], P[f"cost_{t}"], mask
    tot, parts = dev_final(mk)
    e = boot_ev(mk, SAFE, B=200)
    print(f"  {name:32s} dev={tot:.4f} EV={e['final']:.4f}  " + " ".join(parts))
