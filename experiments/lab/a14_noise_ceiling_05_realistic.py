# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a14 step 5 -- the honest exchange rate under REALISTIC operating conditions.

Three safety regimes, so the double-optimism of the BRIEF table is visible:
  (S1) true cost, safety 1.0                 -- pure score axis
  (Sdev) deployed cost model, safety tuned on dev (what diag6/BRIEF did)
  (Sboot) deployed cost model, safety chosen by 880-bootstrap EV (E09 protocol,
          the only regime that is honest about the private set)
Evaluation is always the EXPECTED score sum(p_sel)/N with latent p.

Uses an exact-breakpoint allocator: the Lagrangian selection is piecewise
constant in the penalty, so all penalties that matter are the pairwise line
crossings.  This makes bootstrap sweeps ~1000x cheaper and is exact.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_WEIGHT, TIER_MULT, MODEL_IDS, tier_result  # noqa: E402

dv = load_split("dev")
N = len(dv)
IDX = np.arange(N)
CACHE = HERE / "_a14_cache"
D = np.load(CACHE / "pdraws_dev.npz")
PD = D["p"].astype(np.float64)
PHAT = D["phat"].astype(np.float64)
P43 = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
SAFE43 = {"fast": 0.98, "balanced": 0.87, "premium": 0.85}
GMAX = 1600


# ------------------------------------------------------------------ fast exact core
class Alloc:
    """precompute the whole selection path of the Lagrangian allocator."""

    def __init__(self, pred_score, pred_cost, true_cost, eval_score):
        n = pred_score.shape[0]
        ls = pred_cost[:, 0].sum()
        cn = pred_cost / ls
        pens = [0.0]
        for a in range(3):
            for b in range(a + 1, 3):
                dc = cn[:, b] - cn[:, a]
                ds = pred_score[:, b] - pred_score[:, a]
                ok = np.abs(dc) > 1e-15
                x = ds[ok] / dc[ok]
                pens.append(x[(x > 0) & np.isfinite(x)])
        pens = np.unique(np.concatenate([np.atleast_1d(p) for p in pens]))
        if len(pens) > GMAX:
            pens = np.quantile(pens, np.linspace(0, 1, GMAX))
        # evaluate on midpoints between breakpoints (+ one beyond the last)
        grid = np.concatenate([[0.0], 0.5 * (pens[:-1] + pens[1:]), [pens[-1] * 1.5 + 1.0]])
        grid = np.unique(grid)
        U = pred_score[None, :, :] - grid[:, None, None] * cn[None, :, :]
        sel = U.argmax(axis=2)                       # (G, n)
        self.grid = grid
        self.sel = sel
        self.pc = np.take_along_axis(pred_cost, sel.T, 1).T      # (G, n) predicted cost
        self.tc = np.take_along_axis(true_cost, sel.T, 1).T      # (G, n) true cost
        self.ev = np.take_along_axis(eval_score, sel.T, 1).T     # (G, n) evaluated score
        self.pc0 = pred_cost[:, 0]
        self.tc0 = true_cost[:, 0]
        self.ev0 = eval_score[:, 0]
        self.n = n

    def run(self, W, mult, safety):
        """W is (n, B) bootstrap count matrix (columns sum to n).  Returns
        (score (B,), true_ratio (B,), passed (B,))."""
        cap = (self.pc0 @ W) * np.maximum(1.0, mult * safety)     # (B,)
        tot = self.pc @ W                                         # (G, B)
        ok = tot <= cap[None, :]
        # first grid index that fits (grid is increasing penalty -> decreasing cost)
        g = np.argmax(ok, axis=0)
        none = ~ok.any(axis=0)
        g[none] = len(self.grid) - 1
        sc = np.einsum('bn,nb->b', self.ev[g], W) / W.sum(0)
        tr = np.einsum('bn,nb->b', self.tc[g], W) / (self.tc0 @ W)
        if none.any():
            # deployed fallback: everything to light
            sc[none] = (self.ev0 @ W[:, none]) / W[:, none].sum(0)
            tr[none] = 1.0
        passed = tr <= mult + 1e-15
        return sc, tr, passed


def counts_full():
    return np.ones((N, 1))


def counts_boot(B, seed):
    rng = np.random.default_rng(seed)
    W = np.zeros((N, B))
    for b in range(B):
        idx = rng.integers(0, N, N)
        W[:, b] = np.bincount(idx, minlength=N)
    return W


# ------------------------------------------------------------------ validation
def validate():
    print("validation: fast allocator vs labdata.tier_result on the deployed config")
    W1 = counts_full()
    tot = 0.0
    for t in TIERS:
        A = Alloc(P43[f"score_{t}"], P43[f"cost_{t}"], dv.cost, dv.score)
        sc, tr, ps = A.run(W1, TIER_MULT[t], SAFE43[t])
        r = tier_result(P43[f"score_{t}"], P43[f"cost_{t}"], dv, t, SAFE43[t])
        print(f"  {t:9s} fast score={sc[0]:.6f} ratio={tr[0]:.6f} pass={ps[0]}  |  "
              f"ref score={r['score']:.6f} ratio={r['ratio']:.6f} pass={r['passed']}")
        tot += TIER_WEIGHT[t] * (sc[0] if ps[0] else 0.0)
    print(f"  final (fast) = {tot:.6f}   [held-out E43 = 0.7019]")


# ------------------------------------------------------------------ safety regimes
SAF_GRID = np.round(np.arange(0.60, 1.2001, 0.01), 3)


def best_safety_dev(A, mult):
    W1 = counts_full()
    best = (None, -1)
    for s in SAF_GRID:
        sc, tr, ps = A.run(W1, mult, float(s))
        if ps[0] and sc[0] > best[1]:
            best = (float(s), float(sc[0]))
    return best


def best_safety_boot(A, mult, W):
    best = (None, -1, None)
    for s in SAF_GRID:
        sc, tr, ps = A.run(W, mult, float(s))
        ev = float(np.mean(np.where(ps, sc, 0.0)))
        if ev > best[1]:
            best = (float(s), ev, float(np.mean(~ps)))
    return best


def eval_config(score_of_tier, cost_of_tier, eval_score, Wb):
    """returns dict of the three regimes -> (final, detail)."""
    out = {}
    tot1 = tot_dev = tot_boot = 0.0
    det = {}
    for t in TIERS:
        A = Atrue = Alloc(score_of_tier(t), dv.cost, dv.cost, eval_score)
        sc, tr, ps = Atrue.run(counts_full(), TIER_MULT[t], 1.0)
        tot1 += TIER_WEIGHT[t] * (sc[0] if ps[0] else 0.0)
        B = Alloc(score_of_tier(t), cost_of_tier(t), dv.cost, eval_score)
        s_dev, v_dev = best_safety_dev(B, TIER_MULT[t])
        tot_dev += TIER_WEIGHT[t] * v_dev
        s_b, ev_b, bust = best_safety_boot(B, TIER_MULT[t], Wb)
        tot_boot += TIER_WEIGHT[t] * ev_b
        det[t] = dict(s_dev=s_dev, v_dev=v_dev, s_boot=s_b, ev_boot=ev_b, bust=bust)
    out["S1"] = tot1
    out["Sdev"] = tot_dev
    out["Sboot"] = tot_boot
    out["det"] = det
    return out


def make_pred(p, q, rng, rho_err):
    lam = q * q
    mu = p.mean(0); sd = p.std(0)
    e = rng.standard_normal(p.shape)
    if rho_err > 0:
        c = rng.standard_normal((p.shape[0], 1))
        e = np.sqrt(rho_err) * c + np.sqrt(1 - rho_err) * e
    return mu + lam * (p - mu) + np.sqrt(lam * (1 - lam)) * sd * e


def main():
    validate()
    Wb = counts_boot(200, 4242)

    print("\n" + "=" * 104)
    print("CURRENT SYSTEM, honestly evaluated (allocate on E43 predictions, evaluate E[p|k])")
    print("=" * 104)
    cur = eval_config(lambda t: P43[f"score_{t}"], lambda t: P43[f"cost_{t}"], PHAT, Wb)
    print(f"  true cost / safety 1.0           : {cur['S1']:.4f}")
    print(f"  pred cost / dev-tuned safety     : {cur['Sdev']:.4f}   "
          + " ".join(f"{t[:4]}@{cur['det'][t]['s_dev']:.2f}" for t in TIERS))
    print(f"  pred cost / bootstrap-EV safety  : {cur['Sboot']:.4f}   "
          + " ".join(f"{t[:4]}@{cur['det'][t]['s_boot']:.2f}(bust {100*cur['det'][t]['bust']:.1f}%)"
                     for t in TIERS))
    # realised-score version of the same thing, for comparison with the BRIEF
    curR = eval_config(lambda t: P43[f"score_{t}"], lambda t: P43[f"cost_{t}"], dv.score, Wb)
    print(f"  [realised-score eval, same allocs]: S1={curR['S1']:.4f} "
          f"Sdev={curR['Sdev']:.4f} Sboot={curR['Sboot']:.4f}")

    print("\n" + "=" * 104)
    print("HONEST EXCHANGE RATE -- corr(pred, latent p) -> final score, three safety regimes")
    print("=" * 104)
    rng = np.random.default_rng(777)
    QG = [0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00]
    for rho_err in (0.0, 0.6):
        print(f"\n--- cross-model prediction-error correlation = {rho_err} ---")
        print(f"{'corr(p)':>8s} {'corr(s)':>8s} | {'S1 true/1.0':>12s} {'Sdev':>8s} "
              f"{'Sboot':>8s} | {'safety(boot)':>22s}")
        for q in QG:
            r1, rd, rb = [], [], []
            saf = []
            for rep in range(3):
                p = PD[rng.integers(PD.shape[0])]
                pr = make_pred(p, q, rng, rho_err)
                r = eval_config(lambda t: pr, lambda t: P43[f"cost_{t}"], p, Wb)
                r1.append(r["S1"]); rd.append(r["Sdev"]); rb.append(r["Sboot"])
                saf.append([r["det"][t]["s_boot"] for t in TIERS])
            saf = np.mean(saf, axis=0)
            corr_s = q * np.mean([PD.reshape(-1, 3)[:, j].std() / dv.score[:, j].std()
                                  for j in range(3)])
            print(f"{q:8.2f} {corr_s:8.2f} | {np.mean(r1):12.4f} {np.mean(rd):8.4f} "
                  f"{np.mean(rb):8.4f} | {'/'.join(f'{x:.2f}' for x in saf):>22s}")

    print("\n" + "=" * 104)
    print("BRIEF-STYLE (DISHONEST) EXCHANGE RATE, reproduced for comparison:")
    print("blend the E43 predictions toward the REALISED score and evaluate on the realised score")
    print("=" * 104)
    print(f"{'lam':>6s} {'corr(pred,s)':>13s} {'corr(pred,p)':>13s} | {'Sdev':>8s} {'Sboot':>8s}"
          f" | {'honest eval Sdev':>17s} {'honest Sboot':>13s}")
    ratio = np.mean([PD.reshape(-1, 3)[:, j].std() / dv.score[:, j].std() for j in range(3)])
    for lam in (0.0, 0.05, 0.10, 0.15, 0.25, 0.5):
        mk = lambda t, lam=lam: (1 - lam) * P43[f"score_{t}"] + lam * dv.score
        rR = eval_config(mk, lambda t: P43[f"cost_{t}"], dv.score, Wb)
        rH = eval_config(mk, lambda t: P43[f"cost_{t}"], PHAT, Wb)
        cs = np.mean([np.corrcoef(mk("fast")[:, j], dv.score[:, j])[0, 1] for j in range(3)])
        print(f"{lam:6.2f} {cs:13.3f} {cs/ratio:13.3f} | {rR['Sdev']:8.4f} {rR['Sboot']:8.4f}"
              f" | {rH['Sdev']:17.4f} {rH['Sboot']:13.4f}")


if __name__ == "__main__":
    main()
