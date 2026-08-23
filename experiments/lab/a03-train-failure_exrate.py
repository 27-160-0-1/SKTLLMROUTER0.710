# SPDX-License-Identifier: Apache-2.0
"""a03: an HONEST exchange rate for score-head quality.

The BRIEF's exchange-rate table blends the predictions toward the *realised*
score and then scores against that same realised score, so part of the gain is
the allocator exploiting the label noise it has been handed.  Here the label
noise is regenerated:

  p_sim  = family mean + rho * (s_dev - family mean)      (latent field,
           rho chosen so var(p_sim) matches the binomial-corrected var(p))
  s_sim  ~ Binomial(num_generations, p_sim) / num_generations   (fresh labels)

A predictor is blended toward p_sim (an honest E[s] improvement) and scored
against s_sim.  The "leaky" control blends toward s_sim itself, reproducing
the BRIEF protocol inside the same simulation.

Costs are untouched: real predicted costs, real true costs.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lab"))

import labdata  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "a03curve", Path(__file__).with_name("a03-train-failure_curve.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

GRID = np.round(np.arange(0.60, 1.0001, 0.01), 3)


def tier_eval(ps, pc, dv, s_eval, tier, safety):
    sel = labdata.allocate(ps, pc, dv.cost, labdata.TIER_MULT[tier], safety)
    idx = np.arange(len(sel))
    ratio = dv.cost[idx, sel].sum() / dv.cost[:, 0].sum()
    ok = ratio <= labdata.TIER_MULT[tier] + 1e-15
    return float(s_eval[idx, sel].mean()) if ok else 0.0


def best_final_on(ps, pc, dv, s_eval):
    tot, per = 0.0, {}
    for t in labdata.TIERS:
        b = max(tier_eval(ps, pc, dv, s_eval, t, float(s)) for s in GRID)
        per[t] = b
        tot += labdata.TIER_WEIGHT[t] * b
    return tot, per


def main(reps=8, seed=11):
    dv = labdata.load_split("dev")
    m = np.load(_c.CACHE / "meta.npz", allow_pickle=True)
    Ydv, ng, fdv = m["Ydv"], m["ngdv"], m["fdv"]
    d = np.load(ROOT / "reports/lab/dev_preds_e43.npz")
    base_s = d["score_fast"].copy()
    base_c = d["cost_fast"].copy()

    # latent field: shrink the realised score toward the family mean so that
    # var(p_sim) matches the binomial-corrected latent variance
    s = Ydv[:, :3]
    fam_mean = np.zeros_like(s)
    for f in set(fdv.tolist()):
        k = fdv == f
        fam_mean[k] = s[k].mean(axis=0)
    dev_ = s - fam_mean
    rho = []
    for j in range(3):
        vw = float(np.mean(dev_[:, j] ** 2))
        noise = float(np.mean(s[:, j] * (1 - s[:, j]) * ng[:, j] / (ng[:, j] - 1) / ng[:, j]))
        rho.append(np.sqrt(max(vw - noise, 0.0) / vw))
    rho = np.asarray(rho)
    p_sim = np.clip(fam_mean + rho * dev_, 0.0, 1.0)
    print("== simulation setup")
    print("   rho (shrink factor) =", np.round(rho, 4))
    print("   var(p_sim) =", np.round(p_sim.var(0), 5),
          " target var(p) =", np.round(s.var(0) - np.asarray(
              [np.mean(s[:, j] * (1 - s[:, j]) * ng[:, j] / (ng[:, j] - 1) / ng[:, j])
               for j in range(3)]), 5))
    print("   corr(base pred, p_sim) =", np.round(
        [np.corrcoef(base_s[:, j], p_sim[:, j])[0, 1] for j in range(3)], 3))

    lams = (0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.00)
    rng = np.random.default_rng(seed)
    res = {("honest", l): [] for l in lams}
    res.update({("leaky", l): [] for l in lams})
    corr = {("honest", l): [] for l in lams}
    corr.update({("leaky", l): [] for l in lams})
    for r in range(reps):
        k = rng.binomial(ng.astype(int), p_sim)
        s_sim = k / ng
        for lam in lams:
            for mode in ("honest", "leaky"):
                tgt = p_sim if mode == "honest" else s_sim
                ps = np.clip((1 - lam) * base_s + lam * tgt, 0.0, 1.0)
                f, _ = best_final_on(ps, base_c, dv, s_sim)
                res[(mode, lam)].append(f)
                corr[(mode, lam)].append([np.corrcoef(ps[:, j], s_sim[:, j])[0, 1]
                                          for j in range(3)])
    print(f"\n== exchange rate, {reps} independent label draws, dev-tuned safety per tier")
    print("  lam  | honest: corr(pred,s_sim)      final   | leaky (BRIEF protocol): corr   final")
    for lam in lams:
        ch = np.mean(corr[("honest", lam)], 0)
        cl = np.mean(corr[("leaky", lam)], 0)
        fh, fl = np.mean(res[("honest", lam)]), np.mean(res[("leaky", lam)])
        sh = np.std(res[("honest", lam)])
        print(f"  {lam:.2f} | {ch[0]:.3f}/{ch[1]:.3f}/{ch[2]:.3f}   {fh:.4f} (sd {sh:.4f}) "
              f"| {cl[0]:.3f}/{cl[1]:.3f}/{cl[2]:.3f}   {fl:.4f}")

    # ---- level-only vs gain-only improvement -------------------------------
    print("\n== where does an honest score improvement have to land?")
    print("   level-only = all three model predictions shifted by the same amount")
    print("   gain-only  = per-item mean kept, the 3-vector shape moved to the truth")
    print("  lam  | level-only: corr   final    | gain-only: corr   final")
    Lp = base_s.mean(axis=1, keepdims=True)
    Lt = p_sim.mean(axis=1, keepdims=True)
    for lam in (0.0, 0.10, 0.25, 0.50, 1.00):
        fs_l, fs_g, c_l, c_g = [], [], [], []
        rng3 = np.random.default_rng(seed)
        for r in range(reps):
            s_sim = rng3.binomial(ng.astype(int), p_sim) / ng
            pl = np.clip(base_s + lam * (Lt - Lp), 0.0, 1.0)
            pg = np.clip(Lp + (1 - lam) * (base_s - Lp) + lam * (p_sim - Lt), 0.0, 1.0)
            fs_l.append(best_final_on(pl, base_c, dv, s_sim)[0])
            fs_g.append(best_final_on(pg, base_c, dv, s_sim)[0])
            c_l.append(np.mean([np.corrcoef(pl[:, j], s_sim[:, j])[0, 1] for j in range(3)]))
            c_g.append(np.mean([np.corrcoef(pg[:, j], s_sim[:, j])[0, 1] for j in range(3)]))
        print(f"  {lam:.2f} | {np.mean(c_l):.3f}   {np.mean(fs_l):.4f}          "
              f"| {np.mean(c_g):.3f}   {np.mean(fs_g):.4f}")

    # ---- per-head exchange rate (which of the 3 score heads is worth fixing) -
    print("\n== per-head: blend ONLY column j toward the latent truth")
    print("  lam  |   light-only     mid-only      k1-only     all three")
    for lam in (0.0, 0.25, 0.50, 1.00):
        row = []
        for j in (0, 1, 2, None):
            fs = []
            rng4 = np.random.default_rng(seed)
            for r in range(reps):
                s_sim = rng4.binomial(ng.astype(int), p_sim) / ng
                ps = base_s.copy()
                cols = range(3) if j is None else (j,)
                for cc in cols:
                    ps[:, cc] = (1 - lam) * base_s[:, cc] + lam * p_sim[:, cc]
                fs.append(best_final_on(np.clip(ps, 0, 1), base_c, dv, s_sim)[0])
            row.append(np.mean(fs))
        print(f"  {lam:.2f} | " + "  ".join(f"{v:.4f}      " for v in row))

    # cost-side control: blend the predicted cost toward the true cost
    print("\n== cost-side exchange rate (no label noise involved), same simulation")
    print("  lam  | log-cost RMSE l/m/k1        final    fast    balanced  premium")
    tc = dv.cost
    ref = np.mean([np.mean((np.log(base_c[:, j]) - np.log(tc[:, j])) ** 2) ** 0.5
                   for j in range(3)])
    for lam in (0.0, 0.10, 0.25, 0.50, 1.0):
        pc = np.exp((1 - lam) * np.log(base_c) + lam * np.log(tc))
        rm = [float(np.sqrt(np.mean((np.log(pc[:, j]) - np.log(tc[:, j])) ** 2))) for j in range(3)]
        fs, pt = [], []
        rng2 = np.random.default_rng(seed)
        for r in range(reps):
            s_sim = rng2.binomial(ng.astype(int), p_sim) / ng
            tot, per = best_final_on(base_s, pc, dv, s_sim)
            fs.append(tot)
            pt.append([per[t] for t in labdata.TIERS])
        q = np.mean(pt, 0)
        print(f"  {lam:.2f} | {rm[0]:.3f}/{rm[1]:.3f}/{rm[2]:.3f}        {np.mean(fs):.4f}  "
              f"{q[0]:.4f}  {q[1]:.4f}   {q[2]:.4f}")
    print(f"  (baseline mean log-cost RMSE {ref:.3f})")


if __name__ == "__main__":
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
