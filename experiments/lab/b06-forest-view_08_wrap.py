# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b06 forest-view: closing diagnostics.

  1. Where does the oracle-partition advantage live -- score side or cost side?
     (decides whether the 'better partition' route is reachable at all)
  2. The paired dev-difference noise floor CONDITIONAL on neither config busting
     -- the scale against which a "+0.002 dev" claim must be judged.
  3. corr(EV, dev) across the structural frontier rungs, and the significance of
     the -0.149 measured across the 120 random constant vectors.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP
from sklearn.cluster import KMeans
import bench2 as B

OUT = Path("reports/lab/b06_wrap.json")


def main():
    lab = Lab()
    cv, arr = B.stage(lab, DEPLOYED_EXP, tag="base")
    rep = {}
    base = B.run(lab, cv, arr, None, label="BASE")

    # ---- 1. oracle partition split into its score and cost halves
    Z = lab.targets.copy(); Z = (Z - Z.mean(0)) / Z.std(0)
    for K in (9, 18):
        lbl = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(Z)

        def parts(a):
            idx = a["idx"]; L = lbl[idx]
            p = np.zeros((len(idx), 3)); c = np.zeros((len(idx), 3))
            for g in np.unique(L):
                m = L == g
                p[m] = lab.targets[idx][m][:, :3].mean(axis=0)
                c[m] = np.exp(lab.targets[idx][m][:, 3:6].mean(axis=0))
            return np.clip(p, 0, 1), c

        def t_both(l, a, ps, pc, tier):
            p, c = parts(a); return p, c

        def t_score(l, a, ps, pc, tier):
            p, c = parts(a); return p, pc

        def t_cost(l, a, ps, pc, tier):
            p, c = parts(a)
            c[:, 1] = np.maximum(c[:, 1], c[:, 0] * (1 + 1e-12))
            c[:, 2] = np.maximum(c[:, 2], c[:, 1] * (1 + 1e-12))
            return ps, c

        rb = B.run(lab, cv, arr, None, transform=t_both, label=f"oracle K={K} both")
        rs = B.run(lab, cv, arr, None, transform=t_score, label=f"oracle K={K} SCORE only")
        rc = B.run(lab, cv, arr, None, transform=t_cost, label=f"oracle K={K} COST only")
        rep[f"oracle{K}"] = dict(both=[rb["EV"], rb["dev"]], score=[rs["EV"], rs["dev"]],
                                 cost=[rc["EV"], rc["dev"]])

    # ---- 2. paired dev noise floor, conditional on neither config busting
    print("\npaired dev-difference bootstrap, conditional on neither config busting:")
    idx = arr["idx"]; r = np.arange(len(idx)); n = len(idx)
    def picks_for(cfg, safety):
        out = {}
        for t in TIERS:
            ps, pc = lab.compose(arr, cfg, t)
            out[t] = lab.allocate(ps, pc, MULTS[t], safety[t])
        return out
    P0 = picks_for(dict(DEPLOYED_CFG), base["safety"])
    rows = {}
    rng = np.random.default_rng(4); nb = 4000
    S = np.array([lab.true_s[idx][r, P0[t]] for t in TIERS])
    C = np.array([lab.true_c[idx][r, P0[t]] for t in TIERS])
    L = lab.true_c[idx][:, 0]
    for lbl2, cfg in (("legacy_w=0.0", dict(legacy_w=0.0)), ("fam_w=0.50", dict(fam_w=0.5)),
                      ("conf_scale=0.5", dict(conf_scale=0.5)),
                      ("blend_balanced=0.0", dict(blend_balanced=0.0))):
        rr = B.run(lab, cv, arr, cfg, verbose=False)
        P1 = picks_for(dict(DEPLOYED_CFG, **cfg), rr["safety"])
        S1 = np.array([lab.true_s[idx][r, P1[t]] for t in TIERS])
        C1 = np.array([lab.true_c[idx][r, P1[t]] for t in TIERS])
        d_all, d_ok = [], []
        for b in range(nb):
            s = rng.integers(0, n, size=n)
            bse = L[s].sum()
            v0 = v1 = 0.0; ok = True
            for i, t in enumerate(TIERS):
                r0 = C[i][s].sum() / bse; r1 = C1[i][s].sum() / bse
                if r0 > MULTS[t] or r1 > MULTS[t]:
                    ok = False
                v0 += W[t] * (S[i][s].mean() if r0 <= MULTS[t] else 0.0)
                v1 += W[t] * (S1[i][s].mean() if r1 <= MULTS[t] else 0.0)
            d_all.append(v1 - v0)
            if ok:
                d_ok.append(v1 - v0)
        d_all = np.array(d_all); d_ok = np.array(d_ok)
        rows[lbl2] = dict(point=rr["dev"] - base["dev"], sd_all=float(d_all.std()),
                          sd_nobust=float(d_ok.std()), frac_nobust=len(d_ok) / nb)
        print(f"  {lbl2:20s} dev diff {rr['dev']-base['dev']:+.6f}  sd(all)={d_all.std():.5f}"
              f"  sd(no-bust)={d_ok.std():.5f}  P(no bust)={len(d_ok)/nb:.3f}"
              f"  |t|_nobust={abs(rr['dev']-base['dev'])/max(d_ok.std(),1e-9):.2f}")
    rep["paired"] = rows

    # ---- 3. correlations
    fr = json.loads(Path("reports/lab/b06_frontier.json").read_text())
    ev = np.array([x["EV"] for x in fr]); dv = np.array([x["dev"] for x in fr])
    keep = np.array([x["label"] != "0 all-light" for x in fr])
    print(f"\nfrontier rungs: pearson(EV,dev) all={np.corrcoef(ev,dv)[0,1]:+.3f}  "
          f"excl. all-light={np.corrcoef(ev[keep],dv[keep])[0,1]:+.3f}  (n={keep.sum()})")
    sb = json.loads(Path("reports/lab/b06_selbias.json").read_text())
    e2 = np.array([x["EV"] for x in sb["rows"]]); d2 = np.array([x["dev"] for x in sb["rows"]])
    rho = np.corrcoef(e2, d2)[0, 1]; nn = len(e2)
    se = 1.0 / np.sqrt(nn - 3)
    z = 0.5 * np.log((1 + rho) / (1 - rho))
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    print(f"120 random constant vectors: pearson={rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"  -> {'indistinguishable from 0' if lo < 0 < hi else 'significant'}")
    rep["corr"] = dict(frontier_all=float(np.corrcoef(ev, dv)[0, 1]),
                       frontier_excl_light=float(np.corrcoef(ev[keep], dv[keep])[0, 1]),
                       box_rho=float(rho), box_ci=[float(lo), float(hi)],
                       dev_sd_over_box=float(d2.std()), dev_range_over_box=float(d2.max() - d2.min()))
    OUT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
