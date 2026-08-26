# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a06 step 5: what drives k1 output length / cost, and where the budget risk sits."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a06_features import build  # noqa: E402

CTX = 32768


def rho(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(spearmanr(a, b).statistic)


def main():
    tr, dv = build("train"), build("dev")
    for d, nm in ((tr, "train"), (dv, "dev")):
        sp, fam, X, names = d["split"], d["fam"], d["X"], d["names"]
        og = d["otok"]                       # (n,3) output tokens PER GENERATION
        ig = d["itok"]                       # (n,)  input tokens per generation
        y = np.log1p(og[:, 2])
        print(f"\n########## {nm} ##########")
        print("A. correlates of log k1 output tokens / generation")
        print(f"{'family':16s} {'n':>4s} {'r(inputtok)':>11s} {'r(own s_k1)':>11s} "
              f"{'r(s_light)':>10s} {'r(o_light)':>10s} {'r(o_mid)':>9s} "
              f"{'med o_k1':>9s} {'p95':>7s} {'sd log':>7s}")
        for f in sorted(set(fam)) + ["ALL"]:
            m = np.ones(len(sp), bool) if f == "ALL" else (fam == f)
            print(f"{f:16s} {m.sum():4d} {rho(ig[m], y[m]):11.3f} "
                  f"{rho(sp.score[m,2], y[m]):11.3f} {rho(sp.score[m,0], y[m]):10.3f} "
                  f"{rho(og[m,0], og[m,2]):10.3f} {rho(og[m,1], og[m,2]):9.3f} "
                  f"{np.median(og[m,2]):9.0f} {np.percentile(og[m,2],95):7.0f} "
                  f"{np.std(y[m]):7.3f}")

        print("\nB. budget concentration: share of total k1 cost held by the top items")
        c2 = sp.cost[:, 2]
        o = np.sort(c2)[::-1]
        tot = c2.sum()
        for k in (1, 5, 10, 20, 50, 100):
            print(f"   top {k:4d} items ({100*k/len(sp):4.1f}%) hold "
                  f"{100*o[:k].sum()/tot:5.1f}% of all k1 cost "
                  f"(= {o[:k].sum()/sp.cost[:,0].sum():5.2f} light-budgets)")
        print(f"   family of the top-20 k1-cost items: "
              f"{dict(zip(*np.unique(fam[np.argsort(-c2)[:20]], return_counts=True)))}")

        print("\nC. context-limit truncation (input+output per generation vs 32768)")
        for j, mn in enumerate(("light", "mid", "k1")):
            tot_tok = ig + og[:, j]
            near = tot_tok > CTX * 0.97
            print(f"   {mn:6s} max in+out/gen = {tot_tok.max():8.0f}  "
                  f">31.8k: {int(near.sum()):3d} items  "
                  f"max out/gen={og[:,j].max():8.0f}  "
                  f"families={dict(zip(*np.unique(fam[near], return_counts=True))) if near.any() else '{}'}")

        print("\nD. score of the items that hit the k1 output ceiling")
        top = np.argsort(-og[:, 2])[:30]
        print(f"   top-30 longest k1 outputs: mean s = {np.round(sp.score[top].mean(0),3)}, "
              f"families={dict(zip(*np.unique(fam[top], return_counts=True)))}")

    # ---- within-family predictability of log k1 output length (train->OOF, dev check)
    print("\n########## E. within-family predictability of log(k1 out tok) ##########")
    print("   5-fold OOF on train with a small GBM on the 35 prompt features")
    print(f"{'family':16s} {'n':>4s} {'sd(y)':>6s} {'OOF r':>7s} {'OOF R2':>7s} {'dev r':>7s}")
    sptr, spdv = tr["split"], dv["split"]
    for f in sorted(set(tr["fam"])):
        m = tr["fam"] == f
        Xf, yf = tr["X"][m], np.log1p(tr["otok"][m, 2])
        if m.sum() < 40:
            continue
        oof = np.zeros_like(yf)
        for a, b in KFold(5, shuffle=True, random_state=0).split(Xf):
            g = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15,
                                              min_samples_leaf=15, learning_rate=0.08,
                                              random_state=0)
            g.fit(Xf[a], yf[a])
            oof[b] = g.predict(Xf[b])
        g = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15,
                                          min_samples_leaf=15, learning_rate=0.08,
                                          random_state=0).fit(Xf, yf)
        md = dv["fam"] == f
        pd_ = g.predict(dv["X"][md])
        yd = np.log1p(dv["otok"][md, 2])
        r2 = 1 - ((oof - yf) ** 2).sum() / ((yf - yf.mean()) ** 2).sum()
        print(f"{f:16s} {m.sum():4d} {np.std(yf):6.3f} {rho(oof, yf):7.3f} {r2:7.3f} "
              f"{rho(pd_, yd):7.3f}")


if __name__ == "__main__":
    main()
