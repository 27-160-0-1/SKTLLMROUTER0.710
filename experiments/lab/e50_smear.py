# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""E50 - cost re-transformation against the selection-induced cost bias.

Diagnosis (dbg_infl.py).  Write inflation = true_ratio / predicted_ratio at the
cap.  On Train-OOF it is ~0.99-1.04, on Dev ~0.98-1.07, and the gap is entirely
the *selected* items: for the legacy-OOF stage on dev, predicted/true cost is
0.807 on the selected set but 0.840 on the light baseline.  The allocator picks
items whose cost is under-predicted, so E[c | selected] > c_hat.

The runtime predicts log-cost and exponentiates, i.e. it uses the conditional
*median*.  Replacing it with the conditional *mean* removes exactly this bias:
for a lognormal residual, E[c | m_hat] = exp(m_hat + sigma^2 / 2).  Unlike E32
(which inflated uncertain items only in the decision) the correction is applied
in BOTH the decision and the cap accounting, so it is a calibration change, not
a risk penalty.

Variants
  none      deployed
  global    one sigma per model, from Train-OOF log residuals
  duan      Duan smearing: multiply by mean(exp(residual)) per model
  family    sigma per (family, model)
  hetero    per-item sigma from a residual-magnitude head, exp(kappa * sigma^2)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, MULTS, DEPLOYED_CFG, DEPLOYED_EXP, DEPLOYED_SAFETY  # noqa: E402
import bench2 as B  # noqa: E402
from famrepair import classify_v3  # noqa: E402


def log_residual(lab, cv, cfg, tier):
    _ps, pc = lab.compose(cv, cfg, tier)
    return np.log(lab.true_c[cv["idx"]]) - np.log(pc)


def make_scale(lab, cv, cfg, mode, kappa=1.0):
    """Return a callable tier -> (n,3) multiplicative correction for any arr."""
    per_tier = {}
    for t in TIERS:
        R = log_residual(lab, cv, cfg, t)
        if mode == "global":
            per_tier[t] = ("const", np.exp(0.5 * kappa * R.var(axis=0)))
        elif mode == "duan":
            per_tier[t] = ("const", np.exp(R).mean(axis=0))
        elif mode == "family":
            fam = lab.fam_arr[cv["idx"]]
            tab, glob = {}, np.exp(0.5 * kappa * R.var(axis=0))
            for f in np.unique(fam):
                m = fam == f
                if m.sum() >= 25:
                    tab[f] = np.exp(0.5 * kappa * R[m].var(axis=0))
            per_tier[t] = ("family", tab, glob)
        elif mode == "hetero":
            # residual-magnitude heads on the same meta features available at
            # runtime (the composed prediction row + family one-hot)
            X = np.hstack([lab.compose(cv, cfg, t)[0], np.log(lab.compose(cv, cfg, t)[1]),
                           lab.fam_onehot[cv["idx"]], lab.dense[cv["idx"]]])
            heads = []
            for j in range(3):
                h = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                                  max_leaf_nodes=15, min_samples_leaf=30,
                                                  l2_regularization=3.0, early_stopping=True,
                                                  validation_fraction=0.15, random_state=11)
                h.fit(X, np.abs(R[:, j]))
                heads.append(h)
            per_tier[t] = ("hetero", heads, kappa)
        else:
            per_tier[t] = ("const", np.ones(3))
    return per_tier


def transform_for(per_tier, cfg):
    def f(lab, arr, ps, pc, tier):
        spec = per_tier[tier]
        if spec[0] == "const":
            K = np.tile(spec[1], (len(pc), 1))
        elif spec[0] == "family":
            _k, tab, glob = spec
            fam = lab.fam_arr[arr["idx"]]
            K = np.tile(glob, (len(fam), 1))
            for f_, v in tab.items():
                K[fam == f_] = v
        else:
            _k, heads, kappa = spec
            X = np.hstack([ps, np.log(pc), lab.fam_onehot[arr["idx"]], lab.dense[arr["idx"]]])
            # |residual| ~ sigma * sqrt(2/pi)  ->  sigma = |r| * sqrt(pi/2)
            sig = np.column_stack([h.predict(X) for h in heads]) * np.sqrt(np.pi / 2)
            K = np.exp(0.5 * kappa * sig ** 2)
        out = pc * K
        out[:, 1] = np.maximum(out[:, 1], out[:, 0] * (1 + 1e-12))
        out[:, 2] = np.maximum(out[:, 2], out[:, 1] * (1 + 1e-12))
        return ps, out
    return f


def inflation(lab, src, cfg, tf, tier, sf):
    ps, pc = lab.compose(src, cfg, tier)
    ps, pc = tf(lab, src, ps, pc, tier)
    idx = src["idx"]; r = np.arange(len(idx))
    pick = lab.allocate(ps, pc, MULTS[tier], sf)
    pr = pc[r, pick].sum() / pc[:, 0].sum()
    tr = lab.true_c[idx][r, pick].sum() / lab.true_c[idx][:, 0].sum()
    return tr / pr


if __name__ == "__main__":
    lab = Lab()
    use_famv3 = "--famv3" in sys.argv
    if use_famv3:
        lab.set_family([classify_v3(t) for t in lab.texts])
    tag = "bothv3" if use_famv3 else "legoof"
    exp = dict(DEPLOYED_EXP, legacy_oof_meta=True)
    cv, arr = B.stage(lab, exp, tag=tag)
    cfg = dict(DEPLOYED_CFG)
    out = []
    modes = [("none", 1.0), ("global", 1.0), ("global", 0.5), ("global", 1.5),
             ("duan", 1.0), ("family", 1.0), ("hetero", 1.0), ("hetero", 0.5)]
    for mode, k in modes:
        pt = make_scale(lab, cv, cfg, mode, k)
        tf = transform_for(pt, cfg)
        lbl = f"{mode} k={k}"
        r = B.run(lab, cv, arr, cfg, transform=tf, label=lbl)
        infl = {t: (round(inflation(lab, cv, cfg, tf, t, r["safety"][t]), 4),
                    round(inflation(lab, arr, cfg, tf, t, r["safety"][t]), 4)) for t in TIERS}
        print(f"    inflation train/dev: " +
              " ".join(f"{t}={infl[t][0]}/{infl[t][1]}" for t in TIERS), flush=True)
        if mode in ("global", "duan") and k == 1.0:
            print("    factors:", {t: np.round(pt[t][1], 4).tolist() for t in TIERS})
        out.append({"mode": lbl, "EV": r["EV"], "dev": r["dev"], "safety": r["safety"],
                    "bust": {t: r["det"][t]["bust"] for t in TIERS}, "infl": infl,
                    "dev_tiers": r["dev_tiers"]})
    Path("reports/lab/e50_smear.json").write_text(json.dumps(out, indent=2, default=float),
                                                  encoding="utf-8")
