# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""E51 - what is an external per-item difficulty prior worth?  (go/no-go)

The rules explicitly permit "exact-prompt or prompt-hash lookup against public
data" and shipping lookup tables built from public sources, and
colab-label/build_pool.py already proved the official prompt templates are
exactly reproducible from the public benchmarks.  So we could run open models
locally over the full public datasets, store per-item success rates, and look
them up at evaluation time by prompt hash.

E41 measured that A.X-3.1-Light run locally agrees with the official light score
at corr 0.726 (binary agreement 0.876).  Before spending GPU-days building that
table, simulate the feature and measure what it is worth.

Simulation.  The official score is k/n with n in {2,4}, a noisy view of a latent
success probability p.  Draw p from the item's Beta-Binomial posterior, then
draw an INDEPENDENT proxy observation f = Binomial(n_sim, q)/n_sim where
q = mix(p, global mean) is degraded until corr(f, official score) matches the
target.  Independence is the point: a real local run shares only p with the
official label, not its realised noise.

Arms: which columns the proxy covers (light only / light+mid / all three), how
many generations, and how faithful the proxy model is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, DEPLOYED_CFG, DEPLOYED_EXP  # noqa: E402
import bench2 as B  # noqa: E402


def beta_posterior(lab):
    """Per (family, model) Beta prior by moment matching -> posterior a, b."""
    fam = lab.fam_arr
    n = lab.ngen.astype(int)
    k = np.rint(lab.true_s * n).astype(int)
    A = np.ones_like(lab.true_s)
    Bp = np.ones_like(lab.true_s)
    for f in np.unique(fam):
        m = fam == f
        for j in range(3):
            x = lab.true_s[m, j]
            mu, var = x.mean(), x.var()
            nn = n[m, j].mean()
            if not (0 < mu < 1) or nn <= 1:
                A[m, j], Bp[m, j] = 1.0, 1.0
                continue
            vp = (var - (mu - mu * mu) / nn) / (1 - 1.0 / nn)
            vp = float(np.clip(vp, 1e-3, mu * (1 - mu) - 1e-3))
            c = mu * (1 - mu) / vp - 1
            A[m, j], Bp[m, j] = max(c * mu, 0.05), max(c * (1 - mu), 0.05)
    return A + k, Bp + (n - k)


def make_proxy(lab, post_a, post_b, cols, n_sim, fidelity, seed):
    """fidelity in [0,1]: 1 = the proxy model is the evaluation model itself."""
    rng = np.random.default_rng(seed)
    p = rng.beta(post_a, post_b)
    out = np.zeros((lab.n, len(cols)))
    for c, j in enumerate(cols):
        q = fidelity * p[:, j] + (1 - fidelity) * rng.beta(post_a[:, j].mean(),
                                                           post_b[:, j].mean(), size=lab.n)
        out[:, c] = rng.binomial(n_sim, np.clip(q, 0, 1)) / n_sim
    return out


if __name__ == "__main__":
    lab = Lab()
    exp = dict(DEPLOYED_EXP, legacy_oof_meta=True)
    pa, pb = beta_posterior(lab)
    base_cv, base_arr = B.stage(lab, exp, tag="legoof")
    base = B.run(lab, base_cv, base_arr, DEPLOYED_CFG, label="no proxy feature")

    arms = [
        ("light only, n=4, fid .85", [0], 4, 0.85),
        ("light only, n=8, fid .85", [0], 8, 0.85),
        ("light+mid, n=4, fid .85", [0, 1], 4, 0.85),
        ("light+mid+k1, n=4, fid .85", [0, 1, 2], 4, 0.85),
        ("light+mid+k1, n=8, fid .95", [0, 1, 2], 8, 0.95),
        ("light+mid+k1, n=4, fid .60", [0, 1, 2], 4, 0.60),
        ("light+mid+k1, n=4, fid .40", [0, 1, 2], 4, 0.40),
    ]
    out = [{"arm": "none", "EV": base["EV"], "dev": base["dev"], "safety": base["safety"]}]
    for name, cols, n_sim, fid in arms:
        devs, evs, cors = [], [], []
        for seed in (1, 2, 3):
            f = make_proxy(lab, pa, pb, cols, n_sim, fid, seed)
            cors.append([float(np.corrcoef(f[:, c], lab.true_s[:, j])[0, 1])
                         for c, j in enumerate(cols)])
            lab.set_extra_features(f)
            cv, arr = B.stage(lab, exp, tag=f"proxy_s{seed}", force=True)
            r = B.run(lab, cv, arr, DEPLOYED_CFG, label=f"{name} s{seed}", verbose=False)
            devs.append(r["dev"]); evs.append(r["EV"])
        print(f"{name:30s} EV={np.mean(evs):.6f} ({np.mean(evs)-base['EV']:+.4f})  "
              f"dev={np.mean(devs):.6f} ({np.mean(devs)-base['dev']:+.4f})  "
              f"corr(f,s)={np.round(np.mean(cors, axis=0), 3).tolist()}  "
              f"devs={[round(d,4) for d in devs]}", flush=True)
        out.append({"arm": name, "EV": float(np.mean(evs)), "dev": float(np.mean(devs)),
                    "corr": np.mean(cors, axis=0).tolist(), "devs": devs, "evs": evs})
    lab.set_extra_features(None)
    Path("reports/lab/e51_proxyvalue.json").write_text(json.dumps(out, indent=2, default=float),
                                                       encoding="utf-8")
