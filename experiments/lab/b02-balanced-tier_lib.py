# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the b02 balanced-tier study.

Nothing here writes to the repo except the b02_* caches under reports/lab/.
"""
from __future__ import annotations
import pickle, sys, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP, DEPLOYED_SAFETY  # noqa
import bench2 as B  # noqa
import protocol as P  # noqa


def load_stage(tag="base"):
    b = pickle.loads(Path(f"reports/lab/stage_{tag}.pkl").read_bytes())
    return b["cv"], b["arr"]


# ------------------------------------------------------------------ EB labels
def eb_labels(lab):
    """Empirical-Bayes posterior mean of the latent success probability p.

    score = k/n with n = num_generations in {2,4}.  Per (family, model) fit a
    Beta prior by moments after removing the binomial sampling variance, then
    return the posterior mean.  This is the honest 'expected score' label: the
    realised label is an unbiased but very noisy draw, and any oracle built on
    it over-states concentration (winner's curse).
    """
    s = lab.true_s
    n = lab.ngen
    fam = lab.fam_arr
    out = np.zeros_like(s)
    prior = {}
    for f in sorted(set(fam)):
        rows = np.where(fam == f)[0]
        for m in range(3):
            sv = s[rows, m]
            nv = n[rows, m]
            mu = float(sv.mean())
            var_s = float(sv.var())
            # E[p(1-p)] = E[s(1-s)] / (1 - 1/n)
            w = 1.0 - 1.0 / nv
            epq = float(np.mean(sv * (1 - sv) / np.maximum(w, 1e-9)))
            var_p = var_s - float(np.mean(epq / nv))
            var_p = max(var_p, 1e-6)
            var_p = min(var_p, mu * (1 - mu) - 1e-6) if 0 < mu < 1 else var_p
            if not (0 < mu < 1):
                strength = 1e6
            else:
                strength = max(mu * (1 - mu) / var_p - 1.0, 1e-3)
            prior[(f, m)] = (mu, strength)
            out[rows, m] = (nv * sv + strength * mu) / (nv + strength)
    return out, prior


# ------------------------------------------------- fixed-count frontier policy
def count_policy(ps, pc, n_mid, n_k1):
    """Route exactly n_k1 items to k1 (top chord-efficiency) and, among the rest,
    n_mid items to mid (top rung-1 efficiency).  Returns picks (m,)."""
    m = len(ps)
    d1s = ps[:, 1] - ps[:, 0]
    d1c = np.maximum(pc[:, 1] - pc[:, 0], 1e-12)
    dcs = ps[:, 2] - ps[:, 0]
    dcc = np.maximum(pc[:, 2] - pc[:, 0], 1e-12)
    e_k = dcs / dcc
    e_m = d1s / d1c
    pick = np.zeros(m, dtype=int)
    k_order = np.argsort(-e_k, kind="stable")
    ksel = k_order[:n_k1]
    pick[ksel] = 2
    rest = np.setdiff1d(np.arange(m), ksel, assume_unique=False)
    m_order = rest[np.argsort(-e_m[rest], kind="stable")]
    pick[m_order[:n_mid]] = 1
    return pick


def realised(lab, idx, pick, mult):
    r = np.arange(len(idx))
    tc = lab.true_c[idx]
    ts = lab.true_s[idx]
    ratio = tc[r, pick].sum() / tc[:, 0].sum()
    return float(ts[r, pick].mean()), float(ratio), bool(ratio <= mult + 1e-15)


# --------------------------------------------------------- kappa-2 transform
def kappa_transform(k1=1.0, k2=1.0, tiers=("balanced",)):
    def tf(lab, arr, ps, pc, tier):
        if tier not in tiers:
            return ps, pc
        pc = pc.copy()
        pc[:, 1] *= k1
        pc[:, 2] *= k2
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        return ps, pc
    return tf
