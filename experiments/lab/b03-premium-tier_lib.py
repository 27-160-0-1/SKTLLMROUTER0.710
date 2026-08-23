# SPDX-License-Identifier: Apache-2.0
"""b03 shared: premium-tier transforms, all fitted train-only / cross-fit."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP

FOLDS = 5
SEED = 1234


def inner_folds(n, folds=FOLDS, seed=SEED):
    return np.random.default_rng(seed).integers(0, folds, size=n)


# ---------------------------------------------------------------- C6  rescue
def fit_rescue(smid, sk1, fam, shrink=30.0):
    """Per-family affine  s_k1_hat = A_f + B_f * s_mid_hat  (ridge-shrunk to the global line)."""
    out = {}
    Xg = np.column_stack([np.ones(len(smid)), smid])
    gb = np.linalg.lstsq(Xg, sk1, rcond=None)[0]
    for f in sorted(set(fam)):
        s = fam == f
        n = int(s.sum())
        if n < 8:
            out[f] = gb
            continue
        X = np.column_stack([np.ones(n), smid[s]])
        b = np.linalg.lstsq(X, sk1[s], rcond=None)[0]
        w = n / (n + shrink)
        out[f] = w * b + (1 - w) * gb
    out["__global__"] = gb
    return out


def apply_rescue(coef, smid, fam):
    a = np.array([coef.get(f, coef["__global__"])[0] for f in fam])
    b = np.array([coef.get(f, coef["__global__"])[1] for f in fam])
    return np.clip(a + b * smid, 0.0, 1.0)


# ------------------------------------------------- C4  variance re-transform
def fit_sigma2(lab, idx, pcmat, shrink=30.0):
    """Per (family, model) variance of log(true/pred) cost, shrunk to the global value."""
    res = np.log(lab.true_c[idx]) - np.log(pcmat)
    fam = lab.fam_arr[idx]
    g = res.var(axis=0)
    out = {}
    for f in sorted(set(fam)):
        s = fam == f
        n = int(s.sum())
        v = res[s].var(axis=0)
        w = n / (n + shrink)
        out[f] = w * v + (1 - w) * g
    out["__global__"] = g
    return out


def apply_sigma2(sig, fam, kappa):
    M = np.array([sig.get(f, sig["__global__"]) for f in fam])
    return np.exp(kappa * M)
