# SPDX-License-Identifier: Apache-2.0
"""a03: build the exact E43 feature matrix (30 dense + 8192 word + 8192 char)
for train and dev, plus targets / families / ngen, and cache to the scratchpad.

Mirrors tools/train_learned_router_gpu.py::_feature_matrix exactly (same
raw_dense_features / feature_items, same standardisation on train stats).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lab"))

from ossp_router import learned_router, legacy_hash_regex  # noqa: E402
from ossp_router.protocol import MODEL_IDS, load_input, load_outcomes, load_bundled_policy  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402
from ossp_router.heuristic import episode_text  # noqa: E402

CACHE = Path(r"C:\Users\PJ05\AppData\Local\Temp\claude\C--portable-skt-LLM1-LLM-ROUTE-0-7000\377d7fd8-9983-4bec-bb08-163cc405f7a3\scratchpad")
CACHE.mkdir(parents=True, exist_ok=True)

WORD_BINS = 8192
CHAR_BINS = 8192


def feature_matrix(inputs, dense_mean=None, dense_scale=None):
    raw_dense = np.asarray(
        [learned_router.raw_dense_features(ep) for ep in inputs.episodes], dtype=np.float32
    )
    if dense_mean is None:
        dense_mean = raw_dense.mean(axis=0)
    if dense_scale is None:
        dense_scale = raw_dense.std(axis=0)
        dense_scale = np.where(dense_scale > 1e-6, dense_scale, 1.0).astype(np.float32)
    rows, cols, vals = [], [], []
    for i, ep in enumerate(inputs.episodes):
        items = learned_router.feature_items(
            ep, word_hash_bins=WORD_BINS, char_hash_bins=CHAR_BINS,
            dense_mean=dense_mean, dense_scale=dense_scale, raw_dense=raw_dense[i],
        )
        for c, v in items.items():
            rows.append(i)
            cols.append(c)
            vals.append(v)
    dim = len(learned_router.DENSE_FEATURE_NAMES) + WORD_BINS + CHAR_BINS
    m = sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)),
                          shape=(len(inputs.episodes), dim), dtype=np.float32)
    m.sum_duplicates()
    m.eliminate_zeros()
    return m, np.asarray(dense_mean, np.float32), np.asarray(dense_scale, np.float32), raw_dense


def targets(inputs, outcomes, policy):
    from decimal import Decimal
    idx = {(r.episode_id, r.model_id): r for r in outcomes.outcomes}
    rows, itok, otok, ngen = [], [], [], []
    for ep in inputs.episodes:
        vs = [idx[(ep.episode_id, m)] for m in MODEL_IDS]
        cs = []
        for r in vs:
            rates = policy.models[r.model_id]
            unit = Decimal(policy.token_unit)
            c = float(rates.fixed_cost + Decimal(r.input_tokens) * rates.input_token_rate / unit
                      + Decimal(r.output_tokens) * rates.output_token_rate / unit)
            cs.append(c)
        rows.append([float(r.score) for r in vs] + [math.log(c) for c in cs])
        itok.append([r.input_tokens for r in vs])
        otok.append([r.output_tokens for r in vs])
        ngen.append([r.num_generations for r in vs])
    return (np.asarray(rows, np.float64), np.asarray(itok, np.float64),
            np.asarray(otok, np.float64), np.asarray(ngen, np.float64))


def legacy_predictions(inputs):
    art = legacy_hash_regex.load_artifact(ROOT / "src/ossp_router/resources/hash-regex-public.v1.json")
    out = []
    for ep in inputs.episodes:
        s, c = legacy_hash_regex.predict_episode(ep, art)
        out.append([s[m] for m in MODEL_IDS] + [math.log(c[m]) for m in MODEL_IDS])
    return np.asarray(out, np.float64)


def main():
    policy = load_bundled_policy()
    tri = load_input(ROOT / "data/materialized/train/inputs.json")
    tro = load_outcomes(ROOT / "data/train/outcomes.json")
    dvi = load_input(ROOT / "data/materialized/dev/inputs.json")
    dvo = load_outcomes(ROOT / "data/dev/outcomes.json")
    print("building train matrix", flush=True)
    Xtr, dmean, dscale, rdtr = feature_matrix(tri)
    print("building dev matrix", flush=True)
    Xdv, _, _, rddv = feature_matrix(dvi, dmean, dscale)
    Ytr, ittr, ottr, ngtr = targets(tri, tro, policy)
    Ydv, itdv, otdv, ngdv = targets(dvi, dvo, policy)
    ftr = np.asarray([classify_family(episode_text(e)) for e in tri.episodes])
    fdv = np.asarray([classify_family(episode_text(e)) for e in dvi.episodes])
    Ltr = legacy_predictions(tri)
    Ldv = legacy_predictions(dvi)
    sparse.save_npz(CACHE / "Xtr.npz", Xtr)
    sparse.save_npz(CACHE / "Xdv.npz", Xdv)
    np.savez_compressed(
        CACHE / "meta.npz", Ytr=Ytr, Ydv=Ydv, ftr=ftr, fdv=fdv, Ltr=Ltr, Ldv=Ldv,
        ngtr=ngtr, ngdv=ngdv, ittr=ittr, itdv=itdv, ottr=ottr, otdv=otdv,
        rdtr=rdtr, rddv=rddv, dmean=dmean, dscale=dscale,
    )
    print("train", Xtr.shape, "nnz", Xtr.nnz, "dev", Xdv.shape, "nnz", Xdv.nnz)
    print("cached to", CACHE)


if __name__ == "__main__":
    main()
