# SPDX-License-Identifier: Apache-2.0
"""a08 probe 14: arithmetic cost of a static-embedding representation
(token-embedding lookup + pooling) for one 880-item tier, measured on this CPU.

Also measures the deployed hashed featuriser on the same batch for reference.
"""
from __future__ import annotations
import glob
import os
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import load_split  # noqa: E402
from ossp_router import learned_router, similarity  # noqa: E402
from ossp_router.protocol import load_input  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

dv = load_split("dev")
texts = dv.texts
CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub"))
p = glob.glob(str(CACHE / "models--skt--A.X-3.1-Light/snapshots/*/tokenizer.json"))
tk = Tokenizer.from_file(p[0])

t0 = time.perf_counter()
enc = tk.encode_batch_fast(texts) if hasattr(tk, "encode_batch_fast") else tk.encode_batch(texts)
t_tok = time.perf_counter() - t0
ids = [np.asarray(e.ids, dtype=np.int64) for e in enc]
ntok = sum(len(x) for x in ids)
print(f"tokenize 880 dev prompts: {t_tok*1000:.0f} ms   total tokens={ntok:,} "
      f"(median {int(np.median([len(x) for x in ids]))}, max {max(len(x) for x in ids):,})")

for V, D in ((32768, 256), (32768, 384), (65536, 512)):
    E = np.random.default_rng(0).standard_normal((V, D)).astype(np.float32) * 0.02
    idsc = [np.clip(x, 0, V - 1) for x in ids]
    t0 = time.perf_counter()
    out = np.empty((len(idsc), D), dtype=np.float32)
    for i, x in enumerate(idsc):
        out[i] = E[x].mean(axis=0)
    t_emb = time.perf_counter() - t0
    print(f"  lookup+meanpool V={V} D={D}: {t_emb*1000:7.0f} ms  "
          f"table={V*D*4/1e6:.0f} MB  ({ntok*D/1e6:.0f} M float adds)")

# reference: the deployed pure-python featuriser on the same batch
inputs = load_input(ROOT / "data/materialized/dev/inputs.json")
eps = list(inputs.episodes)
art = learned_router.load_artifact(ROOT / "src/ossp_router/resources/learned-router.v1.json")
t0 = time.perf_counter()
for ep in eps[:200]:
    d = learned_router.raw_dense_features(ep)
    learned_router.feature_items(ep, word_hash_bins=art.word_hash_bins,
                                 char_hash_bins=art.char_hash_bins,
                                 dense_mean=art.dense_mean, dense_scale=art.dense_scale,
                                 raw_dense=d)
t_feat = (time.perf_counter() - t0) * len(eps) / 200
print(f"deployed hashed featuriser (pure python, extrapolated to 880): {t_feat*1000:.0f} ms")
t0 = time.perf_counter()
for t in texts[:200]:
    similarity.hashed_counts(t)
t_knn = (time.perf_counter() - t0) * len(texts) / 200
print(f"similarity.hashed_counts (pure python, extrapolated to 880):  {t_knn*1000:.0f} ms")
print(f"\nbudget: 90 s per tier, 2 CPU arm64 under QEMU; deployed runtime is ~7 s/tier")
