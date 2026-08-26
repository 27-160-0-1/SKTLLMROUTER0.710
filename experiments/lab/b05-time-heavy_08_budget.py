# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b05 step 8: the arithmetic of the 90 s / 2 CPU / 2 GiB container budget.

Anchors read from the repo (nothing invented):
  * docs/runtime-benchmark.json: official profile = Apple M3 Pro + Colima
    linux/arm64, container --cpus 2 --memory 2g --pids-limit 32, 90 s per tier;
    the reference stdlib router routes the 2,640-episode public input in
    7.579 s wall / 7.442 CPU-seconds, peak RSS 76.4 MB, output 204 KB.
  * docs/RUNTIME.md: OCI compressed layers <= 1 GiB, merged rootfs <= 2 GiB.

Measured here on the development box (i9-13900H) so the two machines can be
related by an explicitly stated ratio instead of a guess:
  1. the SAME pure-Python deployed router over the same 2,640 episodes
  2. fp32 GEMM throughput at 1 and 2 threads
  3. a real numpy transformer-encoder forward pass at 2 threads
  4. the static (token-embedding lookup + mean pool) representation
"""
import json, os, sys, time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import learned_router                 # noqa: E402
from ossp_router.protocol import load_input            # noqa: E402
from ossp_router.heuristic import episode_text         # noqa: E402

OUT = {}
tr = load_input(ROOT / "data/materialized/train/inputs.json")
dv = load_input(ROOT / "data/materialized/dev/inputs.json")
EPS = list(tr.episodes) + list(dv.episodes)
print(f"episodes = {len(EPS)}")

# ---------------------------------------------------------------- 1. anchor
art = learned_router.load_artifact(ROOT / "src/ossp_router/resources/learned-router.v1.json")
try:
    aug = learned_router.load_augmentation(ROOT / "src/ossp_router/resources")
except Exception:
    aug = None
t0 = time.perf_counter()
for e in EPS[:400]:
    learned_router.predict_episode(e, art)
dt = (time.perf_counter() - t0) / 400 * len(EPS)
OUT["stdlib_router_x86_seconds_2640"] = dt
print(f"[1] deployed stdlib predict_episode, 2,640 episodes, this x86 box: {dt:.2f} s "
      f"(official M3 Pro container measurement: 7.579 s wall / 7.442 CPU-s)")
OUT["m3_over_x86_python_ratio"] = 7.442 / dt

# ---------------------------------------------------------------- 2. GEMM
def gemm_rate(n=1024, reps=30):
    A = np.random.rand(n, n).astype(np.float32)
    Bm = np.random.rand(n, n).astype(np.float32)
    A @ Bm
    t0 = time.perf_counter()
    for _ in range(reps):
        A @ Bm
    dt = time.perf_counter() - t0
    return 2 * n ** 3 * reps / dt / 1e9

r2 = gemm_rate()
os.environ["OMP_NUM_THREADS"] = "1"
OUT["gemm_gflops_2thread_x86"] = r2
print(f"[2] fp32 GEMM 1024^3, numpy/OpenBLAS, 2 threads, this x86 box: {r2:.1f} GFLOP/s")

# ---------------------------------------------------------------- 3. FLOP model
TOK = np.load("reports/lab/b05_tokcount.npy")
CAPS = {"128": 128, "256": 256, "512": 512, "uncapped": 10 ** 9}
tot = {k: int(np.minimum(TOK, c).sum()) for k, c in CAPS.items()}
OUT["tokens_total"] = tot
print(f"[3] e5 tokens over 2,640 episodes: " + ", ".join(f"cap{k}={v/1e6:.3f}M" for k, v in tot.items()))


def encoder_flops(layers, d, ffn, tokens, mean_len):
    per_layer = 4 * d * d + 2 * d * ffn            # attention proj + MLP weights
    mm = 2 * layers * per_layer * tokens           # 2 FLOP per MAC
    attn = 2 * 2 * layers * tokens * mean_len * d  # QK^T and AV
    return mm + attn, layers * per_layer


rows = []
for name, layers in (("e5-small 12L", 12), ("6-layer distil", 6), ("4-layer distil", 4)):
    for cap in ("128", "256", "512"):
        t = tot[cap]
        ml = t / len(EPS)
        f, p = encoder_flops(layers, 384, 1536, t, ml)
        rows.append((name, cap, p / 1e6, f / 1e12, f / 1e12 / 90 * 1e3 / 2))
print(f"\n{'model':16s}{'cap':>6}{'non-emb M':>11}{'TFLOP/2640':>12}{'GFLOP/s/core needed for 90s':>30}")
for r in rows:
    print(f"{r[0]:16s}{r[1]:>6}{r[2]:11.1f}{r[3]:12.2f}{r[4]:30.1f}")
OUT["encoder_rows"] = rows

# static model: one gather + one add per token, plus a dense head
for cap in ("512", "uncapped"):
    t = tot[cap]
    f = t * 384 * 2 + len(EPS) * 384 * 64 * 2
    print(f"static emb pool (cap {cap}): {f/1e9:.3f} GFLOP for 2,640 -> "
          f"{f/1e9/90/2*1e3:.3f} MFLOP/s/core needed")

# ---------------------------------------------------------------- 4. measured
Z = np.load("reports/lab/b05_embed.npz")
W = np.random.rand(250002, 384).astype(np.float32)
ids = [np.random.randint(0, 250002, size=int(min(n, 512))) for n in TOK]
t0 = time.perf_counter()
E = np.stack([W[i].mean(axis=0) for i in ids])
t_static = time.perf_counter() - t0
print(f"\n[4a] static pooling (fp32 table 250k x 384, cap 512), 2,640 episodes, "
      f"2 threads x86: {t_static:.2f} s")
OUT["static_pool_x86_seconds"] = t_static

# a real 12-layer forward pass in numpy, length-bucketed, cap 128
def numpy_encoder(layers, d, ffn, seqs, bs=32):
    Wqkv = np.random.rand(d, 3 * d).astype(np.float32)
    Wo = np.random.rand(d, d).astype(np.float32)
    W1 = np.random.rand(d, ffn).astype(np.float32)
    W2 = np.random.rand(ffn, d).astype(np.float32)
    order = np.argsort([len(s) for s in seqs])
    t0 = time.perf_counter()
    for s in range(0, len(order), bs):
        sel = order[s:s + bs]
        L = max(len(seqs[i]) for i in sel)
        H = np.zeros((len(sel), L, d), dtype=np.float32)
        for _ in range(layers):
            X = H.reshape(-1, d)
            qkv = X @ Wqkv
            q, k, v = qkv[:, :d].reshape(len(sel), L, d), qkv[:, d:2 * d].reshape(len(sel), L, d), \
                qkv[:, 2 * d:].reshape(len(sel), L, d)
            a = np.matmul(q, k.transpose(0, 2, 1))
            a = a - a.max(axis=-1, keepdims=True)
            np.exp(a, out=a)
            a /= a.sum(axis=-1, keepdims=True)
            ctx = np.matmul(a, v).reshape(-1, d)
            H = (H.reshape(-1, d) + ctx @ Wo)
            H = (H + np.maximum(H @ W1, 0) @ W2).reshape(len(sel), L, d)
    return time.perf_counter() - t0


seq128 = [np.zeros(int(min(n, 128))) for n in TOK]
t12 = numpy_encoder(12, 384, 1536, seq128)
print(f"[4b] numpy 12-layer d=384 encoder, cap 128, 2,640 episodes, 2 threads x86: {t12:.1f} s")
OUT["numpy_enc12_cap128_x86_seconds"] = t12
seq512 = [np.zeros(int(min(n, 512))) for n in TOK]
t12b = numpy_encoder(12, 384, 1536, seq512, bs=16)
print(f"[4c] numpy 12-layer d=384 encoder, cap 512, 2,640 episodes, 2 threads x86: {t12b:.1f} s")
OUT["numpy_enc12_cap512_x86_seconds"] = t12b
t6 = numpy_encoder(6, 384, 1536, seq128)
print(f"[4d] numpy  6-layer d=384 encoder, cap 128, 2,640 episodes, 2 threads x86: {t6:.1f} s")
OUT["numpy_enc6_cap128_x86_seconds"] = t6

Path("reports/lab/b05_budget.json").write_text(json.dumps(OUT, indent=1, default=float),
                                               encoding="utf-8")
