# SPDX-License-Identifier: Apache-2.0
"""b05 step 4: extract three text representations for all 2,640 episodes.

  static   token-embedding lookup + mean pooling ONLY (no transformer layers).
           This is the representation that certainly fits the 90 s / 2 CPU
           container: cost is one gather + one add per token.
  frozen   the full 12-layer multilingual-e5-small encoder, mean-pooled.
           Training-time only; used to bound what the encoder family can do.
  cls      the same encoder's CLS vector (e5 convention is mean pooling, kept
           only as a robustness check).

Everything is cached to reports/lab/b05_embed.npz.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.protocol import load_input          # noqa: E402
from ossp_router.heuristic import episode_text       # noqa: E402
from transformers import AutoTokenizer, AutoModel    # noqa: E402

OUT = Path("reports/lab/b05_embed.npz")
MODEL = "intfloat/multilingual-e5-small"
CAP = 512

tr = load_input(ROOT / "data/materialized/train/inputs.json")
dv = load_input(ROOT / "data/materialized/dev/inputs.json")
texts = [episode_text(e) for e in list(tr.episodes) + list(dv.episodes)]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).eval().cuda().half()
print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M;",
      "embedding", model.embeddings.word_embeddings.weight.shape)

enc = tok(texts, add_special_tokens=True, truncation=True, max_length=CAP)["input_ids"]
lens = np.array([len(x) for x in enc])
order = np.argsort(lens)                      # length bucketing

W = model.embeddings.word_embeddings.weight.detach().float().cpu().numpy()
static = np.zeros((len(texts), W.shape[1]), dtype=np.float32)
t0 = time.perf_counter()
for i, ids in enumerate(enc):
    static[i] = W[ids].mean(axis=0)
print(f"static pooling (fp32, python loop, this x86 box): {time.perf_counter()-t0:.2f}s")

frozen = np.zeros((len(texts), 384), dtype=np.float32)
cls = np.zeros((len(texts), 384), dtype=np.float32)
B = 32
t0 = time.perf_counter()
with torch.no_grad():
    for s in range(0, len(order), B):
        idx = order[s:s + B]
        batch = [enc[i] for i in idx]
        L = max(len(b) for b in batch)
        ii = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(batch), L), dtype=torch.long)
        for j, b in enumerate(batch):
            ii[j, :len(b)] = torch.tensor(b); am[j, :len(b)] = 1
        out = model(input_ids=ii.cuda(), attention_mask=am.cuda()).last_hidden_state
        m = am.cuda().unsqueeze(-1).half()
        pooled = (out * m).sum(1) / m.sum(1)
        frozen[idx] = pooled.float().cpu().numpy()
        cls[idx] = out[:, 0].float().cpu().numpy()
print(f"frozen 12-layer encode (4090, fp16): {time.perf_counter()-t0:.1f}s")

np.savez_compressed(OUT, static=static, frozen=frozen, cls=cls, lens=lens)
print("saved", OUT, static.shape, frozen.shape)
