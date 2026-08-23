# SPDX-License-Identifier: Apache-2.0
"""b05 step 3: the arithmetic of the 90 s / 2 CPU / 2 GiB inference budget.

Facts used (all read from the repo, none invented):
  docs/runtime-benchmark.json  -> official profile is an Apple M3 Pro Colima
     linux/arm64 VM, container --cpus 2 --memory 2g, 90 s per tier; the
     reference stdlib router processes the 2,640-episode public input in
     7.579 s wall / 7.442 CPU-seconds with a 76 MB RSS.
  docs/RUNTIME.md              -> 2 CPU, 2 GiB, 32 pids, 90 s, /tmp 256 MiB,
     OCI compressed layers <= 1 GiB, merged rootfs <= 2 GiB.
"""
import io, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ossp_router.protocol import load_input          # noqa: E402
from ossp_router.heuristic import episode_text       # noqa: E402

tr = load_input(ROOT / "data/materialized/train/inputs.json")
dv = load_input(ROOT / "data/materialized/dev/inputs.json")
eps = list(tr.episodes) + list(dv.episodes)
texts = [episode_text(e) for e in eps]
chars = np.array([len(t) for t in texts])
print(f"episodes={len(texts)}  chars: mean={chars.mean():.0f} median={np.median(chars):.0f} "
      f"p95={np.percentile(chars,95):.0f} max={chars.max()} total={chars.sum()/1e6:.2f}M")

# runtime already truncates; check what the deployed feature extractor sees
from ossp_router import learned_router                # noqa: E402
lim = getattr(learned_router, "TEXT_LIMIT", None)
print("learned_router TEXT_LIMIT =", lim)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer               # noqa: E402
for name in ("intfloat/multilingual-e5-small",):
    tok = AutoTokenizer.from_pretrained(name)
    t0 = time.perf_counter()
    enc = tok(texts, add_special_tokens=True)["input_ids"]
    dt = time.perf_counter() - t0
    n = np.array([len(x) for x in enc])
    print(f"\n{name}: vocab={tok.vocab_size} tokenise 2640 docs in {dt:.2f}s "
          f"({dt/len(texts)*1000:.2f} ms/doc, fast={tok.is_fast})")
    for cap in (128, 256, 512, 1024, 10**9):
        c = np.minimum(n, cap)
        print(f"  cap={cap:>10}: tokens mean={c.mean():.0f} median={np.median(c):.0f} "
              f"p95={np.percentile(c,95):.0f} total={c.sum()/1e6:.3f}M")
    np.save(Path("reports/lab/b05_tokcount.npy"), n)
