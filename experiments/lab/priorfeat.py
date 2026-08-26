# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Turn the locally-generated per-item labels into router features.

`local-llm/run_llama.py` scores public benchmark items with an open-weight model
and records the success rate against the public gold answer, the generation
length, and the self-consistency of the samples.  At evaluation time the router
cannot run a model, so the table is keyed by the SHA-256 of the exact prompt
text and consulted by lookup -- which CHALLENGE_RULES permits explicitly
("정확한 프롬프트나 프롬프트 해시를 사용하는 공개 자료 조회도 허용합니다").

Self-consistency needs no gold answer, so items whose published answer we could
not match (DeepMind-Mathematics, most of BABILong) still contribute two of the
three signals.

One *column* is one model's pass.  Consecutive columns are ordered weakest to
strongest, and the cross-column deltas that follow the per-column blocks are the
decision-relevant part: the step between a weaker and a stronger model is a
direct proxy for the upgrade gain the allocator has to rank.

Mirrors `ossp_router.learned_router.prior_features` exactly, so the lab harness
and the container runtime compute identical inputs.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NCOL = 11          # per-column features
NDELTA = 5         # per adjacent column pair
ITEM_FILES = ("colab-label/bundle/union.jsonl", "colab-label/bundle/all.jsonl",
              "colab-label/bundle/public_all.jsonl", "colab-label/bundle/ext.jsonl")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompts(items_paths=None):
    out = {}
    for ip in (items_paths or ITEM_FILES):
        ip = ROOT / ip if not Path(ip).is_absolute() else Path(ip)
        if not ip.exists():
            continue
        for line in ip.read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = json.loads(line)
                out.setdefault(it["id"], it["prompt"])
    return out


def load_column(paths, prompts=None):
    """-> dict(entries, means, glob, tag) for one model pass."""
    prompts = prompts or _prompts()
    entries, by_fam, tag = {}, defaultdict(list), None
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tag = tag or r.get("model", "unknown")
            t = prompts.get(r["id"])
            if t is None:
                continue
            out = r.get("out_tokens_total")
            if out is None:
                out = sum(r.get("out_tokens", []) or [0])
            lo = float(np.log1p(out / max(r.get("n", 4), 1)))
            sc_raw = r.get("sc")
            row = (float(r["score"]) if r.get("score") is not None else -1.0, lo,
                   float(sc_raw) if sc_raw is not None else -1.0)
            entries[sha(t)] = row
            by_fam[r["family"]].append(row)

    def _mean(rows, j):
        vals = [x[j] for x in rows if x[j] >= 0.0]
        return float(np.mean(vals)) if vals else 0.0

    means = {f: (_mean(v, 0), _mean(v, 1), _mean(v, 2))
             for f, v in by_fam.items() if len(v) >= 8}
    flat = [x for v in by_fam.values() for x in v]
    glob = (_mean(flat, 0), _mean(flat, 1), _mean(flat, 2)) if flat else (0.0, 0.0, 0.0)
    return {"tag": tag or "unknown", "entries": entries, "means": means, "glob": glob}


def _col_row(digest, family, col):
    e = col["entries"].get(digest)
    if e is None:
        return [0.0] * NCOL, None
    score, out_length, consistency = e
    m = col["means"].get(family, col["glob"])
    hs = 1.0 if score >= 0.0 else 0.0
    hc = 1.0 if consistency >= 0.0 else 0.0
    return [1.0, hs,
            score if hs else 0.0,
            (score - m[0]) if hs else 0.0,
            1.0 if (hs and score == 0.0) else 0.0,
            1.0 if (hs and score == 1.0) else 0.0,
            out_length, out_length - m[1],
            hc, consistency if hc else 0.0,
            (consistency - m[2]) if hc else 0.0], e


def row_for(text, family, columns):
    digest = sha(text)
    feats, rows = [], []
    for col in columns:
        block, e = _col_row(digest, family, col)
        feats.extend(block); rows.append(e)
    for i in range(1, len(rows)):
        low, high = rows[i - 1], rows[i]
        if low is None or high is None:
            feats.extend([0.0] * NDELTA)
            continue
        both = 1.0 if (low[0] >= 0.0 and high[0] >= 0.0) else 0.0
        feats.extend([1.0, both,
                      (high[0] - low[0]) if both else 0.0,
                      high[1] - low[1],
                      (high[2] - low[2]) if (low[2] >= 0.0 and high[2] >= 0.0) else 0.0])
    return feats


def build_features(texts, families, columns):
    return np.asarray([row_for(t, f, columns) for t, f in zip(texts, families)], dtype=float)


def coverage_report(F, families, ncols=1):
    fam = np.asarray(families)
    out = {"overall": float(F[:, 0].mean())}
    for f in np.unique(fam):
        out[str(f)] = float(F[fam == f, 0].mean())
    for c in range(1, ncols):
        out[f"col{c}"] = float(F[:, c * NCOL].mean())
    return out


def mask_coverage(F, keep_fraction, seed=0, ncols=1):
    """Simulate a private set where only `keep_fraction` of prompts are in the table."""
    rng = np.random.default_rng(seed)
    G = F.copy()
    present = np.where(F[:, 0] > 0)[0]
    drop = rng.permutation(present)[int(round(len(present) * keep_fraction)):]
    G[drop] = 0.0
    return G
