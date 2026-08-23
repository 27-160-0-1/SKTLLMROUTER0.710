# SPDX-License-Identifier: Apache-2.0
"""Fidelity gate for a new prior column, measured against the organiser's own labels.

A prior column is only worth its GPU hours if it predicts the model it proxies better than
the column it replaces.  E56's numbers are the bar:

    column A  skt/A.X-3.1-Light Q6_K   corr(A, s_light) = 0.739
    column B  Qwen2.5-14B Q4_K_M       corr(B, s_mid)   = 0.612   <- the proxy being replaced

This script joins a labels file to the 2,640 public Train+Dev items by prompt SHA-256 and
reports, per target model, correlation / within-.25 agreement / mean calibration, overall and
per family.  It also reports how many of the labelled prompts are digest-identical to entries
already in the shipped artifact's columns, which verifies that the pool was re-rendered the
same way (a low overlap means the prompts drifted and the columns will not align).

Usage:
    PYTHONPATH=src python colab-label/prior_column_report.py \
        --labels colab-label/out/labels_mid.jsonl \
        --items  colab-label/bundle/all.jsonl colab-label/bundle/ext.jsonl \
        [--artifact src/ossp_router/resources/learned-router.v1.json] \
        [--inputs data/materialized/train/inputs.json data/materialized/dev/inputs.json] \
        [--outcomes data/train/outcomes.json data/dev/outcomes.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import MODEL_IDS, load_input, load_outcomes  # noqa: E402


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--items", type=Path, nargs="+", required=True)
    ap.add_argument("--artifact", type=Path, default=ROOT / "src/ossp_router/resources/learned-router.v1.json")
    ap.add_argument("--inputs", type=Path, nargs="+",
                    default=[ROOT / "data/materialized/train/inputs.json", ROOT / "data/materialized/dev/inputs.json"])
    ap.add_argument("--outcomes", type=Path, nargs="+",
                    default=[ROOT / "data/train/outcomes.json", ROOT / "data/dev/outcomes.json"])
    a = ap.parse_args()

    prompts = {}
    for path in a.items:
        for row in read_jsonl(path):
            prompts[row["id"]] = row["prompt"]
    print(f"[report] item pool: {len(prompts)} prompts from {len(a.items)} file(s)")

    labels = {}
    for path in a.labels:
        for row in read_jsonl(path):
            text = prompts.get(row["id"])
            if text is None:
                continue
            labels[sha(text)] = row
    print(f"[report] labels joined to prompts: {len(labels)}")

    # --- digest overlap with the shipped columns: did the pool re-render identically? ---
    if a.artifact.exists():
        art = json.loads(a.artifact.read_text(encoding="utf-8"))
        for i, col in enumerate((art.get("prior_lookup") or {}).get("columns", [])):
            keys = set(col["entries"])
            inter = len(keys & set(labels))
            print(f"[report] digest overlap with shipped column {i} ({col.get('tag')}): "
                  f"{inter}/{len(keys)} = {inter/max(len(keys),1):.3f}")

    # --- fidelity against the organiser's labels on the public 2,640 ---
    episodes, index = [], {}
    for ip, op in zip(a.inputs, a.outcomes):
        batch = load_input(ip)
        episodes.extend(batch.episodes)
        for o in load_outcomes(op).outcomes:
            index[(o.episode_id, o.model_id)] = o

    rows, fams = [], []
    for ep in episodes:
        row = labels.get(sha(episode_text(ep)))
        if row is None or row.get("score") is None:
            continue
        rows.append((row, ep))
        fams.append(row.get("family", "?"))
    print(f"[report] public items covered with a score: {len(rows)}/{len(episodes)} = {len(rows)/len(episodes):.3f}")
    if len(rows) < 30:
        print("[report] too few covered items to judge fidelity")
        return 0

    p = np.array([r["score"] for r, _ in rows], dtype=float)
    fams = np.array(fams)
    print(f"[report] prior mean {p.mean():.3f}")
    print(f"{'target':<12} {'corr':>7} {'within.25':>10} {'true mean':>10}")
    for m in MODEL_IDS:
        t = np.array([float(index[(ep.episode_id, m)].score) for _, ep in rows])
        corr = float(np.corrcoef(p, t)[0, 1])
        within = float((np.abs(p - t) <= 0.25).mean())
        print(f"{m:<12} {corr:>7.3f} {within:>10.3f} {t.mean():>10.3f}")

    mid = np.array([float(index[(ep.episode_id, "ax31")].score) for _, ep in rows])
    print("\nper family vs ax31 (the model this column proxies):")
    print(f"{'family':<18} {'n':>5} {'corr':>7} {'within.25':>10}")
    for f in sorted(set(fams)):
        sel = fams == f
        if sel.sum() < 15:
            continue
        c = float(np.corrcoef(p[sel], mid[sel])[0, 1]) if np.std(p[sel]) > 0 else float("nan")
        print(f"{f:<18} {sel.sum():>5} {c:>7.3f} {float((np.abs(p[sel]-mid[sel])<=0.25).mean()):>10.3f}")

    corr_mid = float(np.corrcoef(p, mid)[0, 1])
    print(f"\nGATE: corr(this column, ax31) = {corr_mid:.3f} vs 0.612 for the Qwen2.5-14B column B.")
    print("ADOPT and rebuild the prior" if corr_mid >= 0.66 else
          "BELOW BAR — a column that does not beat 0.612 will not move the router; stop here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
