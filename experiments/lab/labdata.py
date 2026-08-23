# SPDX-License-Identifier: Apache-2.0
"""Fast numpy view of the public Train/Dev data + exact allocation/scoring.

Loaded once, reused by every lab experiment.  Nothing here touches the
deployment artifacts; it only reads data/ and src/ossp_router for the
policy constants and the family classifier.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MODEL_IDS = ("ax31-light", "ax31", "axk1-think")
TIERS = ("fast", "balanced", "premium")
TIER_MULT = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
TIER_WEIGHT = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
# cost coefficients from configs/routing-policy.v1.json
RATES = {
    "ax31-light": (0.0, 1.0, 4.0),
    "ax31": (0.0, 2.127, 8.509),
    "axk1-think": (0.0, 6.565, 26.260),
}
TOKEN_UNIT = 1_000_000.0


@dataclass
class Split:
    name: str
    episode_ids: list
    texts: list          # prompt text as the router sees it
    score: np.ndarray    # (n, 3)
    cost: np.ndarray     # (n, 3)
    itok: np.ndarray     # (n, 3)
    otok: np.ndarray     # (n, 3)
    ngen: np.ndarray     # (n, 3)

    def __len__(self):
        return len(self.episode_ids)


def _episode_text(ep: dict) -> str:
    if "prompt" in ep and ep["prompt"] is not None:
        return ep["prompt"]
    return "\n".join(m["content"] for m in ep["messages"])


def load_split(name: str) -> Split:
    inputs = json.loads((ROOT / f"data/materialized/{name}/inputs.json").read_text(encoding="utf-8"))
    outcomes = json.loads((ROOT / f"data/{name}/outcomes.json").read_text(encoding="utf-8"))
    omap = {e["episode_id"]: e["models"] for e in outcomes["episodes"]}
    ids, texts = [], []
    s = np.zeros((len(inputs["episodes"]), 3))
    c = np.zeros_like(s)
    it = np.zeros_like(s)
    ot = np.zeros_like(s)
    ng = np.zeros_like(s)
    for i, ep in enumerate(inputs["episodes"]):
        ids.append(ep["episode_id"])
        texts.append(_episode_text(ep))
        models = omap[ep["episode_id"]]
        for j, m in enumerate(MODEL_IDS):
            row = models[m]
            s[i, j] = float(row["score"])
            it[i, j] = row["input_tokens"]
            ot[i, j] = row["output_tokens"]
            ng[i, j] = row["num_generations"]
            f, ir, orr = RATES[m]
            c[i, j] = f + row["input_tokens"] * ir / TOKEN_UNIT + row["output_tokens"] * orr / TOKEN_UNIT
    return Split(name, ids, texts, s, c, it, ot, ng)


def load_all():
    return load_split("train"), load_split("dev")


# ---------------------------------------------------------------- allocation
def allocate(pred_score: np.ndarray, pred_cost: np.ndarray, true_cost: np.ndarray,
             multiplier: float, safety: float) -> np.ndarray:
    """Replicates learned_router.select_models. Returns selected index per row.

    pred_* drive the decision; the caller scores with the true arrays.
    """
    light_total = pred_cost[:, 0].sum()
    cap = light_total * max(1.0, multiplier * safety)

    def choose(pen):
        util = pred_score - pen * pred_cost / light_total
        # tie-break: prefer the cheaper model (lower index) -> argmax picks first max
        return util.argmax(axis=1)

    sel = choose(0.0)
    total = pred_cost[np.arange(len(sel)), sel].sum()
    if total > cap:
        low, high = 0.0, 1.0
        sel = choose(high)
        total = pred_cost[np.arange(len(sel)), sel].sum()
        while total > cap and high < 2 ** 60:
            low, high = high, high * 2.0
            sel = choose(high)
            total = pred_cost[np.arange(len(sel)), sel].sum()
        for _ in range(40):
            mid = (low + high) / 2.0
            cand = choose(mid)
            ct = pred_cost[np.arange(len(cand)), cand].sum()
            if ct <= cap:
                high, sel, total = mid, cand, ct
            else:
                low = mid
    if total > cap:
        sel = np.zeros(len(pred_score), dtype=int)
    return sel


def tier_result(pred_score, pred_cost, split: Split, tier: str, safety: float):
    sel = allocate(pred_score, pred_cost, split.cost, TIER_MULT[tier], safety)
    n = len(split)
    idx = np.arange(n)
    true_total = split.cost[idx, sel].sum()
    ratio = true_total / split.cost[:, 0].sum()
    passed = ratio <= TIER_MULT[tier] + 1e-15
    score = split.score[idx, sel].mean()
    return dict(sel=sel, ratio=ratio, passed=bool(passed),
                score=float(score), tier_score=float(score if passed else 0.0))


def final_score(pred_score, pred_cost, split: Split, safety: dict, verbose=False):
    total = 0.0
    out = {}
    for t in TIERS:
        r = tier_result(pred_score, pred_cost, split, t, safety[t])
        out[t] = r
        total += TIER_WEIGHT[t] * r["tier_score"]
        if verbose:
            print(f"  {t}: score={r['score']:.6f} ratio={r['ratio']:.6f} passed={r['passed']}")
    out["final"] = total
    return out


# ---------------------------------------------------------------- oracle
def oracle_tier(split: Split, tier: str, use_score=None):
    """Exact LP/greedy oracle: maximise sum(score) s.t. cost budget.

    Uses the same Lagrangian bisection with TRUE score and cost => this is the
    achievable ceiling for the deployed allocator shape.
    """
    s = split.score if use_score is None else use_score
    return tier_result(s, split.cost, split, tier, 1.0)


if __name__ == "__main__":
    tr, dv = load_all()
    for sp in (tr, dv):
        print(f"== {sp.name}: n={len(sp)}")
        print("   mean score per model:", np.round(sp.score.mean(0), 4))
        print("   mean cost  per model:", np.round(sp.cost.mean(0), 6))
        print("   cost ratio vs light :", np.round(sp.cost.mean(0) / sp.cost[:, 0].mean(), 3))
        for t in TIERS:
            o = oracle_tier(sp, t)
            print(f"   oracle {t}: score={o['score']:.4f} ratio={o['ratio']:.4f}")
        w = sum(TIER_WEIGHT[t] * oracle_tier(sp, t)["tier_score"] for t in TIERS)
        print(f"   ORACLE FINAL = {w:.4f}")
        print("   all-light =", round(sp.score[:, 0].mean(), 4),
              " all-mid =", round(sp.score[:, 1].mean(), 4),
              " all-k1 =", round(sp.score[:, 2].mean(), 4))
