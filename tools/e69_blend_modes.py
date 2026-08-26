# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""E69 step 3 -- which half of the blend destabilises premium?

With the w=0.25 blend on both columns, premium no longer certifies zero-bust anywhere on the
0.50+ grid (it did at 0.56 without the blend).  Plausible mechanism: boosting mid scores on
hits makes the allocator buy more upgrades near the cap, fattening the realised-ratio tail.
This measures premium bust rates and scores at 0.50-0.56 under four modes:

    none        no blend (reference)
    light       blend column A into light only
    mid         blend 34B column into mid only
    both        the adopted candidate

If `light` alone keeps premium's zero-bust at 0.56 while `mid` breaks it, the fix is a
per-model weight rather than a lower triple.  Fast/balanced are also re-checked at their
certified values.

Usage: PYTHONPATH=src python tools/e69_blend_modes.py [--boot 1500]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from bust_probability import allocate  # noqa: E402
from ossp_router import learned_router  # noqa: E402
from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import MODEL_IDS, TIERS, load_bundled_policy, load_input, load_outcomes  # noqa: E402

MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=ROOT / "reports/e67_append/learned-router.v1.json")
    ap.add_argument("--boot", type=int, default=1500)
    a = ap.parse_args()

    raw = json.loads(a.artifact.read_text(encoding="utf-8"))
    raw.pop("public_lookup", None)
    raw.pop("prior_score_blend", None)          # predictions WITHOUT the runtime blend
    art = learned_router.parse_artifact(raw, base_path=a.artifact.parent)
    cols = {c["tag"]: c for c in raw["prior_lookup"]["columns"]}
    col_light = cols["axlight-q6k-v2"]["entries"]
    col_mid = cols["ax31-34b-v1"]["entries"]

    policy = load_bundled_policy(); unit = Decimal(policy.token_unit)
    eps = list(load_input(ROOT / "data/materialized/dev/inputs.json").episodes)
    idx = {(o.episode_id, o.model_id): o for o in load_outcomes(ROOT / "data/dev/outcomes.json").outcomes}
    n = len(eps)

    def tc(eid, mid):
        o = idx[(eid, mid)]; r = policy.models[mid]
        return float(r.fixed_cost + Decimal(o.input_tokens) * r.input_token_rate / unit
                     + Decimal(o.output_tokens) * r.output_token_rate / unit)

    C = np.array([[tc(e.episode_id, m) for m in MODEL_IDS] for e in eps])
    S = np.array([[float(idx[(e.episode_id, m)].score) for m in MODEL_IDS] for e in eps])

    pl = np.full(n, np.nan); pm = np.full(n, np.nan)
    for i, e in enumerate(eps):
        d = hashlib.sha256(episode_text(e).encode("utf-8")).hexdigest()
        r = col_light.get(d)
        if r is not None and r[0] >= 0.0:
            pl[i] = r[0]
        r = col_mid.get(d)
        if r is not None and r[0] >= 0.0:
            pm[i] = r[0]
    hl = np.isfinite(pl); hm = np.isfinite(pm)

    P = {}
    for tier in TIERS:
        pr = [learned_router.predict_episode_augmented(e, art, tier) for e in eps]
        P[tier] = (np.array([[p[0][m] for m in MODEL_IDS] for p in pr]),
                   np.array([[p[1][m] for m in MODEL_IDS] for p in pr]))

    W = 0.25

    def blended(ps, mode):
        out = ps.copy()
        if mode in ("light", "both"):
            out[hl, 0] = (1 - W) * ps[hl, 0] + W * pl[hl]
        if mode in ("mid", "both"):
            out[hm, 1] = (1 - W) * ps[hm, 1] + W * pm[hm]
        return out

    rng = np.random.default_rng(7)
    samples = [rng.integers(0, n, size=n) for _ in range(a.boot)]
    small = [rng.integers(0, n, size=n // 2) for _ in range(a.boot)]
    hit_at = [rng.integers(0, n) for _ in range(a.boot)]
    rows = np.arange(n)

    def profile(tier, safety, mode):
        ps = blended(P[tier][0], mode); pc = P[tier][1]
        mult = MULTS[tier]
        pk = allocate(ps, pc, mult, safety)
        score = S[rows, pk].mean()
        busts = {}
        for name in ("plain", "runaway", "inflation", "small"):
            ss = small if name == "small" else samples
            b = 0
            for k, s in enumerate(ss):
                q = allocate(ps[s], pc[s], mult, safety)
                r = np.arange(len(s))
                cost = C[s][r, q].copy()
                light = C[s][:, 0].sum()
                if name == "runaway":
                    cost[hit_at[k] % len(s)] += 0.065 * light
                elif name == "inflation":
                    cost = cost * np.where(q == 2, 1.25, np.where(q == 1, 1.10, 1.0))
                if cost.sum() / light > mult:
                    b += 1
            busts[name] = b / len(ss)
        return score, busts

    print(f"=== premium, w={W}, {a.boot} resamples/scenario ===")
    print(f"{'mode':<8}{'safety':>8}{'score':>9}{'plain':>9}{'runaway':>9}{'infl':>9}{'small':>9}")
    for mode in ("none", "light", "mid", "both"):
        for safety in (0.50, 0.52, 0.54, 0.56):
            score, busts = profile("premium", safety, mode)
            zero = all(v == 0.0 for v in busts.values())
            print(f"{mode:<8}{safety:>8.2f}{score:>9.4f}"
                  + "".join(f"{busts[k]:>9.2%}" for k in ("plain", "runaway", "inflation", "small"))
                  + ("   <- 무초과" if zero else ""))
        print()
    print("=== fast @0.90 / balanced @0.70,0.72 재확인 (mode=both) ===")
    for tier, saf_list in (("fast", (0.90,)), ("balanced", (0.70, 0.72))):
        for safety in saf_list:
            score, busts = profile(tier, safety, "both")
            zero = all(v == 0.0 for v in busts.values())
            print(f"{tier:<9}{safety:>6.2f}{score:>9.4f}"
                  + "".join(f"{busts[k]:>9.2%}" for k in ("plain", "runaway", "inflation", "small"))
                  + ("   <- 무초과" if zero else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
