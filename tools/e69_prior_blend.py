# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""E69 step 2 -- blend the prior columns' own scores into the final score row on lookup hits.

The offline columns are direct measurements: column A (the organiser's light model, Q6) agrees
with the true light score at corr ~0.70, the 34B column with the true mid score at ~0.71.
Today they only reach the allocator through GBM features, diluted by everything else.  On a
hit, the cheapest possible use is a decision-layer blend:

    ps[light] <- (1-w)·ps[light] + w·colA_score        (where colA has a score)
    ps[mid]   <- (1-w)·ps[mid]   + w·colC_score        (where colC has a score)

E42's caveat applies (a better score head can still lose EV), so the judgement is the weighted
EV at the no-bust triple with a stem-grouped paired bootstrap, not RMSE.

Usage: PYTHONPATH=src python tools/e69_prior_blend.py [--artifact A.json] [--boot 1000]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
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

WEIGHTS = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
SAFETY = {"fast": 0.90, "balanced": 0.70, "premium": 0.56}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=ROOT / "reports/e67_append/learned-router.v1.json")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--weights", default="0.0,0.25,0.5,0.75")
    a = ap.parse_args()

    raw = json.loads(a.artifact.read_text(encoding="utf-8")); raw.pop("public_lookup", None)
    art = learned_router.parse_artifact(raw, base_path=a.artifact.parent)
    cols = {c["tag"]: c for c in json.loads(a.artifact.read_text(encoding="utf-8"))["prior_lookup"]["columns"]}
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

    # prior scores per episode (nan = miss or unscored)
    pl = np.full(n, np.nan); pm = np.full(n, np.nan)
    for i, e in enumerate(eps):
        d = hashlib.sha256(episode_text(e).encode("utf-8")).hexdigest()
        r = col_light.get(d)
        if r is not None and r[0] >= 0.0:
            pl[i] = r[0]
        r = col_mid.get(d)
        if r is not None and r[0] >= 0.0:
            pm[i] = r[0]

    P = {}
    for tier in TIERS:
        pr = [learned_router.predict_episode_augmented(e, art, tier) for e in eps]
        P[tier] = (np.array([[p[0][m] for m in MODEL_IDS] for p in pr]),
                   np.array([[p[1][m] for m in MODEL_IDS] for p in pr]))

    hl = np.isfinite(pl); hm = np.isfinite(pm)
    print(f"[blend] scored hits: colA {hl.sum()}/{n}, col34B {hm.sum()}/{n}")
    ps0 = P["premium"][0]
    print(f"[blend] corr with truth on hits:  stack light {np.corrcoef(ps0[hl,0],S[hl,0])[0,1]:.3f} vs colA {np.corrcoef(pl[hl],S[hl,0])[0,1]:.3f}"
          f"   |  stack mid {np.corrcoef(ps0[hm,1],S[hm,1])[0,1]:.3f} vs col34B {np.corrcoef(pm[hm],S[hm,1])[0,1]:.3f}")

    stem = {}
    with (ROOT / "analysis/episodes.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem[row["episode_id"]] = row.get("stem_id") or row["episode_id"]
    groups = defaultdict(list)
    for i, e in enumerate(eps):
        groups[stem.get(e.episode_id, e.episode_id)].append(i)
    glist = list(groups.values())

    def blended(ps, w):
        out = ps.copy()
        out[hl, 0] = (1 - w) * ps[hl, 0] + w * pl[hl]
        out[hm, 1] = (1 - w) * ps[hm, 1] + w * pm[hm]
        return out

    rows = np.arange(n)

    def full_ev(w):
        tot = 0.0
        for tier in TIERS:
            ps, pc = P[tier]
            pk = allocate(blended(ps, w), pc, MULTS[tier], SAFETY[tier])
            got = S[rows, pk].mean()
            if C[rows, pk].sum() / C[:, 0].sum() > MULTS[tier]:
                got = 0.0
            tot += WEIGHTS[tier] * got
        return tot

    ws = [float(x) for x in a.weights.split(",")]
    print(f"\n{'w':>6}{'weighted EV (full dev)':>24}")
    for w in ws:
        print(f"{w:>6.2f}{full_ev(w):>24.6f}")

    # paired bootstrap of the best nonzero w vs w=0
    best_w = max(ws[1:], key=full_ev) if len(ws) > 1 else 0.0
    rng = np.random.default_rng(7)
    deltas = []
    for _ in range(a.boot):
        pick = rng.integers(0, len(glist), size=len(glist))
        s = np.concatenate([glist[g] for g in pick]); r = np.arange(len(s))
        vals = {}
        for w in (0.0, best_w):
            tot = 0.0
            for tier in TIERS:
                ps, pc = P[tier]
                pk = allocate(blended(ps, w)[s], pc[s], MULTS[tier], SAFETY[tier])
                if C[s][r, pk].sum() / C[s][:, 0].sum() <= MULTS[tier]:
                    tot += WEIGHTS[tier] * S[s][r, pk].mean()
            vals[w] = tot
        deltas.append(vals[best_w] - vals[0.0])
    d = np.array(deltas); lo, hi = np.quantile(d, [0.05, 0.95])
    print(f"\n[blend] w={best_w} vs w=0, {a.boot} stem-grouped paired resamples:")
    print(f"   mean {d.mean():+.6f}   90% CI [{lo:+.6f}, {hi:+.6f}]   P(>0) {np.mean(d > 0):.3f}")
    print("   -> " + ("ADOPT-CANDIDATE" if lo > 0 else "noise" if np.mean(d > 0) < 0.8 else "lean, not significant"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
