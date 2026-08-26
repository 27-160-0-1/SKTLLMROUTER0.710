# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b06 forest-view: the SIMPLICITY FRONTIER.

Measures, under the bench2 honest protocol (10-fold OOF on Train only, safety
chosen from the 3-seed x 400-sample bootstrap EV on those OOF rows, Dev scored
once), a ladder of configurations from "always light" to the full deployed
stack.  Reports EV and dev with the safety triple, plus the number of free
parameters that were fitted on 1,760 episodes to get there.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B

OUT = Path("reports/lab/b06_frontier.json")


# ------------------------------------------------------------------ transforms
def t_light(lab, arr, ps, pc, tier):
    """Always light: give every item score 1 for light, 0 for the upgrades, so the
    envelope slope is negative everywhere and no upgrade is ever bought."""
    n = ps.shape[0]
    q = np.zeros((n, 3)); q[:, 0] = 1.0
    return q, pc


def t_fam(lab, arr, ps, pc, tier):
    f = arr["fam"]
    p = np.clip(f[:, :3], 0.0, 1.0)
    c = np.exp(np.clip(f[:, 3:6], -50, 50))
    c[:, 1] = np.maximum(c[:, 1], c[:, 0] * (1 + 1e-12))
    c[:, 2] = np.maximum(c[:, 2], c[:, 1] * (1 + 1e-12))
    return p, c


def t_legacy(lab, arr, ps, pc, tier):
    f = arr["legacy"]
    p = np.clip(f[:, :3], 0.0, 1.0)
    c = np.exp(np.clip(f[:, 3:6], -50, 50))
    c[:, 1] = np.maximum(c[:, 1], c[:, 0] * (1 + 1e-12))
    c[:, 2] = np.maximum(c[:, 2], c[:, 1] * (1 + 1e-12))
    return p, c


def _mk_mix(w):
    def t(lab, arr, ps, pc, tier):
        f = (1 - w) * arr["legacy"] + w * arr["fam"]
        p = np.clip(f[:, :3], 0.0, 1.0)
        c = np.exp(np.clip(f[:, 3:6], -50, 50))
        c[:, 1] = np.maximum(c[:, 1], c[:, 0] * (1 + 1e-12))
        c[:, 2] = np.maximum(c[:, 2], c[:, 1] * (1 + 1e-12))
        return p, c
    return t


def t_ridge(lab, arr, ps, pc, tier):
    f = arr["lin"]
    p = np.clip(f[:, :3], 0.0, 1.0)
    c = np.exp(np.clip(f[:, 3:6], -50, 50))
    c[:, 1] = np.maximum(c[:, 1], c[:, 0] * (1 + 1e-12))
    c[:, 2] = np.maximum(c[:, 2], c[:, 1] * (1 + 1e-12))
    return p, c


def t_knn(lab, arr, ps, pc, tier):
    f = arr["knn"][:, :6]
    p = np.clip(f[:, :3], 0.0, 1.0)
    c = np.exp(np.clip(f[:, 3:6], -50, 50))
    c[:, 1] = np.maximum(c[:, 1], c[:, 0] * (1 + 1e-12))
    c[:, 2] = np.maximum(c[:, 2], c[:, 1] * (1 + 1e-12))
    return p, c


def main():
    lab = Lab()
    cv, arr = B.stage(lab, DEPLOYED_EXP, tag="base")
    res = []
    t0 = time.perf_counter()

    def go(label, cfg=None, transform=B.ident, nparam=None, note=""):
        r = B.run(lab, cv, arr, cfg, transform=transform, label=label)
        r["nparam"] = nparam; r["note"] = note
        r.pop("cfg", None); r.pop("curves", None)
        res.append(r)
        return r

    # ---- rung 0: no model at all
    go("0 all-light", transform=t_light, nparam=0,
       note="no fitted parameter; allocator never upgrades")

    # ---- rung 1: family means only (9 families x 6 targets)
    go("1 family-mean only", transform=t_fam, nparam=54,
       note="9 regex families x (3 scores + 3 log-costs)")

    # ---- rung 2: legacy 256-bin hash-regex alone
    go("2 legacy hash-regex", transform=t_legacy, nparam=256 * 6,
       note="official 256-bin ridge, refit per fold")

    # ---- rung 3: legacy + family blend (the deployed fam_w = .15)
    for w in (0.15, 0.30, 0.50):
        go(f"3 legacy+family w={w:.2f}", transform=_mk_mix(w), nparam=256 * 6 + 54 + 1,
           note="linear blend of rungs 1 and 2")

    # ---- reference rungs: single components
    go("R ridge-16414 alone", transform=t_ridge, nparam=16414 * 6)
    go("R kNN-16 alone", transform=t_knn, nparam=0, note="1,760 stored target rows")

    # ---- rung 4: linear ensemble only (no meta GBM): blend_* = 0
    go("4 linear ens (no GBM)", cfg=dict(blend_fast=0.0, blend_balanced=0.0, blend_premium=0.0),
       nparam=16414 * 6 + 256 * 6 + 54 + 3, note="legacy*.9 + ridge*.1, +fam .15, +kNN .25")

    # ---- rung 5: meta GBM only (blend_* = 1)
    go("5 meta GBM only", cfg=dict(blend_fast=1.0, blend_balanced=1.0, blend_premium=1.0),
       note="22 GBM heads + gain/rank reconstruction")

    # ---- rung 6: the full deployed stack
    go("6 FULL deployed stack", nparam=None, note="everything")

    print(f"[b06] frontier in {time.perf_counter()-t0:.0f}s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("\n| rung | EV | dev | safety | bust% |")
    for r in res:
        s = "/".join(f"{r['safety'][t]:.3f}" for t in TIERS)
        b = "/".join(f"{r['det'][t]['bust']*100:.1f}" for t in TIERS)
        print(f"| {r['label']:26s} | {r['EV']:.6f} | {r['dev']:.6f} | {s} | {b} |")


if __name__ == "__main__":
    main()
