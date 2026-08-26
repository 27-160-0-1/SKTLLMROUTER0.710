# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b07-visualiser - the measured anchors for the project-history figure.

Claimed gains are transcribed from EXPERIMENT_LOG.md (each row carries the
metric the log used).  MEASURED gains are recomputed here, all in one frame:
train-only fit -> dev 880 scored exactly, at a stated safety triple.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP, DEPLOYED_SAFETY  # noqa
import bench2 as B  # noqa

OUT = ROOT / "reports/lab/figs"
OUT.mkdir(parents=True, exist_ok=True)
PUB = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}

# The pre-E43 constant set (E04 alpha30 / E03 blend .75 / E06 fam .3 + conf .4 /
# E07 tier blends / E14 gain alpha .5 / E27 rank beta .25).
E27_CFG = dict(legacy_w=0.75, fam_w=0.3, conf_scale=0.4, gain_alpha=0.5, rank_beta=0.25,
               blend_fast=0.6, blend_balanced=0.3, blend_premium=0.45)
E27_EXP = dict(DEPLOYED_EXP, ridge_alpha=30.0)
LIN_ONLY = dict(DEPLOYED_CFG, fam_w=0.0, conf_scale=0.0,
                blend_fast=0.0, blend_balanced=0.0, blend_premium=0.0)


def dev_final(lab, arr, cfg, safety):
    idx = arr["idx"]
    tot, tiers = 0.0, {}
    for t in TIERS:
        ps, pc = lab.compose(arr, cfg, t)
        pick = lab.allocate(ps, pc, MULTS[t], safety[t])
        r = np.arange(len(idx))
        ratio = float(lab.true_c[idx][r, pick].sum() / lab.true_c[idx][:, 0].sum())
        sc = float(lab.true_s[idx][r, pick].mean())
        ok = ratio <= MULTS[t] + 1e-15
        tiers[t] = dict(score=sc, ratio=ratio, passed=bool(ok))
        tot += W[t] * (sc if ok else 0.0)
    return dict(final=float(tot), tiers=tiers)


def main():
    lab = Lab()
    res = {}

    # --- replica frame: train-only fit -> dev, published safety .98/.89/.88
    arr43 = lab.fit_predict(lab.train_idx, lab.dev_idx, DEPLOYED_EXP)
    arr27 = lab.fit_predict(lab.train_idx, lab.dev_idx, E27_EXP)
    res["replica"] = {
        "E43 cfg @.98/.89/.88": dev_final(lab, arr43, DEPLOYED_CFG, PUB),
        "E43 cfg @.98/.87/.85": dev_final(lab, arr43, DEPLOYED_CFG, DEPLOYED_SAFETY),
        "E27 cfg @.98/.89/.88": dev_final(lab, arr27, E27_CFG, PUB),
        "E27 cfg @.98/.87/.85": dev_final(lab, arr27, E27_CFG, DEPLOYED_SAFETY),
        "linear ensemble only @.98/.89/.88": dev_final(lab, arr43, LIN_ONLY, PUB),
    }

    # --- bench2 frame (EV-selected safety, honest): baseline vs legacy-OOF meta (C1)
    cv0, a0 = B.stage(lab, DEPLOYED_EXP, tag="b07-visualiser")
    cv1, a1 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="b07-legoof")
    r0 = B.run(lab, cv0, a0, DEPLOYED_CFG, label="bench2 baseline")
    r1 = B.run(lab, cv1, a1, DEPLOYED_CFG, label="bench2 legacy-OOF (C1)")
    res["bench2"] = {k: dict(EV=v["EV"], dev=v["dev"], safety=v["safety"],
                             bust={t: v["det"][t]["bust"] for t in TIERS})
                     for k, v in (("baseline", r0), ("legacy-OOF meta (C1)", r1))}

    (OUT / "b07_history_measured.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in res["replica"].items():
        print(f"  {k:38s} dev={v['final']:.6f}  " +
              " ".join(f"{t[:4]}={v['tiers'][t]['score']:.4f}/r{v['tiers'][t]['ratio']:.3f}"
                       f"{'' if v['tiers'][t]['passed'] else '!BUST'}" for t in TIERS))
    print("wrote", OUT / "b07_history_measured.json")


if __name__ == "__main__":
    main()
