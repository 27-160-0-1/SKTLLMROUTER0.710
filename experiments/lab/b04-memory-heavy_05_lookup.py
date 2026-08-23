# SPDX-License-Identifier: Apache-2.0
"""b04 - value of a prompt-hash lookup table, as a function of coverage.

A lookup entry that carries the *outcomes* of a public item turns that item's
prediction into the truth.  The same table covers the same fraction of the OOF
rows the safety ratio is chosen from, so the transform is applied to BOTH sides
of the bench2 protocol - otherwise the safety would be tuned for a pipeline that
does not exist.

Arms:
  full   - the entry carries score AND cost      (what an outcomes table gives)
  score  - the entry carries score only
  cost   - the entry carries cost only
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")
from harness import DEPLOYED_EXP, DEPLOYED_CFG
import bench2 as B

BASE_EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)
OUT = Path("reports/lab/b04_lookup.json")


def make(lab, frac, kind, seed=4):
    u = np.random.default_rng(seed).random(lab.n)
    hit = u < frac

    def tr(_lab, arr, ps, pc, tier):
        rows = arr["idx"]
        m = hit[rows]
        if not m.any():
            return ps, pc
        ps = ps.copy(); pc = pc.copy()
        if kind in ("full", "score"):
            ps[m] = _lab.true_s[rows[m]]
        if kind in ("full", "cost"):
            pc[m] = _lab.true_c[rows[m]]
        return ps, pc
    return tr


if __name__ == "__main__":
    lab = lib.MemLab(verbose=False)
    cv, arr = B.stage(lab, BASE_EXP, tag="b04_R1")
    res = {}
    r = B.run(lab, cv, arr, DEPLOYED_CFG, label="coverage 0.00 (R1)")
    res["base"] = {k: v for k, v in r.items() if k != "curves"}
    for kind in ("full", "score", "cost"):
        for f in (0.05, 0.10, 0.25, 0.50, 1.00):
            r = B.run(lab, cv, arr, DEPLOYED_CFG, transform=make(lab, f, kind),
                      label=f"{kind:5s} coverage {f:.2f}")
            res[f"{kind}_{f:.2f}"] = {k: v for k, v in r.items() if k != "curves"}
    OUT.write_text(json.dumps(res, indent=1, default=float))
