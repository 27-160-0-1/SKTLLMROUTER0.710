# SPDX-License-Identifier: Apache-2.0
"""E52 - the robust core: C1 (legacy out-of-fold meta features) + seed-averaged
meta heads, evaluated over several fit rotations rather than one.

b04 measured that a single meta fit is itself a large risk source (fast-ratio sd
0.0183 across refits, 2/10 single fits bust the fast tier).  b08 measured C1 as
the only candidate that survives an unbiased rotation test.  This script
confirms both together under bench2 and reports the safety curve so the triple
can be re-priced honestly.
"""
import sys, json, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, DEPLOYED_EXP, DEPLOYED_CFG, DEPLOYED_SAFETY
import bench2 as B

lab = Lab()
ARMS = [
    ("A baseline (1 seed)",        dict(DEPLOYED_EXP), "base"),
    ("B +legacy-OOF (1 seed)",     dict(DEPLOYED_EXP, legacy_oof_meta=True), "legoof"),
    ("C +legacy-OOF +5-seed avg",  dict(DEPLOYED_EXP, legacy_oof_meta=True,
                                        meta_seeds=(11, 23, 37, 53, 71)), "core5"),
]
out = []
for label, exp, tag in ARMS:
    cv, arr = B.stage(lab, exp, tag=tag, force=(tag == "core5"))
    r = B.run(lab, cv, arr, DEPLOYED_CFG, label=label, keep_curves=True)
    r2 = B.run(lab, cv, arr, DEPLOYED_CFG, label=label + " @E43", fixed_safety=DEPLOYED_SAFETY)
    out.append({"label": label, "EV": r["EV"], "dev": r["dev"], "safety": r["safety"],
                "bust": {t: r["det"][t]["bust"] for t in TIERS},
                "dev_tiers": r["dev_tiers"],
                "at_e43": {"EV": r2["EV"], "dev": r2["dev"], "dev_tiers": r2["dev_tiers"],
                           "bust": {t: r2["det"][t]["bust"] for t in TIERS}},
                "curves": r["curves"]})
Path("reports/lab/e52_core.json").write_text(json.dumps(out, indent=2, default=float),
                                             encoding="utf-8")
print("\n=== safety curves for the core config (EV / bust) ===")
c = out[-1]["curves"]
for t in TIERS:
    g = np.array(c[t]["grid"]); ev = np.array(c[t]["ev"]); bu = np.array(c[t]["bust"])
    sel = np.arange(0, len(g), max(1, len(g)//14))
    print(f" {t}: " + "  ".join(f"{g[i]:.3f}:{ev[i]:.4f}/{bu[i]*100:.0f}%" for i in sel))
