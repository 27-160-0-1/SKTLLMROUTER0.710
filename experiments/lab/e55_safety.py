# SPDX-License-Identifier: Apache-2.0
"""E55 - the safety trade-off for the final configuration, priced honestly.

For each candidate triple: the scenario-averaged bootstrap EV and bust rate
computed on Train-only out-of-fold rows, and the single held-out dev sample.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_EXP, DEPLOYED_CFG
import bench2 as B
import priorfeat as PF

CORE = dict(DEPLOYED_EXP, legacy_oof_meta=True, meta_seeds=(11, 23, 37, 53, 71))
lab = Lab()
entries, means, glob = PF.load_table(["local-llm/labels_axlight.jsonl"])
lab.set_extra_features(PF.build_features(lab.texts, lab.fam_arr, entries, means, glob))
cv, arr = B.stage(lab, CORE, tag="prior100")

TRIPLES = [
    ("E43 deployed", {"fast": 0.98, "balanced": 0.87, "premium": 0.85}),
    ("E43 '0.7019' run", {"fast": 0.98, "balanced": 0.89, "premium": 0.88}),
    ("plain-bootstrap argmax", {"fast": 0.96, "balanced": 0.84, "premium": 0.84}),
    ("stress argmax (shipped)", {"fast": 0.94, "balanced": 0.80, "premium": 0.73}),
    ("stress, one notch up", {"fast": 0.95, "balanced": 0.83, "premium": 0.78}),
    ("very conservative", {"fast": 0.91, "balanced": 0.76, "premium": 0.68}),
]
rows = []
for name, sf in TRIPLES:
    r = B.run(lab, cv, arr, DEPLOYED_CFG, label=name, fixed_safety=sf, verbose=False)
    rp = B.run(lab, cv, arr, DEPLOYED_CFG, label=name, fixed_safety=sf, verbose=False,
               scenarios=("plain",))
    rows.append((name, sf, r, rp))
print(f"{'triple':26s} {'safety':>18s} {'EV(stress)':>11s} {'EV(plain)':>10s} "
      f"{'bust% f/b/p (stress)':>22s} {'dev':>9s}  per-tier dev ratio")
for name, sf, r, rp in rows:
    b = "/".join(f"{r['det'][t]['bust']*100:.1f}" for t in TIERS)
    ratios = " ".join(f"{r['dev_tiers'][t]['ratio']:.3f}"
                      f"{'' if r['dev_tiers'][t]['passed'] else '!'}" for t in TIERS)
    print(f"{name:26s} {'/'.join(f'{sf[t]:.2f}' for t in TIERS):>18s} "
          f"{r['EV']:11.6f} {rp['EV']:10.6f} {b:>22s} {r['dev']:9.6f}  {ratios}")
Path("reports/lab/e55_safety.json").write_text(json.dumps(
    [{"name": n, "safety": s, "EV_stress": r["EV"], "EV_plain": p["EV"], "dev": r["dev"],
      "bust": {t: r["det"][t]["bust"] for t in TIERS},
      "dev_tiers": r["dev_tiers"]} for n, s, r, p in rows], indent=2, default=float),
    encoding="utf-8")
lab.set_extra_features(None)
