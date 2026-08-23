# SPDX-License-Identifier: Apache-2.0
"""b04 - free ensembling: average the meta heads of stages that were already built.

Each arm is a different fit of the same target, so averaging their meta / gain /
rank_eff blocks is a bagged stack that costs only artifact size (N x 1.83 MB of
trees) and N x the tree-evaluation time.  Measured under the bench2 protocol on
BOTH the OOF rows (so the safety ratio is chosen from the ensemble, not from a
single member) and dev.
"""
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")
from harness import DEPLOYED_CFG, TIERS, MULTS
import bench2 as B

OUT = Path("reports/lab/b04_stackens.json")
FIX = {"fast": 0.940, "balanced": 0.820, "premium": 0.760}


def load(tag):
    f = Path(f"reports/lab/stage_b04_{tag}.pkl")
    if not f.exists():
        return None
    b = pickle.loads(f.read_bytes())
    return b["cv"], b["arr"]


def blend(parts):
    cv0, arr0 = parts[0]
    out = []
    for pos in (0, 1):
        base = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in parts[0][pos].items()}
        for k in ("meta", "gain", "rank_eff"):
            base[k] = np.mean([p[pos][k] for p in parts], axis=0)
        out.append(base)
    return out[0], out[1]


COMBOS = {
    "R1": ["R1"],
    "ens_kNN(R1,K32,K64)": ["R1", "K32", "K64"],
    "ens_kNN(R1,K32,K64,K128)": ["R1", "K32", "K64", "K128"],
    "ens_view(R1,VW,VF,VS)": ["R1", "VW", "VF", "VS"],
    "ens_all": ["R1", "K32", "K64", "K128", "X16", "X64", "VW", "VF", "VS", "VALL"],
}

if __name__ == "__main__":
    lab = lib.MemLab(verbose=False)
    res = {}
    for name, tags in COMBOS.items():
        parts = [load(t) for t in tags]
        if any(p is None for p in parts):
            print(f"skip {name} (missing {[t for t,p in zip(tags,parts) if p is None]})")
            continue
        cv, arr = blend(parts) if len(parts) > 1 else parts[0]
        r = B.run(lab, cv, arr, DEPLOYED_CFG, label=name)
        rf = B.run(lab, cv, arr, DEPLOYED_CFG, label=f"  {name} @fix", fixed_safety=FIX)
        res[name] = dict(members=tags, EV=r["EV"], dev=r["dev"], safety=r["safety"],
                         bust={t: r["det"][t]["bust"] for t in TIERS},
                         dev_fix=rf["dev"],
                         ratio_fix={t: rf["dev_tiers"][t]["ratio"] for t in TIERS},
                         maxpass={t: FIX[t] * MULTS[t] / rf["dev_tiers"][t]["ratio"] for t in TIERS})
        OUT.write_text(json.dumps(res, indent=1, default=float))
