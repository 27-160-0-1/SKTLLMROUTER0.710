# SPDX-License-Identifier: Apache-2.0
"""b04 - one table over every built stage.

Three views of each arm, because the single-sample dev number is dominated by the
bust lottery at the knife-edge safety the OOF bootstrap picks:
  EV        honest bootstrap expectation, bust priced in (bench2 protocol)
  dev@sel   dev at the safety the OOF bootstrap chose  (the competition number)
  dev@fix   dev at ONE common safety triple for every arm (bust-lottery removed)
  maxpass   the largest per-tier safety at which dev would still have passed
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")
from harness import DEPLOYED_CFG, TIERS, MULTS, W
import bench2 as B
import pickle

FIX = {"fast": 0.940, "balanced": 0.820, "premium": 0.760}
ORDER = ["R0", "R1", "K32", "K64", "K128", "X16", "X64", "VW", "VF", "VS", "VALL",
         "S3", "S5", "S8", "BEST"]
OUT = Path("reports/lab/b04_table.json")


def load(tag):
    f = Path(f"reports/lab/stage_b04_{tag}.pkl")
    if not f.exists():
        return None
    b = pickle.loads(f.read_bytes())
    return b["cv"], b["arr"]


if __name__ == "__main__":
    lab = lib.MemLab(verbose=False)
    rows = {}
    print(f"{'arm':6s} {'EV':>9s} {'bust%f/b/p':>14s} {'dev@sel':>9s} {'safety@sel':>18s} "
          f"{'dev@fix':>9s} {'ratio@fix f/b/p':>20s} {'maxpass f/b/p':>18s}")
    for tag in ORDER:
        got = load(tag)
        if got is None:
            continue
        cv, arr = got
        rs = B.run(lab, cv, arr, DEPLOYED_CFG, label=tag, verbose=False)
        rf = B.run(lab, cv, arr, DEPLOYED_CFG, label=tag, verbose=False, fixed_safety=FIX)
        mp = {}
        for t in TIERS:
            d = rf["dev_tiers"][t]
            mp[t] = FIX[t] * MULTS[t] / d["ratio"]
        rows[tag] = dict(EV=rs["EV"], dev_sel=rs["dev"], safety=rs["safety"],
                         bust={t: rs["det"][t]["bust"] for t in TIERS},
                         dev_fix=rf["dev"], ratio_fix={t: rf["dev_tiers"][t]["ratio"] for t in TIERS},
                         score_fix={t: rf["dev_tiers"][t]["score"] for t in TIERS},
                         passed_fix={t: rf["dev_tiers"][t]["passed"] for t in TIERS},
                         maxpass=mp, knn_cols=int(arr["knn"].shape[1]))
        print(f"{tag:6s} {rs['EV']:9.6f} "
              f"{'/'.join(f'{rs['det'][t]['bust']*100:.1f}' for t in TIERS):>14s} "
              f"{rs['dev']:9.6f} "
              f"{'/'.join(f'{rs['safety'][t]:.3f}' for t in TIERS):>18s} "
              f"{rf['dev']:9.6f} "
              f"{'/'.join(f'{rf['dev_tiers'][t]['ratio']:.3f}' for t in TIERS):>20s} "
              f"{'/'.join(f'{mp[t]:.3f}' for t in TIERS):>18s}", flush=True)
    OUT.write_text(json.dumps(rows, indent=1, default=float))
