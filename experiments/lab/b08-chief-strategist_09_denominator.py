# SPDX-License-Identifier: Apache-2.0
"""b08 - is the BUDGET DENOMINATOR a lever?  (wide-grid rerun of 07 section 3)

cap = (predicted light total) * mult * safety, but the real cap is
(TRUE light total) * mult.  The level error is absorbed by the safety scalar; the
question is whether its DISPERSION across batches is a material part of the
realised-ratio variance.  Replacing the denominator by the truth is equivalent to
a per-batch safety multiplier D_true/D_pred, so we sweep a grid wide enough that
both arms reach their optimum, and compare at matched mean realised ratio.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG            # noqa: E402
import bench2 as B                                                # noqa: E402
from importlib import import_module                               # noqa: E402
CE = import_module("b08-chief-strategist_07_ceiling")
RK = import_module("b08-chief-strategist_06_risk")

R = 16
KEYS = RK.KEYS
WIDE = {"fast": np.arange(0.60, 1.101, 0.01),
        "balanced": np.arange(0.45, 1.051, 0.01),
        "premium": np.arange(0.40, 1.051, 0.01)}

if __name__ == "__main__":
    t0 = time.perf_counter()
    lab = Lab()
    res = pickle.loads(Path("reports/lab/b08_rot_arr.pkl").read_bytes())
    cvL, arrL = B.stage(lab, dict(legacy_oof_meta=True), tag="legoof")
    arrs = [res["C1"][s] for s in range(R)] + [{k: arrL[k] for k in KEYS}]

    print("=== D_true/D_pred per replicate (C1, fast tier composition) ===")
    fs = []
    for a in arrs:
        ps, pc = lab.compose(a, DEPLOYED_CFG, "fast")
        fs.append(float(lab.true_c[a["idx"]][:, 0].sum() / pc[:, 0].sum()))
    fs = np.array(fs)
    print(f"  mean={fs.mean():.4f} sd={fs.std(ddof=1):.4f} cv={fs.std(ddof=1)/fs.mean():.4%} "
          f"min={fs.min():.4f} max={fs.max():.4f}")

    print("\n=== ratio dispersion at MATCHED mean realised ratio ===")
    for t in TIERS:
        g = WIDE[t]
        s0 = CE.tier_stats_den(lab, arrs, DEPLOYED_CFG, t, g, exact_den=False)
        s1 = CE.tier_stats_den(lab, arrs, DEPLOYED_CFG, t, g, exact_den=True)
        print(f"  -- {t} (cap {MULTS[t]}) --")
        print("    target_r  pred-den: sd_tot sd_item raw   |  exact-den: sd_tot sd_item raw")
        for tgt in (0.75, 0.85, 0.92, 0.97):
            r_t = MULTS[t] * tgt
            i0 = int(np.argmin(np.abs(np.exp(s0["m_log"]) - r_t)))
            i1 = int(np.argmin(np.abs(np.exp(s1["m_log"]) - r_t)))
            print(f"    {r_t:7.3f}   {s0['sd_tot'][i0]:.4f} {s0['sd_item'][i0]:.4f} "
                  f"{s0['raw'][i0]:.4f}  |  {s1['sd_tot'][i1]:.4f} {s1['sd_item'][i1]:.4f} "
                  f"{s1['raw'][i1]:.4f}")

    print("\n=== EV880 with a wide grid ===")
    for ed in (False, True):
        CE.best(lab, arrs, DEPLOYED_CFG, f"C1 exact_den={ed}", exact_den=ed)
    # patched grids
    def best_wide(exact_den, n=880):
        tot = 0.0; sfy = {}; parts = []
        for t in TIERS:
            g = WIDE[t]
            st = CE.tier_stats_den(lab, arrs, DEPLOYED_CFG, t, g, exact_den)
            sd = np.sqrt(st["sd_fit"] ** 2 + st["sd_item"] ** 2 * (880.0 / n))
            pp = norm.cdf((np.log(MULTS[t]) - st["m_log"]) / np.maximum(sd, 1e-9))
            ev = st["raw"] * pp
            gi = int(np.argmax(ev))
            sfy[t] = float(g[gi]); tot += W[t] * ev[gi]
            parts.append(f"{t[:4]} s={g[gi]:.3f} r={np.exp(st['m_log'][gi]):.3f} "
                         f"raw={st['raw'][gi]:.4f} Pb={100*(1-pp[gi]):.1f}%")
        print(f"  wide exact_den={str(exact_den):5s} EV{n}={tot:.6f}  " + " | ".join(parts), flush=True)
        return tot
    a = best_wide(False)
    b = best_wide(True)
    print(f"  ORACLE value of a perfect budget denominator: {b - a:+.6f} EV880")
    print(f"[b08] done ({time.perf_counter()-t0:.0f}s)")
