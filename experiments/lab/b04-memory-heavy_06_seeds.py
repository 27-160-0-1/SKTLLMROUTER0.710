# SPDX-License-Identifier: Apache-2.0
"""b04 - across-fit dispersion of the realised budget ratio, and what averaging buys.

bench2's bootstrap resamples ITEMS with the fitted model held fixed, so it cannot
see the component of the realised-budget-ratio variance that comes from re-fitting
the meta trees.  That component is what a07 identified as eating the premium
margin.  Here it is measured directly in the real harness: one train->dev fit per
GBM random_state, then the seed-averaged head.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")
from harness import DEPLOYED_EXP, DEPLOYED_CFG, DEPLOYED_SAFETY, TIERS, MULTS, W

BASE_EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)
SEEDS = (11, 23, 37, 51, 67, 83, 97, 109, 127, 149)
R1_SAFETY = {"fast": 0.960, "balanced": 0.855, "premium": 0.835}
OUT = Path("reports/lab/b04_seeds.json")


def evaluate(lab, arr, safety):
    out = {}
    for t in TIERS:
        ps, pc = lab.compose(arr, DEPLOYED_CFG, t)
        pick = lab.allocate(ps, pc, MULTS[t], safety[t])
        r = np.arange(len(arr["idx"]))
        tc = lab.true_c[arr["idx"]]; ts = lab.true_s[arr["idx"]]
        ratio = tc[r, pick].sum() / tc[:, 0].sum()
        sc = ts[r, pick].mean()
        ok = ratio <= MULTS[t] + 1e-15
        out[t] = dict(ratio=float(ratio), score=float(sc), passed=bool(ok),
                      max_safety=float(MULTS[t] / ratio * safety[t]))
    out["final"] = float(sum(W[t] * (out[t]["score"] if out[t]["passed"] else 0.0) for t in TIERS))
    return out


if __name__ == "__main__":
    lab = lib.MemLab(verbose=False)
    fits = {}
    for s in SEEDS:
        lab.spec["seeds"] = (s,)
        t0 = time.perf_counter()
        fits[s] = lab.fit_predict(lab.train_idx, lab.dev_idx, BASE_EXP)
        print(f"seed {s:4d} fitted in {time.perf_counter()-t0:.0f}s", flush=True)

    res = {"per_seed": {}, "ens": {}}
    for name, safety in (("deployed", DEPLOYED_SAFETY), ("R1", R1_SAFETY)):
        res["per_seed"][name] = {}
        for s in SEEDS:
            e = evaluate(lab, fits[s], safety)
            res["per_seed"][name][s] = e
            print(f"[{name}] seed {s:4d} " + " ".join(
                f"{t[:4]}={e[t]['ratio']:.3f}{'' if e[t]['passed'] else '!'}" for t in TIERS)
                + f"  final={e['final']:.4f}  max_prem_safety={e['premium']['max_safety']:.3f}",
                flush=True)
        for n in (2, 3, 5, 8, 10):
            ens = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in fits[SEEDS[0]].items()}
            for k in ("meta", "gain", "rank_eff"):
                ens[k] = np.mean([fits[s][k] for s in SEEDS[:n]], axis=0)
            e = evaluate(lab, ens, safety)
            res["ens"].setdefault(name, {})[n] = e
            print(f"[{name}] ENS{n:<3d}  " + " ".join(
                f"{t[:4]}={e[t]['ratio']:.3f}{'' if e[t]['passed'] else '!'}" for t in TIERS)
                + f"  final={e['final']:.4f}  max_prem_safety={e['premium']['max_safety']:.3f}",
                flush=True)
        rr = {t: np.array([res["per_seed"][name][s][t]["ratio"] for s in SEEDS]) for t in TIERS}
        fin = np.array([res["per_seed"][name][s]["final"] for s in SEEDS])
        print(f"[{name}] ratio sd " + " ".join(f"{t[:4]}={rr[t].std(ddof=1):.4f}(mean {rr[t].mean():.3f})"
                                               for t in TIERS)
              + f"  E[final over seed draw]={fin.mean():.4f} (sd {fin.std(ddof=1):.4f})", flush=True)
        res.setdefault("summary", {})[name] = dict(
            ratio_mean={t: float(rr[t].mean()) for t in TIERS},
            ratio_sd={t: float(rr[t].std(ddof=1)) for t in TIERS},
            final_mean=float(fin.mean()), final_sd=float(fin.std(ddof=1)),
            n_bust=int(sum(1 for s in SEEDS for t in TIERS if not res["per_seed"][name][s][t]["passed"])))
    OUT.write_text(json.dumps(res, indent=1, default=float))
