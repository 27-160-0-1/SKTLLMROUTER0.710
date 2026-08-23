# SPDX-License-Identifier: Apache-2.0
"""b06 forest-view: (a) side-ceilings that BOUND the round-1 candidate list,
(b) one-at-a-time sensitivity of EV and dev to each of the 8 post-hoc constants.

The cost-side ceiling is the single most useful number for auditing C4/C5/C7/C8:
they all act on the budget-ratio dispersion, so their JOINT effect cannot exceed
the gain from replacing the predicted cost with the true cost everywhere.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B

OUT = Path("reports/lab/b06_ceilings.json")


def main():
    lab = Lab()
    cv, arr = B.stage(lab, DEPLOYED_EXP, tag="base")
    res = []

    def go(label, cfg=None, transform=B.ident, note=""):
        r = B.run(lab, cv, arr, cfg, transform=transform, label=label)
        r["note"] = note
        r.pop("cfg", None); r.pop("curves", None)
        res.append(r)
        return r

    base = go("BASE full stack", note="deployed cfg")

    # -------------------------------------------------- ceilings, cost side
    def t_truecost(lab_, a, ps, pc, tier):
        return ps, lab_.true_c[a["idx"]].copy()

    go("CEIL cost=true", transform=t_truecost,
       note="bounds C4+C5+C7+C8 jointly (all act on budget-ratio dispersion)")

    # cost with only the per-model SUM calibrated (removes the Jensen bias but
    # keeps all per-item dispersion) -- the 'level' part of the cost problem
    def t_sumcal(lab_, a, ps, pc, tier):
        tc = lab_.true_c[a["idx"]]
        k = tc.sum(axis=0) / np.maximum(pc.sum(axis=0), 1e-300)
        return ps, pc * k[None, :]

    go("CEIL cost sum-calibrated", transform=t_sumcal,
       note="oracle per-model multiplier; isolates the LEVEL part of cost error")

    # cost with the per-item dispersion removed but the sum kept wrong: replace
    # each item's cost by its family mean cost -> the 'shape' part
    def t_famcost(lab_, a, ps, pc, tier):
        idx = a["idx"]; fam = lab_.fam_arr[idx]
        out = pc.copy()
        for f in np.unique(fam):
            m = fam == f
            out[m] = pc[m].mean(axis=0)
        out[:, 1] = np.maximum(out[:, 1], out[:, 0] * (1 + 1e-12))
        out[:, 2] = np.maximum(out[:, 2], out[:, 1] * (1 + 1e-12))
        return ps, out

    go("ABL cost=family-mean-of-pred", transform=t_famcost,
       note="destroys within-family cost ranking; how much is the per-item cost head worth?")

    # -------------------------------------------------- ceilings, score side
    def t_truescore(lab_, a, ps, pc, tier):
        return lab_.true_s[a["idx"]].copy(), pc

    go("CEIL score=true", transform=t_truescore, note="realised-score oracle (label noise included)")

    def t_truegain(lab_, a, ps, pc, tier):
        ts = lab_.true_s[a["idx"]]
        d = np.column_stack([ts[:, 1] - ts[:, 0], ts[:, 2] - ts[:, 1]])
        q = np.column_stack([ps[:, 0], ps[:, 0] + d[:, 0], ps[:, 0] + d[:, 0] + d[:, 1]])
        return np.clip(q, -1, 2), pc

    go("CEIL gains=true (level kept)", transform=t_truegain,
       note="the only decision-relevant score channel (BRIEF2 s2)")

    def t_truelevel(lab_, a, ps, pc, tier):
        ts = lab_.true_s[a["idx"]]
        lvl = ts.mean(axis=1) - ps.mean(axis=1)
        return np.clip(ps + lvl[:, None], 0, 1), pc

    go("ABL level=true (gains kept)", transform=t_truelevel,
       note="allocator-invariant channel; should be worth ~nothing")

    def t_both(lab_, a, ps, pc, tier):
        return lab_.true_s[a["idx"]].copy(), lab_.true_c[a["idx"]].copy()

    go("CEIL score+cost=true", transform=t_both)

    # score = family mean (destroy within-family score ranking, keep cost head)
    def t_famscore(lab_, a, ps, pc, tier):
        idx = a["idx"]; fam = lab_.fam_arr[idx]
        out = ps.copy()
        for f in np.unique(fam):
            m = fam == f
            out[m] = ps[m].mean(axis=0)
        return out, pc

    go("ABL score=family-mean-of-pred", transform=t_famscore,
       note="how much of the score head is WITHIN-family resolution?")

    # -------------------------------------------------- constant sensitivity
    sweeps = {
        "legacy_w":       [0.0, 0.3, 0.6, 0.9, 1.0],
        "fam_w":          [0.0, 0.075, 0.15, 0.30, 0.50],
        "conf_scale":     [0.0, 0.125, 0.25, 0.50],
        "gain_alpha":     [0.0, 0.25, 0.5, 0.75, 1.0],
        "rank_beta":      [0.0, 0.2, 0.4, 0.6, 0.8],
        "blend_fast":     [0.0, 0.3, 0.6, 0.9],
        "blend_balanced": [0.0, 0.225, 0.45, 0.7],
        "blend_premium":  [0.0, 0.15, 0.3, 0.6],
    }
    sens = {}
    t0 = time.perf_counter()
    for k, vals in sweeps.items():
        rows = []
        for v in vals:
            r = B.run(lab, cv, arr, {k: v}, label=f"  {k}={v}", verbose=False)
            rows.append(dict(v=float(v), EV=r["EV"], dev=r["dev"],
                             safety={t: r["safety"][t] for t in TIERS}))
        sens[k] = rows
        dep = DEPLOYED_CFG[k]
        evs = [x["EV"] for x in rows]; dvs = [x["dev"] for x in rows]
        bi = int(np.argmax(evs))
        print(f"{k:16s} dep={dep:<6} EVrange={max(evs)-min(evs):.5f} "
              f"devrange={max(dvs)-min(dvs):.5f}  bestEV@{rows[bi]['v']:<6} "
              f"(EV {evs[bi]:.5f}, dev {dvs[bi]:.5f})  " +
              " ".join(f"{x['v']}:{x['EV']:.4f}/{x['dev']:.4f}" for x in rows), flush=True)
    print(f"[b06] sensitivity in {time.perf_counter()-t0:.0f}s")

    OUT.write_text(json.dumps(dict(ceilings=res, sensitivity=sens), indent=1, default=float),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
