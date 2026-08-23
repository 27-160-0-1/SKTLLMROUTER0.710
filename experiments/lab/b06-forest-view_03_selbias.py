# SPDX-License-Identifier: Apache-2.0
"""b06 forest-view: how much measured gain must a candidate show before adoption?

Method (no oracle anywhere):
  * draw K random post-hoc constant vectors from the plausible box,
  * score each one under bench2 -> (EV on Train-OOF, dev held-out),
  * measure corr(EV, dev), the OLS transfer slope, and the best-of-k order
    statistic: if the fleet screens k candidates on EV and adopts the argmax,
    how much dev does it actually get?
  * measure the noise floors: EV seed-noise, and the PAIRED bootstrap sd of a
    dev difference (the scale against which any "+0.002 dev" must be judged).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP
import bench2 as B
import protocol as P

OUT = Path("reports/lab/b06_selbias.json")
BOX = {"legacy_w": (0.0, 1.0), "fam_w": (0.0, 0.5), "conf_scale": (0.0, 0.5),
       "gain_alpha": (0.0, 1.0), "rank_beta": (0.0, 0.8),
       "blend_fast": (0.0, 0.9), "blend_balanced": (0.0, 0.8), "blend_premium": (0.0, 0.7)}
K = 120


def dev_pick_scores(lab, arr, cfg, safety):
    """Per-item realised score and cost of the dev picks, per tier."""
    idx = arr["idx"]; r = np.arange(len(idx))
    out = {}
    for t in TIERS:
        ps, pc = lab.compose(arr, cfg, t)
        pick = lab.allocate(ps, pc, MULTS[t], safety[t])
        out[t] = (lab.true_s[idx][r, pick], lab.true_c[idx][r, pick], lab.true_c[idx][:, 0])
    return out


def paired_dev_boot(lab, arr, cfgA, sA, cfgB, sB, nboot=2000, seed=5):
    """Bootstrap sd of dev(A) - dev(B) over 880-item resamples of Dev itself."""
    A = dev_pick_scores(lab, arr, cfgA, sA); Bp = dev_pick_scores(lab, arr, cfgB, sB)
    rng = np.random.default_rng(seed); m = len(arr["idx"])
    dif = np.zeros(nboot); fa = np.zeros(nboot); fb = np.zeros(nboot)
    for b in range(nboot):
        s = rng.integers(0, m, size=m)
        va = vb = 0.0
        for t in TIERS:
            sa, ca, la = A[t]; sb, cb, lb = Bp[t]
            base = la[s].sum()
            ra = ca[s].sum() / base; rb = cb[s].sum() / base
            va += W[t] * (sa[s].mean() if ra <= MULTS[t] else 0.0)
            vb += W[t] * (sb[s].mean() if rb <= MULTS[t] else 0.0)
        fa[b] = va; fb[b] = vb; dif[b] = va - vb
    return dict(sd_A=float(fa.std()), sd_B=float(fb.std()), sd_diff=float(dif.std()),
                mean_diff=float(dif.mean()))


def main():
    lab = Lab()
    cv, arr = B.stage(lab, DEPLOYED_EXP, tag="base")
    rng = np.random.default_rng(20260819)
    t0 = time.perf_counter()

    rows = []
    for i in range(K):
        cfg = {k: float(rng.uniform(*v)) for k, v in BOX.items()}
        r = B.run(lab, cv, arr, cfg, label=f"r{i}", verbose=False)
        rows.append(dict(cfg=cfg, EV=r["EV"], dev=r["dev"],
                         safety={t: r["safety"][t] for t in TIERS},
                         bust={t: r["det"][t]["bust"] for t in TIERS}))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{K}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    ev = np.array([x["EV"] for x in rows]); dv = np.array([x["dev"] for x in rows])
    nbust = sum(1 for x in rows if x["dev"] < 0.55)
    print(f"\nK={K} random constant vectors")
    print(f"  EV   mean {ev.mean():.6f} sd {ev.std():.6f}  range [{ev.min():.6f}, {ev.max():.6f}]")
    print(f"  dev  mean {dv.mean():.6f} sd {dv.std():.6f}  range [{dv.min():.6f}, {dv.max():.6f}]"
          f"  ({nbust} of {K} busted a tier on dev)")
    ok = dv > 0.55
    for name, mask in (("all", np.ones(K, bool)), ("non-bust", ok)):
        e, d = ev[mask], dv[mask]
        if len(e) < 5:
            continue
        pear = float(np.corrcoef(e, d)[0, 1])
        sp = float(np.corrcoef(np.argsort(np.argsort(e)), np.argsort(np.argsort(d)))[0, 1])
        slope = float(np.polyfit(e, d, 1)[0])
        print(f"  [{name:8s} n={len(e):3d}] pearson(EV,dev)={pear:+.3f} spearman={sp:+.3f} "
              f"OLS d(dev)/d(EV)={slope:+.3f}")

    # ---------- best-of-k order statistic: screening on EV, paid on dev
    print("\n  best-of-k screening on EV (2,000 draws each):")
    bok = {}
    r2 = np.random.default_rng(3)
    for k in (1, 2, 3, 5, 10, 20, 40):
        gEV = np.zeros(2000); gdev = np.zeros(2000)
        for b in range(2000):
            s = r2.choice(K, size=k, replace=False)
            j = s[int(np.argmax(ev[s]))]
            gEV[b] = ev[j] - ev.mean(); gdev[b] = dv[j] - dv.mean()
        bok[k] = dict(gain_EV=float(gEV.mean()), gain_dev=float(gdev.mean()),
                      transfer=float(gdev.mean() / gEV.mean()) if gEV.mean() else 0.0,
                      dev_sd=float(gdev.std()))
        print(f"    k={k:3d}  E[EV gain]={gEV.mean():+.6f}  E[dev gain]={gdev.mean():+.6f}"
              f"  transfer={bok[k]['transfer']:+.3f}  sd(dev gain)={gdev.std():.6f}")

    # ---------- noise floors
    print("\n  noise floors:")
    seeds_sets = [(7, 17, 23), (31, 41, 53), (61, 71, 83), (97, 101, 103)]
    evs = []
    for ss in seeds_sets:
        r = B.run(lab, cv, arr, None, label="", verbose=False, seeds=ss)
        evs.append(r["EV"])
    print(f"    EV over 4 disjoint bootstrap-seed triples: {[f'{x:.6f}' for x in evs]}"
          f"  sd={np.std(evs):.6f}")

    base = B.run(lab, cv, arr, None, verbose=False)
    pb = {}
    for lbl, cfg in (("legacy_w=0.0", dict(legacy_w=0.0)),
                     ("fam_w=0.50", dict(fam_w=0.5)),
                     ("blend_balanced=0.0", dict(blend_balanced=0.0)),
                     ("gain_alpha=0.0", dict(gain_alpha=0.0))):
        r = B.run(lab, cv, arr, cfg, verbose=False)
        c = dict(DEPLOYED_CFG, **cfg)
        pb[lbl] = paired_dev_boot(lab, arr, c, r["safety"], dict(DEPLOYED_CFG), base["safety"])
        pb[lbl]["dev_diff_point"] = r["dev"] - base["dev"]
        print(f"    {lbl:20s} dev diff {r['dev']-base['dev']:+.6f}  "
              f"paired boot sd {pb[lbl]['sd_diff']:.6f}  "
              f"|t|={abs(r['dev']-base['dev'])/max(pb[lbl]['sd_diff'],1e-9):.2f}")
    print(f"    marginal dev bootstrap sd (deployed cfg) = {pb['legacy_w=0.0']['sd_B']:.6f}")

    OUT.write_text(json.dumps(dict(rows=rows, bok=bok, ev_seed=evs, paired=pb),
                              indent=1, default=float), encoding="utf-8")
    print(f"[b06] selbias in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
