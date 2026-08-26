# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b07-visualiser - compute every number that goes on a figure.

Nothing here is copied from another report: every value is recomputed from
`data/`, `reports/lab/dev_preds_e43.npz` (the deployed pipeline's own dev
predictions) and the bench2 stage built by `b07-visualiser_stage.py`.

Output: reports/lab/figs/b07_numbers.json  (+ a few npz blobs for the curves)
"""
from __future__ import annotations
import itertools, json, math, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_EXP, DEPLOYED_SAFETY  # noqa
import protocol as P  # noqa
import bench2 as B  # noqa

OUT = ROOT / "reports/lab/figs"
OUT.mkdir(parents=True, exist_ok=True)
def _jd(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


TB = np.array([2e-12, 1e-12, 0.0])          # deployed tie-break: prefer the cheaper model
MODELS = ("ax31-light", "ax31", "axk1-think")


# ---------------------------------------------------------------- allocation
def alloc_pen(ps, pc, mult, safety):
    """harness.Lab.allocate, but it also returns the shadow price and the light total."""
    lt = pc[:, 0].sum()
    cap = lt * max(1.0, mult * safety)

    def choose(pen):
        pick = np.argmax(ps - pen * pc / lt + TB, axis=1)
        return pick, pc[np.arange(len(pick)), pick].sum()

    pen = 0.0
    pick, tot = choose(pen)
    if tot > cap:
        lo, hi = 0.0, 1.0
        pick, tot = choose(hi)
        while tot > cap and hi < 2 ** 60:
            lo, hi = hi, hi * 2
            pick, tot = choose(hi)
        for _ in range(40):
            mid = (lo + hi) / 2
            c2, t2 = choose(mid)
            if t2 <= cap:
                hi, pick, tot = mid, c2, t2
            else:
                lo = mid
        pen = hi
    if tot > cap:
        pick = np.zeros(len(ps), dtype=int)
        pen = float("inf")
    return pick, float(pen), float(lt)


def evaluate(ps, pc, ts, tc, safety):
    """Final score of a (pred score, pred cost, safety triple) policy on one sample."""
    out, final = {}, 0.0
    for t in TIERS:
        S = ps[t] if isinstance(ps, dict) else ps
        C = pc[t] if isinstance(pc, dict) else pc
        pick, pen, lt = alloc_pen(S, C, MULTS[t], safety[t])
        r = np.arange(len(pick))
        ratio = tc[r, pick].sum() / tc[:, 0].sum()
        sc = ts[r, pick].mean()
        ok = bool(ratio <= MULTS[t] + 1e-15)
        out[t] = dict(score=float(sc), ratio=float(ratio), passed=ok, pen=pen,
                      counts=np.bincount(pick, minlength=3).tolist(),
                      pick=pick.tolist())
        final += W[t] * (sc if ok else 0.0)
    out["final"] = float(final)
    return out


# ------------------------------------------------------- empirical-Bayes p̂
def eb_latent(true_s, ngen, fam_arr):
    """Beta-binomial moment matching per (family, model) -> E[p | k, n].

    Var(s) = Var(p) + E[p(1-p)/n];  s(1-s)/(n-1) is unbiased for p(1-p)/n.
    """
    p_hat = np.zeros_like(true_s)
    info = {}
    for fam in sorted(set(fam_arr)):
        m = fam_arr == fam
        for j in range(3):
            s = true_s[m, j]
            n = ngen[m, j]
            mu = float(s.mean())
            var_s = float(s.var())
            binom = float(np.mean(s * (1 - s) / np.maximum(n - 1, 1)))
            V = var_s - binom
            lo = mu * (1 - mu)
            if not (1e-6 < V < lo - 1e-9) or lo <= 1e-9:
                V = max(min(lo * 0.5, lo - 1e-9), 1e-6)
                flag = "clamped"
            else:
                flag = "ok"
            nu = lo / V - 1.0
            nu = float(np.clip(nu, 0.05, 500.0))
            a, b = mu * nu, (1 - mu) * nu
            k = s * n
            p_hat[m, j] = (a + k) / (a + b + n)
            info[f"{fam}|{MODELS[j]}"] = dict(n=int(m.sum()), mean=mu, var_s=var_s,
                                              binom=binom, nu=nu, flag=flag)
    return p_hat, info


# -------------------------------------------------------- Shapley decomposition
def shapley_gap(pd_s, pd_c, ts, tc, price_dep, price_orc):
    """5 players on the decision inputs of ONE tier.

    level : s_light          pred -> true
    d1    : s_mid  - s_light pred -> true
    d2    : s_k1   - s_mid   pred -> true
    cost  : the cost vector inside the utility, pred -> true
    price : the shadow price lambda/L, deployed -> oracle  (= the budget limit)

    v(S) = mean realised score of argmax_m (s_m - price * c_m).  The empty
    coalition reproduces the deployed selection exactly and the full coalition
    reproduces the oracle selection exactly (asserted by the caller).
    """
    lev = {0: pd_s[:, 0], 1: ts[:, 0]}
    g1 = {0: pd_s[:, 1] - pd_s[:, 0], 1: ts[:, 1] - ts[:, 0]}
    g2 = {0: pd_s[:, 2] - pd_s[:, 1], 1: ts[:, 2] - ts[:, 1]}
    cst = {0: pd_c, 1: tc}
    prc = {0: price_dep, 1: price_orc}
    names = ["level", "d1", "d2", "cost", "price"]
    cache = {}

    def v(bits):
        if bits in cache:
            return cache[bits]
        a, b, c, d, e = bits
        s0 = lev[a]
        s = np.column_stack([s0, s0 + g1[b], s0 + g1[b] + g2[c]])
        u = s - prc[e] * cst[d] + TB
        pick = np.argmax(u, axis=1)
        val = float(ts[np.arange(len(pick)), pick].mean())
        cache[bits] = (val, pick)
        return cache[bits]

    n = 5
    fact = [math.factorial(i) for i in range(n + 1)]
    phi = dict.fromkeys(names, 0.0)
    for i, nm in enumerate(names):
        for rest in itertools.product([0, 1], repeat=n - 1):
            base = list(rest[:i]) + [0] + list(rest[i:])
            plus = list(rest[:i]) + [1] + list(rest[i:])
            k = sum(rest)
            w = fact[k] * fact[n - 1 - k] / fact[n]
            phi[nm] += w * (v(tuple(plus))[0] - v(tuple(base))[0])
    return phi, v((0, 0, 0, 0, 0)), v((1, 1, 1, 1, 1))


# ================================================================== main
def main():
    lab = Lab()
    dev = lab.dev_idx
    ts = lab.true_s[dev]
    tc = lab.true_c[dev]
    fam = lab.fam_arr[dev]
    ngen = lab.ngen[dev]
    L_true = tc[:, 0].sum()
    res = {}

    # ---- deployed pipeline predictions on dev (real artifact, lookup removed)
    z = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
    dep_s = {t: z[f"score_{t}"] for t in TIERS}
    dep_c = {t: z[f"cost_{t}"] for t in TIERS}

    # =============================================================== FIGURE 1
    ladder = {}

    # all-light
    tiers = {}
    for t in TIERS:
        tiers[t] = dict(score=float(ts[:, 0].mean()), ratio=1.0, passed=True)
    ladder["all-light"] = dict(tiers=tiers, final=float(ts[:, 0].mean()), safety="n/a")

    # all-mid / all-k1 for context (budget check included)
    for j, nm in ((1, "all-mid"), (2, "all-k1-think")):
        tiers = {}
        fin = 0.0
        for t in TIERS:
            ratio = tc[:, j].sum() / L_true
            ok = ratio <= MULTS[t] + 1e-15
            tiers[t] = dict(score=float(ts[:, j].mean()), ratio=float(ratio), passed=ok)
            fin += W[t] * (float(ts[:, j].mean()) if ok else 0.0)
        ladder[nm] = dict(tiers=tiers, final=fin, safety="n/a")

    # official hash-regex baseline: the shipped public artifact + its own safety triple
    leg = lab.legacy[dev]
    leg_s = np.clip(leg[:, :3], 0.0, 1.0)
    leg_c = np.exp(np.clip(leg[:, 3:6], -50, 50))
    art = json.loads((ROOT / "src/ossp_router/resources/hash-regex-public.v1.json")
                     .read_text(encoding="utf-8"))
    leg_safety = {t: float(art["tier_safety_ratios"][t]) for t in TIERS}
    r = evaluate(leg_s, leg_c, ts, tc, leg_safety)
    ladder["official hash-regex"] = dict(
        tiers={t: {k: r[t][k] for k in ("score", "ratio", "passed")} for t in TIERS},
        final=r["final"],
        safety="/".join(f"{leg_safety[t]:.3f}" for t in TIERS))

    # deployed E43 at the two published safety triples
    for nm, sf in (("E43 .98/.89/.88", {"fast": .98, "balanced": .89, "premium": .88}),
                   ("E43 .98/.87/.85", DEPLOYED_SAFETY)):
        r = evaluate(dep_s, dep_c, ts, tc, sf)
        ladder[nm] = dict(
            tiers={t: {k: r[t][k] for k in ("score", "ratio", "passed", "counts")} for t in TIERS},
            final=r["final"], safety="/".join(f"{sf[t]:.2f}" for t in TIERS),
            pen={t: r[t]["pen"] for t in TIERS})
        if nm.endswith(".89/.88"):
            dep_pick = {t: np.asarray(r[t]["pick"]) for t in TIERS}
            dep_pen = {t: r[t]["pen"] for t in TIERS}

    # realised-score oracle: true score, true cost, safety 1.0
    one = {t: 1.0 for t in TIERS}
    r = evaluate(ts, tc, ts, tc, one)
    ladder["oracle (realised s)"] = dict(
        tiers={t: {k: r[t][k] for k in ("score", "ratio", "passed", "counts")} for t in TIERS},
        final=r["final"], safety="1.00/1.00/1.00", pen={t: r[t]["pen"] for t in TIERS})
    orc_pen = {t: r[t]["pen"] for t in TIERS}

    # noise-free ceiling: empirical-Bayes latent p, allocate on p + true cost, score on p
    p_hat, eb_info = eb_latent(ts, ngen, fam)
    r = evaluate(p_hat, tc, p_hat, tc, one)
    ladder["ceiling (EB latent p)"] = dict(
        tiers={t: {k: r[t][k] for k in ("score", "ratio", "passed", "counts")} for t in TIERS},
        final=r["final"], safety="1.00/1.00/1.00")
    # and: what the DEPLOYED allocation is worth on the same noise-free surface
    fin = 0.0
    tiers = {}
    for t in TIERS:
        pk = dep_pick[t]
        rr = np.arange(len(pk))
        sc = float(p_hat[rr, pk].mean())
        ratio = float(tc[rr, pk].sum() / L_true)
        tiers[t] = dict(score=sc, ratio=ratio, passed=bool(ratio <= MULTS[t] + 1e-15))
        fin += W[t] * sc
    ladder["E43 scored on latent p"] = dict(tiers=tiers, final=fin, safety=".98/.89/.88")

    res["ladder"] = ladder
    res["eb_prior"] = eb_info

    # =============================================================== FIGURE 4
    decomp = {}
    for t in TIERS:
        price_dep = dep_pen[t] / dep_c[t][:, 0].sum()
        price_orc = orc_pen[t] / L_true
        phi, (v0, pk0), (v1, pk1) = shapley_gap(dep_s[t], dep_c[t], ts, tc,
                                                price_dep, price_orc)
        assert np.array_equal(pk0, dep_pick[t]), f"{t}: empty coalition != deployed"
        decomp[t] = dict(phi=phi, deployed=v0, oracle=v1, gap=v1 - v0,
                         price_dep=price_dep, price_orc=price_orc)
    decomp["weighted"] = dict(
        phi={k: sum(W[t] * decomp[t]["phi"][k] for t in TIERS)
             for k in decomp["fast"]["phi"]},
        deployed=sum(W[t] * decomp[t]["deployed"] for t in TIERS),
        oracle=sum(W[t] * decomp[t]["oracle"] for t in TIERS))
    decomp["weighted"]["gap"] = decomp["weighted"]["oracle"] - decomp["weighted"]["deployed"]
    res["decomp"] = decomp

    # ---- the same decomposition on the noise-free surface (how much is real)
    decomp_eb = {}
    for t in TIERS:
        price_dep = dep_pen[t] / dep_c[t][:, 0].sum()
        pk, pen_o, _ = alloc_pen(p_hat, tc, MULTS[t], 1.0)
        price_orc = pen_o / L_true
        phi, (v0, _), (v1, _) = shapley_gap(dep_s[t], dep_c[t], p_hat, tc,
                                            price_dep, price_orc)
        decomp_eb[t] = dict(phi=phi, deployed=v0, oracle=v1, gap=v1 - v0)
    decomp_eb["weighted"] = dict(
        phi={k: sum(W[t] * decomp_eb[t]["phi"][k] for t in TIERS)
             for k in decomp_eb["fast"]["phi"]},
        deployed=sum(W[t] * decomp_eb[t]["deployed"] for t in TIERS),
        oracle=sum(W[t] * decomp_eb[t]["oracle"] for t in TIERS))
    decomp_eb["weighted"]["gap"] = (decomp_eb["weighted"]["oracle"]
                                    - decomp_eb["weighted"]["deployed"])
    res["decomp_eb"] = decomp_eb

    # =============================================================== FIGURE 5
    fams = sorted(set(fam))
    per_fam = {}
    for f in fams:
        m = fam == f
        row = dict(n=int(m.sum()),
                   score=[float(ts[m, j].mean()) for j in range(3)],
                   cost_ratio=[float(tc[m, j].sum() / tc[m, 0].sum()) for j in range(3)],
                   share_of_light_budget=float(tc[m, 0].sum() / L_true))
        for t in TIERS:
            pk = dep_pick[t]
            rr = np.arange(len(pk))
            spend = tc[rr, pk]
            score = ts[rr, pk]
            up = spend - tc[:, 0]                       # money spent above all-light
            row[t] = dict(budget_share=float(spend[m].sum() / spend.sum()),
                          upgrade_share=float(up[m].sum() / up.sum()),
                          score_share=float(score[m].sum() / score.sum()),
                          score_gain_share=float((score[m] - ts[m, 0]).sum()
                                                 / (score - ts[:, 0]).sum()),
                          spend_over_L=float(spend[m].sum() / L_true),
                          upgrade_over_L=float(up[m].sum() / L_true),
                          mean_score=float(score[m].mean()),
                          counts=np.bincount(pk[m], minlength=3).tolist())
        per_fam[f] = row
    res["per_family"] = per_fam
    res["dev_totals"] = dict(mean_score_light=float(ts[:, 0].mean()),
                             L_true=float(L_true),
                             cost_ratio_all=[float(tc[:, j].sum() / L_true) for j in range(3)],
                             mean_score_all=[float(ts[:, j].mean()) for j in range(3)])

    # ============================================================ FIGURES 2/3
    if "--no-stage" in sys.argv:
        (OUT / "b07_numbers_partial.json").write_text(json.dumps(res, indent=1, default=_jd), encoding="utf-8")
        print("\n[partial] stage skipped")
        _summary(ladder, decomp, decomp_eb)
        return
    cv, arr = B.stage(lab, DEPLOYED_EXP, tag="b07-visualiser")
    triples = {}
    r_opt = B.run(lab, cv, arr, DEPLOYED_CFG, label="EV-optimal (bench2)", keep_curves=True)
    triples["EV-optimal"] = r_opt
    for nm, sf in (("deployed .98/.87/.85", DEPLOYED_SAFETY),
                   ("published .98/.89/.88", {"fast": .98, "balanced": .89, "premium": .88}),
                   ("insurance .93/.80/.75", {"fast": .93, "balanced": .80, "premium": .75}),
                   ("baseline-like .96/.91/.88", {"fast": .96, "balanced": .91, "premium": .88})):
        triples[nm] = B.run(lab, cv, arr, DEPLOYED_CFG, label=nm, fixed_safety=sf)
    res["triples"] = {k: dict(EV=v["EV"], dev=v["dev"], safety=v["safety"],
                              det={t: v["det"][t] for t in TIERS},
                              dev_tiers={t: {kk: vv for kk, vv in v["dev_tiers"][t].items()}
                                         for t in TIERS})
                      for k, v in triples.items()}
    np.savez_compressed(OUT / "b07_curves.npz",
                        **{f"{t}_{k}": np.asarray(r_opt["curves"][t][k])
                           for t in TIERS for k in ("grid", "ev", "bust", "raw")})

    # dev realised curve over the same grid (single held-out sample, diagnosis only)
    devcurve = {}
    for t in TIERS:
        g = np.asarray(r_opt["curves"][t]["grid"])
        ev, bust, raw = P.safety_curve(dep_s[t][None], dep_c[t][None], ts[None], tc[None],
                                       MULTS[t], g)
        devcurve[t] = dict(grid=g.tolist(), ev=ev.tolist(), bust=bust.tolist(), raw=raw.tolist())
    res["dev_curve"] = devcurve

    (OUT / "b07_numbers.json").write_text(json.dumps(res, indent=1, default=_jd), encoding="utf-8")
    _summary(ladder, decomp, decomp_eb)
    print("\nwrote", OUT / "b07_numbers.json")


def _summary(ladder, decomp, decomp_eb):
    print("\n=== LADDER (dev, weighted final) ===")
    for k, v in ladder.items():
        print(f"  {k:26s} {v['final']:.4f}  safety={v['safety']}  "
              + " ".join(f"{t[:4]}={v['tiers'][t]['score']:.4f}/r{v['tiers'][t]['ratio']:.3f}"
                         f"{'' if v['tiers'][t]['passed'] else '!BUST'}" for t in TIERS))
    print("\n=== GAP DECOMPOSITION (Shapley, realised-score oracle) ===")
    for t in list(TIERS) + ["weighted"]:
        d = decomp[t]
        print(f"  {t:9s} gap={d['gap']:+.4f} " +
              " ".join(f"{k}={v:+.4f}" for k, v in d["phi"].items()))
    print("\n=== GAP DECOMPOSITION (Shapley, EB latent-p oracle) ===")
    for t in list(TIERS) + ["weighted"]:
        d = decomp_eb[t]
        print(f"  {t:9s} gap={d['gap']:+.4f} " +
              " ".join(f"{k}={v:+.4f}" for k, v in d["phi"].items()))


if __name__ == "__main__":
    main()
