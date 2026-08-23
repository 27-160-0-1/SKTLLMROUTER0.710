# SPDX-License-Identifier: Apache-2.0
"""b02 step 2 - the (fraction to mid, fraction to k1) frontier at every tier.

For a grid of counts the policy is: take the n_k1 items with the highest
light->k1 chord efficiency, then among the rest the n_mid items with the highest
light->mid efficiency.  Rankings come from
  A) the deployed honest predictions  (what we could actually ship), and
  B) the EB posterior labels          (what an expectation-oracle would do).
For every grid point we report the bootstrap expected final tier score with the
bust priced in, the bust probability, and the realised budget ratio.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b02lib", HERE / "b02-balanced-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_SAFETY  # noqa

TIER = sys.argv[1] if len(sys.argv) > 1 else "balanced"
MULT = MULTS[TIER]
lab = Lab()
cv, arr = L.load_stage("base")
eb = np.load("reports/lab/b02_eb.npz")["eb"]

ci = cv["idx"]
ts = lab.true_s[ci]; tc = lab.true_c[ci]; m = len(ci)
ps, pc = lab.compose(cv, DEPLOYED_CFG, TIER)

SEEDS = (7, 17, 23); NBOOT = 300; SIZE = 880
samples = np.concatenate([np.asarray(lab.samples_for(m, s, NBOOT, SIZE)) for s in SEEDS])
B = len(samples)
print(f"[b02] tier={TIER} mult={MULT} OOF rows={m} bootstrap batches={B}")


def rank_keys(S, C):
    d1s = S[:, 1] - S[:, 0]; d1c = np.maximum(C[:, 1] - C[:, 0], 1e-12)
    dcs = S[:, 2] - S[:, 0]; dcc = np.maximum(C[:, 2] - C[:, 0], 1e-12)
    return d1s / d1c, dcs / dcc


def frontier(e_m, e_k, fm_grid, fk_grid, label):
    """For each (frac_mid, frac_k1) return EV, bust, mean realised ratio."""
    # pre-sort inside every bootstrap batch once
    out = {}
    SM = samples                                   # (B, 880) item indices
    ekb = e_k[SM]; emb = e_m[SM]
    ok = np.argsort(-ekb, axis=1, kind="stable")   # k1 priority order
    tcb = tc[SM]; tsb = ts[SM]; ebb = eb[ci][SM]
    base = tcb[:, :, 0].sum(axis=1)
    r = np.arange(B)[:, None]
    for fk in fk_grid:
        nk = int(round(fk * SIZE))
        ksel = ok[:, :nk]
        mask = np.zeros((B, SIZE), bool)
        if nk:
            mask[r, ksel] = True
        em2 = np.where(mask, -np.inf, emb)
        om = np.argsort(-em2, axis=1, kind="stable")
        # cumulative deltas along the mid order
        dk_c = (np.take_along_axis(tcb[:, :, 2], ksel, axis=1)
                - np.take_along_axis(tcb[:, :, 0], ksel, axis=1)).sum(axis=1) if nk else np.zeros(B)
        dk_s = (np.take_along_axis(tsb[:, :, 2], ksel, axis=1)
                - np.take_along_axis(tsb[:, :, 0], ksel, axis=1)).sum(axis=1) if nk else np.zeros(B)
        dk_e = (np.take_along_axis(ebb[:, :, 2], ksel, axis=1)
                - np.take_along_axis(ebb[:, :, 0], ksel, axis=1)).sum(axis=1) if nk else np.zeros(B)
        dm_c = np.cumsum(np.take_along_axis(tcb[:, :, 1] - tcb[:, :, 0], om, axis=1), axis=1)
        dm_s = np.cumsum(np.take_along_axis(tsb[:, :, 1] - tsb[:, :, 0], om, axis=1), axis=1)
        dm_e = np.cumsum(np.take_along_axis(ebb[:, :, 1] - ebb[:, :, 0], om, axis=1), axis=1)
        z = np.zeros((B, 1))
        dm_c = np.concatenate([z, dm_c], axis=1)
        dm_s = np.concatenate([z, dm_s], axis=1)
        dm_e = np.concatenate([z, dm_e], axis=1)
        s0 = tsb[:, :, 0].sum(axis=1); e0 = ebb[:, :, 0].sum(axis=1)
        for fm in fm_grid:
            nm = int(round(fm * (SIZE - nk)))
            nm = min(nm, SIZE - nk)
            cost = base + dk_c + dm_c[:, nm]
            ratio = cost / base
            sc = (s0 + dk_s + dm_s[:, nm]) / SIZE
            se = (e0 + dk_e + dm_e[:, nm]) / SIZE
            bust = ratio > MULT
            out[(fm, fk)] = dict(
                ev=float(np.mean(np.where(bust, 0.0, sc))),
                ev_eb=float(np.mean(np.where(bust, 0.0, se))),
                raw=float(np.mean(sc)), raw_eb=float(np.mean(se)),
                bust=float(np.mean(bust)), ratio=float(np.mean(ratio)),
                ratio_sd=float(np.std(ratio)), nk=nk, nm=nm)
    return out


e_m_pred, e_k_pred = rank_keys(ps, pc)
e_m_eb, e_k_eb = rank_keys(eb[ci], tc)

FM = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FK = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]

for label, (em, ek) in [("PRED", (e_m_pred, e_k_pred)), ("EB-oracle", (e_m_eb, e_k_eb))]:
    res = frontier(em, ek, FM, FK, label)
    print(f"\n=== frontier ({label} ranking), tier={TIER} : EV (bust%) ===")
    print("  fk\\fm " + " ".join(f"{f:>13.2f}" for f in FM))
    for fk in FK:
        cells = []
        for fm in FM:
            d = res[(fm, fk)]
            cells.append(f"{d['ev']:.4f}({d['bust']*100:4.1f})")
        print(f"  {fk:5.2f} " + " ".join(f"{c:>13s}" for c in cells))
    print(f"--- same grid: mean realised ratio (sd) ---")
    print("  fk\\fm " + " ".join(f"{f:>13.2f}" for f in FM))
    for fk in FK:
        cells = [f"{res[(fm,fk)]['ratio']:.3f}({res[(fm,fk)]['ratio_sd']:.3f})" for fm in FM]
        print(f"  {fk:5.2f} " + " ".join(f"{c:>13s}" for c in cells))
    best = max(res.items(), key=lambda kv: kv[1]["ev"])
    beb = max(res.items(), key=lambda kv: kv[1]["ev_eb"])
    print(f"  best EV      fm={best[0][0]:.2f} fk={best[0][1]:.2f} -> EV={best[1]['ev']:.4f} "
          f"bust={best[1]['bust']*100:.1f}% ratio={best[1]['ratio']:.3f} "
          f"nm={best[1]['nm']} nk={best[1]['nk']}")
    print(f"  best EV(EB)  fm={beb[0][0]:.2f} fk={beb[0][1]:.2f} -> EV_eb={beb[1]['ev_eb']:.4f} "
          f"bust={beb[1]['bust']*100:.1f}% ratio={beb[1]['ratio']:.3f} "
          f"nm={beb[1]['nm']} nk={beb[1]['nk']}")

# ---- where does the deployed allocator sit on this frontier? -----------------
print("\n=== deployed allocator position (OOF rows, bootstrap) ===")
for s in [DEPLOYED_SAFETY[TIER]]:
    pick = L.P.exact_allocate(ps[samples], pc[samples], MULT, s)
    r = np.arange(B)[:, None]
    cnt = np.stack([(pick == k).sum(axis=1) for k in range(3)], axis=1).mean(axis=0)
    tcb = tc[samples]; tsb = ts[samples]
    real = np.take_along_axis(tcb, pick[:, :, None], axis=2)[:, :, 0].sum(axis=1)
    ratio = real / tcb[:, :, 0].sum(axis=1)
    sc = np.take_along_axis(tsb, pick[:, :, None], axis=2)[:, :, 0].mean(axis=1)
    bust = ratio > MULT
    print(f"  s={s}: mean L/M/K={cnt.round(1)} (fm={cnt[1]/SIZE:.3f} fk={cnt[2]/SIZE:.3f}) "
          f"EV={np.mean(np.where(bust,0.,sc)):.4f} bust={bust.mean()*100:.1f}% "
          f"ratio={ratio.mean():.3f}({ratio.std():.3f})")
