# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b02 step 3 - which predictor would let us concentrate on k1 safely?

Same fixed-count frontier as step 2, but the ranking keys are built from the
2x2 of {predicted, EB-true} score x {predicted, true} cost.  The budget ratio
and the score are always evaluated on the truth.  Run for every tier so the
"is 2.0x the tier where cost prediction matters most" question gets an answer.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b02lib", HERE / "b02-balanced-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, TIERS, MULTS, W, DEPLOYED_CFG, DEPLOYED_SAFETY  # noqa

lab = Lab()
cv, arr = L.load_stage("base")
eb = np.load("reports/lab/b02_eb.npz")["eb"]
ci = cv["idx"]
ts = lab.true_s[ci]; tc = lab.true_c[ci]; m = len(ci)
SEEDS = (7, 17, 23); NBOOT = 300; SIZE = 880
samples = np.concatenate([np.asarray(lab.samples_for(m, s, NBOOT, SIZE)) for s in SEEDS])
B = len(samples)
tcb = tc[samples]; tsb = ts[samples]; ebb = eb[ci][samples]
base = tcb[:, :, 0].sum(axis=1)
s0 = tsb[:, :, 0].sum(axis=1); e0 = ebb[:, :, 0].sum(axis=1)
r = np.arange(B)[:, None]

FM = np.arange(0.0, 1.001, 0.05)
FK = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]


def keys(S, C):
    d1s = S[:, 1] - S[:, 0]; d1c = np.maximum(C[:, 1] - C[:, 0], 1e-12)
    dcs = S[:, 2] - S[:, 0]; dcc = np.maximum(C[:, 2] - C[:, 0], 1e-12)
    return d1s / d1c, dcs / dcc


def sweep(e_m, e_k, mult):
    emb = e_m[samples]; ekb = e_k[samples]
    ok = np.argsort(-ekb, axis=1, kind="stable")
    best = {}
    for fk in FK:
        nk = int(round(fk * SIZE))
        ksel = ok[:, :nk]
        mask = np.zeros((B, SIZE), bool)
        if nk:
            mask[r, ksel] = True
        om = np.argsort(-np.where(mask, -np.inf, emb), axis=1, kind="stable")
        gk = (lambda A: (np.take_along_axis(A[:, :, 2], ksel, axis=1)
                         - np.take_along_axis(A[:, :, 0], ksel, axis=1)).sum(axis=1)
              if nk else np.zeros(B))
        dk_c, dk_s, dk_e = gk(tcb), gk(tsb), gk(ebb)
        cm = (lambda A: np.concatenate([np.zeros((B, 1)), np.cumsum(
            np.take_along_axis(A[:, :, 1] - A[:, :, 0], om, axis=1), axis=1)], axis=1))
        dm_c, dm_s, dm_e = cm(tcb), cm(tsb), cm(ebb)
        row = []
        for fm in FM:
            nm = min(int(round(fm * (SIZE - nk))), SIZE - nk)
            ratio = (base + dk_c + dm_c[:, nm]) / base
            sc = (s0 + dk_s + dm_s[:, nm]) / SIZE
            se = (e0 + dk_e + dm_e[:, nm]) / SIZE
            bust = ratio > mult
            row.append((float(np.mean(np.where(bust, 0.0, sc))),
                        float(np.mean(np.where(bust, 0.0, se))),
                        float(np.mean(bust)), float(np.mean(ratio)), float(np.std(ratio)), nm))
        i = int(np.argmax([x[0] for x in row]))
        best[fk] = dict(fm=float(FM[i]), ev=row[i][0], ev_eb=row[i][1], bust=row[i][2],
                        ratio=row[i][3], ratio_sd=row[i][4], nm=row[i][5], nk=nk)
    return best


for tier in TIERS:
    mult = MULTS[tier]
    ps, pc = lab.compose(cv, DEPLOYED_CFG, tier)
    variants = {
        "pred s / pred c": keys(ps, pc),
        "pred s / TRUE c": keys(ps, tc),
        "EB   s / pred c": keys(eb[ci], pc),
        "EB   s / TRUE c": keys(eb[ci], tc),
    }
    print(f"\n################ tier={tier} (cap {mult}) ################")
    print(f"  {'ranking':16s} {'fk':>5s} {'nk':>4s} {'fm*':>5s} {'nm':>4s} "
          f"{'EV':>8s} {'EV_EB':>8s} {'bust%':>6s} {'ratio':>7s} {'sd':>6s}")
    for name, (em, ek) in variants.items():
        bb = sweep(em, ek, mult)
        star = max(bb.items(), key=lambda kv: kv[1]["ev"])[0]
        for fk in FK:
            d = bb[fk]
            mark = " *" if fk == star else "  "
            print(f"  {name:16s} {fk:5.2f} {d['nk']:4d} {d['fm']:5.2f} {d['nm']:4d} "
                  f"{d['ev']:8.4f} {d['ev_eb']:8.4f} {d['bust']*100:6.1f} {d['ratio']:7.3f} "
                  f"{d['ratio_sd']:6.3f}{mark}")
        print()
