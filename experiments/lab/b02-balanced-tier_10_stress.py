# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b02 step 10 - stress test of the balanced kappa2 policy + selection-bias
correction for the balanced-specific score constants.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b02lib", HERE / "b02-balanced-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, MULTS, DEPLOYED_CFG  # noqa
import protocol as P  # noqa

MULT = MULTS["balanced"]
lab = Lab()
cv, arr = L.load_stage("base")
ci = cv["idx"]; di = arr["idx"]; m = len(ci)
GRID = np.arange(0.60, 1.301, 0.005)
ps_c, pc_c = lab.compose(cv, DEPLOYED_CFG, "balanced")
ts_c = lab.true_s[ci]; tc_c = lab.true_c[ci]
RATE = {"light": (1.0, 4.0), "mid": (2.127, 8.509), "k1": (6.565, 26.26)}
itok = lab.itok[ci]; otok = lab.otok[ci]

# ---- stress scenarios: recompute the TRUE cost / score under a shift --------
def scen_costs(name):
    tc = tc_c.copy(); ts = ts_c.copy()
    if name == "nominal":
        return ts, tc, np.ones(m)
    if name == "longer-think":         # k1 emits 1.5x more output tokens
        o = otok.copy(); o[:, 2] *= 1.5
        tc = np.column_stack([
            itok[:, 0] * 1.0 / 1e6 + o[:, 0] * 4.0 / 1e6,
            itok[:, 1] * 2.127 / 1e6 + o[:, 1] * 8.509 / 1e6,
            itok[:, 2] * 6.565 / 1e6 + o[:, 2] * 26.26 / 1e6])
        return ts, tc, np.ones(m)
    if name == "longer-mid":           # mid emits 1.5x more output tokens
        o = otok.copy(); o[:, 1] *= 1.5
        tc = np.column_stack([
            itok[:, 0] * 1.0 / 1e6 + o[:, 0] * 4.0 / 1e6,
            itok[:, 1] * 2.127 / 1e6 + o[:, 1] * 8.509 / 1e6,
            itok[:, 2] * 6.565 / 1e6 + o[:, 2] * 26.26 / 1e6])
        return ts, tc, np.ones(m)
    if name == "harder-mix":           # resample weight 3x on the hard families
        hard = {"code", "dmmath", "hrmcr", "aime", "longdoc"}
        w = np.array([3.0 if f in hard else 1.0 for f in lab.fam_arr[ci]])
        return ts, tc, w / w.sum() * m
    if name == "easier-mix":
        easy = {"belebele", "truthfulqa", "gsm8k_or_other", "ruletaker"}
        w = np.array([3.0 if f in easy else 1.0 for f in lab.fam_arr[ci]])
        return ts, tc, w / w.sum() * m
    raise ValueError(name)


rng = np.random.default_rng(31337)
print("=== stress: fix (kappa2, safety) on the nominal Train-OOF EV, then re-score ===")
CANDS = [(1.0, 0.840), (1.0, 0.870), (1.35, 0.885), (1.5, 0.890), (1.5, 0.870), (2.0, 0.895)]
scens = ["nominal", "longer-think", "longer-mid", "harder-mix", "easier-mix"]
print(f"  {'k2':>5s} {'s':>6s} " + " ".join(f"{sc:>17s}" for sc in scens))
for k2, s in CANDS:
    pc = pc_c.copy(); pc[:, 2] *= k2
    cells = []
    for sc in scens:
        ts, tc, w = scen_costs(sc)
        if w is None or np.allclose(w, 1.0):
            smp = np.concatenate([np.asarray(lab.samples_for(m, sd, 400, 880))
                                  for sd in (7, 17, 23)])
        else:
            p = w / w.sum()
            smp = rng.choice(m, size=(1200, 880), replace=True, p=p)
        ev, bu, raw = P.safety_curve(ps_c[smp], pc[smp], ts[smp], tc[smp], MULT, np.array([s]))
        cells.append(f"{ev[0]:.4f}/{bu[0]*100:4.1f}%")
    print(f"  {k2:5.2f} {s:6.3f} " + " ".join(f"{c:>17s}" for c in cells))

# ---- selection bias for the balanced-specific SCORE constants ---------------
print("\n=== nested split-half for the balanced score constants "
      "(choose on A, score on B) ===")
SW = {"blend_balanced": [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0],
      "fam_w": [0.0, 0.075, 0.15, 0.25, 0.35],
      "gain_alpha": [0.0, 0.25, 0.5, 0.75, 1.0]}
comp = {}
for name, vals in SW.items():
    for v in vals:
        cfg = dict(DEPLOYED_CFG); cfg[name] = v
        comp[(name, v)] = lab.compose(cv, cfg, "balanced")
rng = np.random.default_rng(555)
NS = 24
for name, vals in SW.items():
    gains, oracles, picks = [], [], []
    for rep in range(NS):
        perm = rng.permutation(m)
        A, Bx = perm[: m // 2], perm[m // 2:]
        sa = A[rng.integers(0, len(A), size=(300, 880))]
        sb = Bx[rng.integers(0, len(Bx), size=(300, 880))]
        evA = {}
        for v in vals:
            ps, pc = comp[(name, v)]
            ev, bu, raw = P.safety_curve(ps[sa], pc[sa], ts_c[sa], tc_c[sa], MULT, GRID)
            gi = int(np.argmax(ev)); evA[v] = (float(ev[gi]), float(GRID[gi]))
        vstar = max(evA, key=lambda k: evA[k][0])
        picks.append(vstar)
        eb_ = {}
        for v in (DEPLOYED_CFG[name], vstar):
            ps, pc = comp[(name, v)]
            ev, _, _ = P.safety_curve(ps[sb], pc[sb], ts_c[sb], tc_c[sb], MULT,
                                      np.array([evA[v][1]]))
            eb_[v] = float(ev[0])
        gains.append(eb_[vstar] - eb_[DEPLOYED_CFG[name]])
        bb = -1
        for v in vals:
            ps, pc = comp[(name, v)]
            ev, _, _ = P.safety_curve(ps[sb], pc[sb], ts_c[sb], tc_c[sb], MULT, GRID)
            bb = max(bb, float(ev.max()))
        oracles.append(bb - eb_[DEPLOYED_CFG[name]])
    g = np.array(gains); o = np.array(oracles)
    from collections import Counter
    print(f"  {name:16s} chosen={dict(Counter(picks))}")
    print(f"  {'':16s} transferred gain {g.mean():+.5f} +- {g.std()/np.sqrt(NS):.5f} "
          f"(pos {int((g>0).sum())}/{NS}) | same-half oracle {o.mean():+.5f} "
          f"=> selection bias {o.mean()-g.mean():+.5f}")
