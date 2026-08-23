# SPDX-License-Identifier: Apache-2.0
"""a11 step 9 - deterministic score-vs-realised-budget curve, per kappa.

No bootstrap, no risk model: for each relative-price vector kappa we sweep the
safety scalar, record (realised true ratio, realised true mean score) on dev and
compare the curves at MATCHED realised budget.  If the kappa curve sits above
kappa=1 at every budget, the reweighting improves the allocation itself.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
from labdata import load_split, TIERS, TIER_MULT
import importlib.util
_spec = importlib.util.spec_from_file_location("a11path", HERE / "a11-postproc-router_5_path.py")
a11path = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(a11path)

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv)


def curve(tier, kappa, ngrid=1200):
    """(realised ratio, realised score) along the whole Lagrangian path."""
    ps = P[f"score_{tier}"]
    pc = P[f"cost_{tier}"] * np.asarray(kappa)[None, :]
    pa = a11path.path_arrays(ps, pc, dv.score, dv.cost)
    v = pa["valid"].ravel()
    o = np.argsort(-pa["eff"].ravel()[v], kind="stable")
    dct = np.concatenate([[0.0], np.cumsum(pa["dc_t"].ravel()[v][o])])
    dst = np.concatenate([[0.0], np.cumsum(pa["ds_t"].ravel()[v][o])])
    ratio = (dv.cost[:, 0].sum() + dct) / dv.cost[:, 0].sum()
    score = (dv.score[:, 0].sum() + dst) / n
    return ratio, score


def score_at(ratio, score, target):
    """best realised score among path points whose realised ratio <= target."""
    ok = ratio <= target + 1e-12
    return score[ok].max() if ok.any() else np.nan


KAPPAS = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.15), (1.0, 1.0, 1.24), (1.0, 1.0, 1.4),
          (1.0, 1.0, 1.6), (1.0, 1.0, 2.0), (1.0, 0.93, 1.24), (1.0, 0.93, 1.6)]
print("=== realised score at a MATCHED realised budget ratio (dev, deterministic) ===")
for tier in TIERS:
    mult = TIER_MULT[tier]
    targets = np.array([0.80, 0.85, 0.90, 0.95, 1.00]) * mult
    print(f"-- {tier} (cap {mult}); columns = realised ratio target")
    head = f"   {'kappa':18s}" + "".join(f"{x:9.3f}" for x in targets)
    print(head)
    ref = None
    for kap in KAPPAS:
        r, s = curve(tier, kap)
        row = [score_at(r, s, t) for t in targets]
        if ref is None:
            ref = row
        d = "" if kap == KAPPAS[0] else "   delta " + " ".join(f"{a-b:+.4f}" for a, b in zip(row, ref))
        print(f"   {str(kap):18s}" + "".join(f"{x:9.4f}" for x in row))
        if d:
            print(f"   {'':18s}{d}")
    print()

print("=== the same, expressed as the safety scalar needed to hit the target ===")
for tier in ("premium",):
    mult = TIER_MULT[tier]
    for kap in [(1.0, 1.0, 1.0), (1.0, 0.93, 1.24), (1.0, 1.0, 1.6)]:
        r, s = curve(tier, kap)
        # find the path index for each target and report the predicted-ratio there
        ps = P[f"score_{tier}"]; pc = P[f"cost_{tier}"] * np.asarray(kap)[None, :]
        pa = a11path.path_arrays(ps, pc, dv.score, dv.cost)
        v = pa["valid"].ravel(); o = np.argsort(-pa["eff"].ravel()[v], kind="stable")
        dcp = np.concatenate([[0.0], np.cumsum(pa["dc_p"].ravel()[v][o])])
        pred_ratio = (pc[:, 0].sum() + dcp) / pc[:, 0].sum()
        for tgt in (0.85 * mult, 0.90 * mult, 0.95 * mult):
            ok = np.flatnonzero(r <= tgt + 1e-12)
            k = ok[int(np.argmax(s[ok]))]
            print(f"   {tier} kappa={kap} target ratio<={tgt:.2f}: score={s[k]:.4f} "
                  f"realised={r[k]:.4f} pred_ratio={pred_ratio[k]:.4f} "
                  f"=> safety s={pred_ratio[k]/mult:.3f}")
        print()
