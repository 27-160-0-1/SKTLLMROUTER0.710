# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b05 step 12: put an externally trained gain head into the deployed pipeline
and score it honestly.

Sources: reports/lab/b05_ft_oof.npz (fine-tuned 12-layer encoder) and
reports/lab/b05_static_oof.npz (token-embedding-lookup model).  Both were
produced fold-pure on the same 10 folds as the b05base stage, plus a
train-only fit for dev.

  python b05-time-heavy_12_inject.py <npz> [<npz> ...]
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b05lib", HERE / "b05-time-heavy_lib.py")
lib = importlib.util.module_from_spec(spec); sys.modules["b05lib"] = lib
spec.loader.exec_module(lib)
from harness import DEPLOYED_EXP, DEPLOYED_CFG, TIERS, W  # noqa: E402
import bench2 as B  # noqa: E402

EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)
lab = lib.XLab()
cv, arr, parts = lib.build_stage(lab, EXP, tag="b05base")
POS = {int(v): k for k, v in enumerate(cv["idx"])}
D = lab.delta_targets
GP = dict(max_iter=EXP["gbm_iter"], learning_rate=EXP["gbm_lr"],
          max_leaf_nodes=EXP["gbm_leaves"], min_samples_leaf=EXP["gbm_min_leaf"],
          l2_regularization=EXP["gbm_l2"], early_stopping=True,
          validation_fraction=0.15, random_state=11)
OUT = Path("reports/lab/b05_inject.json")
ROWS = json.loads(OUT.read_text()) if OUT.exists() else []

# the baseline GBM gain head, in cv["idx"] order and dev order
BASE_CV = cv["gain"].copy()
BASE_DEV = arr["gain"].copy()


def to_cv_order(g_train_order):
    out = np.zeros((len(cv["idx"]), 2))
    for i in range(len(g_train_order)):
        out[POS[i]] = g_train_order[i]
    return out


def rescale(x, ref):
    """Match the per-column sd of the reference head so the gain magnitudes stay
    on the same scale as the pipeline expects (the allocator is invariant to a
    common factor but NOT to the ratio between the two gains)."""
    s = x.std(axis=0); s = np.where(s > 1e-12, s, 1.0)
    return (x - x.mean(axis=0)) / s * ref.std(axis=0) + ref.mean(axis=0)


def report(name, cvg, devg, cfg=None, secs=0.0):
    dg = lib.gain_axis(lab, cv["idx"], cvg[:, 0], cvg[:, 1])
    cfg = dict(DEPLOYED_CFG, **(cfg or {}))
    r = B.run(lab, dict(cv, gain=cvg), dict(arr, gain=devg), cfg, label=name, verbose=False)
    row = dict(name=name, ga=cfg["gain_alpha"], rb=cfg["rank_beta"], secs=round(secs, 1),
               **{k: round(v, 4) for k, v in dg.items()}, EV=round(r["EV"], 6),
               raw=round(sum(W[t] * r["det"][t]["raw"] for t in TIERS), 6),
               dev=round(r["dev"], 6), safety=[round(r["safety"][t], 3) for t in TIERS],
               ratio=[round(r["dev_tiers"][t]["ratio"], 3) for t in TIERS])
    ROWS.append(row); OUT.write_text(json.dumps(ROWS, indent=1), encoding="utf-8")
    print(f"{name:40s}ga={cfg['gain_alpha']:.2f} rb={cfg['rank_beta']:.2f} "
          f"d1={dg['corr1']:+.4f} d2={dg['corr2']:+.4f} A1={dg['auc1']:.4f} A2={dg['auc2']:.4f} "
          f"EV={row['EV']:.6f} raw={row['raw']:.6f} dev={row['dev']:.6f} sf={row['safety']} "
          f"r={row['ratio']}", flush=True)


CFGS = [None, dict(gain_alpha=1.0, rank_beta=0.0), dict(gain_alpha=0.85, rank_beta=0.0)]

if __name__ == "__main__":
    print("--- baseline (deployed GBM gain head) ---", flush=True)
    for c in CFGS:
        report("REF GBM gain", BASE_CV, BASE_DEV, c)
    for path in sys.argv[1:]:
        Z = np.load(path)
        names = [k for k in Z.files if not k.endswith("|dev")]
        for nm in names:
            if nm + "|dev" not in Z.files:
                print(f"skip {nm}: no dev block"); continue
            g = to_cv_order(Z[nm]); dgv = Z[nm + "|dev"]
            g = rescale(g, BASE_CV); dgv = rescale(dgv, BASE_DEV)
            for c in CFGS:
                report(f"{nm} (pure)", g, dgv, c)
            for w in (0.3, 0.5, 0.7):
                gm = (1 - w) * BASE_CV + w * g
                dm = (1 - w) * BASE_DEV + w * dgv
                for c in CFGS[:2]:
                    report(f"{nm} blend w={w:g}", gm, dm, c)
