# SPDX-License-Identifier: Apache-2.0
"""E68 diagnosis -- where does the "light 0 / k1 1" signal get lost?

E67's gap analysis: on dev's dmmath + cruxeval items the oracle routes to k1, the truth is
light 0.014-0.059 / k1 0.956-0.971, but the router predicts light 0.30-0.37 / k1 0.74-0.84.
Both shrink toward the middle, the predicted efficiency halves, and the upgrade loses the
ranking.

The router already carries E21's ordinal heads, i.e. P(score >= t) per model per threshold,
which are then averaged into E[score].  This measures three things on those items:

  1. Is the *raw* ordinal probability P(light >= 0.25) already sharp (near 0) there, with the
     sharpness lost only in the averaging?  -> then a decision feature, not a new head, fixes it.
  2. Or are the ordinal heads themselves soft?  -> then a new head targeting the joint event
     (light == 0 AND k1 == 1) is needed.
  3. How separable is that joint event from hash features, fold-pure (AUC)?  That is the
     ceiling for any such head.

Usage: PYTHONPATH=src python tools/e68_extreme_diag.py [--artifact A.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ossp_router import learned_router, similarity  # noqa: E402
from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=ROOT / "reports/e66b_append/learned-router.v1.json")
    a = ap.parse_args()

    policy = load_bundled_policy()
    unit = Decimal(policy.token_unit)
    eps, idx, split_of = [], {}, {}
    for split in ("train", "dev"):
        for e in load_input(ROOT / f"data/materialized/{split}/inputs.json").episodes:
            eps.append(e); split_of[e.episode_id] = split
        for o in load_outcomes(ROOT / f"data/{split}/outcomes.json").outcomes:
            idx[(o.episode_id, o.model_id)] = o
    n = len(eps)
    S = np.array([[float(idx[(e.episode_id, m)].score) for m in MODEL_IDS] for e in eps])
    is_dev = np.array([split_of[e.episode_id] == "dev" for e in eps])
    src = {r["episode_id"]: r["source"] for r in csv.DictReader((ROOT / "analysis/episodes.csv").open(encoding="utf-8"))}
    fam = np.array([src[e.episode_id] for e in eps])

    # the joint extreme event, and its base rate
    extreme = (S[:, 0] == 0.0) & (S[:, 2] == 1.0)
    print(f"extreme (light==0 & k1==1): {extreme.sum()}/{n} = {extreme.mean():.3f}   "
          f"dev {extreme[is_dev].sum()}/{is_dev.sum()}")
    print("  by family: " + ", ".join(f"{f} {int(extreme[fam == f].sum())}/{int((fam == f).sum())}"
                                      for f in sorted(set(fam), key=lambda f: -extreme[fam == f].sum())))

    # ---- 1/2: what the shipped meta heads say on dev extremes ----
    raw = json.loads(a.artifact.read_text(encoding="utf-8")); raw.pop("public_lookup", None)
    art = learned_router.parse_artifact(raw, base_path=a.artifact.parent)
    meta = art.meta_gbm
    T = len(meta.ordinal_thresholds)
    dev_i = np.where(is_dev)[0]
    rows = []
    for i in dev_i:
        e = eps[i]; text = episode_text(e); family = similarity.classify_family(text)
        _s, _c, learned_row, legacy_row, raw_dense = learned_router._predict_with_components(e, art, text)
        aug = art.augmentation
        q = similarity.tfidf_vector(text, aug.idf)
        knn_row, top = aug.index.predict(q)
        feats = list(raw_dense)
        oh = [0.0] * len(similarity.FAMILY_NAMES); oh[similarity.FAMILY_NAMES.index(family)] = 1.0
        feats.extend(oh); feats.extend(legacy_row); feats.extend(learned_row)
        feats.extend(knn_row if knn_row else meta.knn_fallback_row); feats.append(top if knn_row else 0.0)
        feats.extend(learned_router.prior_features(text, family, art.prior_lookup))
        p = np.zeros((3, T))
        for m in range(3):
            for t in range(T):
                h = m * T + t
                r = similarity.evaluate_trees(meta.ordinal_baselines[h], meta.ordinal_trees[h], feats)
                p[m, t] = 1 / (1 + np.exp(-max(-50, min(50, r))))
        direct = [similarity.evaluate_trees(meta.baselines[k], meta.trees[k], feats) for k in range(3)]
        rows.append((i, p, direct))
    P = np.array([r[1] for r in rows]); D = np.array([r[2] for r in rows]); di = np.array([r[0] for r in rows])
    ex = extreme[di]
    print("\n=== dev: shipped heads on extreme vs non-extreme items ===")
    print(f"{'quantity':<34}{'extreme':>10}{'others':>10}")
    for label, v in (("P(light>=0.25)  [ordinal raw]", P[:, 0, 0]),
                     ("P(light>=1.0)", P[:, 0, 3]),
                     ("E[light] = mean of ordinal", P[:, 0].mean(1)),
                     ("direct regression light", D[:, 0]),
                     ("P(k1>=1.0)  [ordinal raw]", P[:, 2, 3]),
                     ("E[k1] = mean of ordinal", P[:, 2].mean(1)),
                     ("direct regression k1", D[:, 2])):
        print(f"{label:<34}{v[ex].mean():>10.3f}{v[~ex].mean():>10.3f}")
    # separability of the joint event from what the router already has
    from sklearn.metrics import roc_auc_score
    joint = (1 - P[:, 0, 0]) * P[:, 2, 3]
    print(f"\nAUC of existing signals for the extreme event (dev):")
    print(f"  1-P(light>=.25)            {roc_auc_score(ex, 1 - P[:, 0, 0]):.3f}")
    print(f"  P(k1>=1)                   {roc_auc_score(ex, P[:, 2, 3]):.3f}")
    print(f"  product (1-P_l25)*P_k1     {roc_auc_score(ex, joint):.3f}")
    print(f"  E[k1]-E[light] (mean ord.) {roc_auc_score(ex, P[:, 2].mean(1) - P[:, 0].mean(1)):.3f}")
    print(f"  direct k1 - direct light   {roc_auc_score(ex, D[:, 2] - D[:, 0]):.3f}")

    # ---- 3: ceiling -- a dedicated fold-pure classifier on the same meta features ----
    # reuse the feature vectors we just built for dev; build train ones the same way
    def feats_for(i):
        e = eps[i]; text = episode_text(e); family = similarity.classify_family(text)
        _s, _c, learned_row, legacy_row, raw_dense = learned_router._predict_with_components(e, art, text)
        q = similarity.tfidf_vector(text, art.augmentation.idf); knn_row, top = art.augmentation.index.predict(q)
        f = list(raw_dense); oh = [0.0] * 9; oh[similarity.FAMILY_NAMES.index(family)] = 1.0
        f.extend(oh); f.extend(legacy_row); f.extend(learned_row)
        f.extend(knn_row if knn_row else meta.knn_fallback_row); f.append(top if knn_row else 0.0)
        f.extend(learned_router.prior_features(text, family, art.prior_lookup)); return f
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.array([feats_for(i) for i in range(n)]); y = extreme.astype(int)
    # NOTE: learned_row / knn on train rows are in-sample here (not OOF) -- this is an upper
    # bound on separability, not a deployable number.
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                                         min_samples_leaf=30, l2_regularization=3.0, random_state=11)
    clf.fit(X[~is_dev], y[~is_dev])
    pd_ = clf.predict_proba(X[is_dev])[:, 1]
    print(f"\ndedicated extreme-event classifier, train->dev AUC: {roc_auc_score(y[is_dev], pd_):.3f}   "
          f"(upper bound: train-side features are in-sample)")
    for thr in (0.5, 0.7, 0.85):
        sel = pd_ >= thr
        if sel.sum():
            print(f"  P>={thr}: {sel.sum():>3} flagged, precision {y[is_dev][sel].mean():.3f}, "
                  f"recall {(y[is_dev][sel].sum() / max(y[is_dev].sum(), 1)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
