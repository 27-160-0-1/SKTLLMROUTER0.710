# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Train and embed the stacked gradient-boosted meta heads.

Meta-features per episode (58): raw dense 30 + family one-hot 9 + legacy
(3 scores, 3 log costs) + learned-linear (3+3, out-of-fold refit on Train)
+ public-train kNN (3+3 + top-1 similarity).  Six HistGradientBoosting heads
are trained on public Train against true scores / log costs, exported as
plain node arrays, and blended 30% over the augmented prediction row.

Run AFTER build_router_augmentation.py and BEFORE build_public_lookup.py.
Requires scikit-learn/scipy at build time only; the runtime evaluates the
exported trees with the standard library.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ossp_router import learned_router, legacy_hash_regex, similarity
from ossp_router.heuristic import episode_text
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

import os as _os
BLEND_WEIGHTS = {"fast": float(_os.environ.get("ROUTER_BLEND_FAST", 0.6)),
                 "balanced": float(_os.environ.get("ROUTER_BLEND_BALANCED", 0.3)),
                 "premium": float(_os.environ.get("ROUTER_BLEND_PREMIUM", 0.45))}   # E43 override hooks
GBM_PARAMS = dict(
    max_iter=300,
    learning_rate=0.06,
    max_leaf_nodes=15,
    min_samples_leaf=30,
    l2_regularization=3.0,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=11,
)
OOF_FOLDS = 5
OOF_SEED = 0
LEGACY_OOF_SEED = 7
LEGACY_HASH_BINS = 256
LEGACY_ALPHA = 100.0
LEGACY_OOF_META = _os.environ.get('ROUTER_LEGACY_OOF_META', '1') == '1'
# E52 adopted seed-averaged meta heads but the averaging was never implemented here.
# ROUTER_META_SEEDS=1 restores the previous single-fit behaviour.
META_SEEDS = max(1, int(_os.environ.get('ROUTER_META_SEEDS', '5')))
RIDGE_ALPHA = float(_os.environ.get("ROUTER_RIDGE_ALPHA", 30.0))
SAFETY_RATIOS = {"fast": float(_os.environ.get("ROUTER_SAFETY_FAST", 0.98)),
                 "balanced": float(_os.environ.get("ROUTER_SAFETY_BALANCED", 0.89)),
                 "premium": float(_os.environ.get("ROUTER_SAFETY_PREMIUM", 0.88))}
GAIN_ALPHA = float(_os.environ.get("ROUTER_GAIN_ALPHA", 0.5))
ORDINAL_THRESHOLDS = (0.25, 0.5, 0.75, 1.0)
RANK_BETA = float(_os.environ.get("ROUTER_RANK_BETA", 0.25))
RANK_LUT_NODES = 65
RANK_FLOOR_QUANTILE = 0.05


def _targets(inputs, outcomes, policy):
    from decimal import Decimal

    index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    rows = []
    for episode in inputs.episodes:
        values = [index[(episode.episode_id, m)] for m in MODEL_IDS]
        costs = []
        for value in values:
            rates = policy.models[value.model_id]
            unit = Decimal(policy.token_unit)
            costs.append(float(
                rates.fixed_cost
                + Decimal(value.input_tokens) * rates.input_token_rate / unit
                + Decimal(value.output_tokens) * rates.output_token_rate / unit
            ))
        rows.append(
            [float(v.score) for v in values] + [math.log(c) for c in costs]
        )
    return np.asarray(rows, dtype=np.float64)


def _oof_linear(train_inputs, artifact, targets):
    from scipy import sparse
    from sklearn.linear_model import Ridge

    rows, cols, vals = [], [], []
    for row_index, episode in enumerate(train_inputs.episodes):
        items = learned_router.feature_items(
            episode,
            word_hash_bins=artifact.word_hash_bins,
            char_hash_bins=artifact.char_hash_bins,
            dense_mean=artifact.dense_mean,
            dense_scale=artifact.dense_scale,
        )
        for c, v in items.items():
            rows.append(row_index); cols.append(c); vals.append(v)
    dim = (
        len(learned_router.DENSE_FEATURE_NAMES)
        + artifact.word_hash_bins + artifact.char_hash_bins
    )
    matrix = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(len(train_inputs.episodes), dim)
    )
    oof = np.zeros((matrix.shape[0], targets.shape[1]))
    fold_of = np.random.default_rng(OOF_SEED).integers(0, OOF_FOLDS, size=matrix.shape[0])
    for fold in range(OOF_FOLDS):
        hold = fold_of == fold
        model = Ridge(alpha=RIDGE_ALPHA, solver="sparse_cg")
        model.fit(matrix[~hold], targets[~hold])
        oof[hold] = model.predict(matrix[hold])
    return oof


def _oof_legacy(train_inputs, targets):
    """Out-of-fold refit of the official 256-bin hash-regex head.

    The shipped legacy artifact was fitted on all 1,760 public Train episodes,
    so using its predictions as a meta feature makes them in-sample for every
    training row (score correlation 0.60) while they are out-of-sample at
    prediction time (0.39).  The tree ensemble then learns to over-trust them.
    Refitting inside folds — exactly what the ridge block already does — removes
    that mismatch.  Reproduces baselines/train_hash_regex._fit_ridge at its
    published alpha, verified bit-for-bit against the shipped artifact.
    """
    from ossp_router import legacy_hash_regex

    raw = np.asarray(
        [legacy_hash_regex.raw_feature_vector(e, LEGACY_HASH_BINS)
         for e in train_inputs.episodes],
        dtype=np.float64,
    )

    def fit(rows):
        matrix = raw[rows]
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = (matrix - mean) / scale
        intercept = targets[rows].mean(axis=0)
        centered = targets[rows] - intercept
        count, columns = standardized.shape
        if count <= columns:
            coefficients = standardized.T @ np.linalg.solve(
                standardized @ standardized.T + LEGACY_ALPHA * np.eye(count), centered)
        else:
            coefficients = np.linalg.solve(
                standardized.T @ standardized + LEGACY_ALPHA * np.eye(columns),
                standardized.T @ centered)
        return mean, scale, intercept, coefficients

    def predict(head, rows):
        mean, scale, intercept, coefficients = head
        row = (raw[rows] - mean) / scale @ coefficients + intercept
        # the runtime clamps scores and enforces cost monotonicity
        row[:, :len(MODEL_IDS)] = np.clip(row[:, :len(MODEL_IDS)], 0.0, 1.0)
        costs = np.exp(np.clip(row[:, len(MODEL_IDS):], -50.0, 50.0))
        costs[:, 1] = np.maximum(costs[:, 1], costs[:, 0] * (1 + 1e-12))
        costs[:, 2] = np.maximum(costs[:, 2], costs[:, 1] * (1 + 1e-12))
        row[:, len(MODEL_IDS):] = np.log(costs)
        return row

    out = np.zeros_like(targets)
    fold_of = np.random.default_rng(LEGACY_OOF_SEED).integers(
        0, OOF_FOLDS, size=raw.shape[0])
    for fold in range(OOF_FOLDS):
        hold = fold_of == fold
        out[hold] = predict(fit(np.where(~hold)[0]), np.where(hold)[0])
    return out


def _loo_knn(train_inputs, augmentation, targets):
    texts = [episode_text(e) for e in train_inputs.episodes]
    global_mean = targets.mean(axis=0)
    rows = []
    for i, text in enumerate(texts):
        query = similarity.tfidf_vector(text, augmentation.idf)
        scores = {}
        for gram, value in query.items():
            for document, stored in augmentation.index.postings.get(gram, ()):
                if document == i:
                    continue
                scores[document] = scores.get(document, 0.0) + value * stored
        if not scores:
            rows.append(list(global_mean) + [0.0])
            continue
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:similarity.NEIGHBORS]
        total = sum(sim for _d, sim in ranked)
        blended = np.zeros(targets.shape[1])
        for document, sim in ranked:
            blended += (sim / total) * targets[document]
        rows.append(list(blended) + [ranked[0][1]])
    return np.asarray(rows), global_mean


def _export_model(model):
    trees = []
    for predictors in model._predictors:
        nodes = predictors[0].nodes
        trees.append([
            [
                int(node["is_leaf"]),
                float(node["value"]),
                int(node["feature_idx"]),
                float(node["num_threshold"]),
                int(node["left"]),
                int(node["right"]),
            ]
            for node in nodes
        ])
    return float(model._baseline_prediction.ravel()[0]), trees


def _fit_export(factory, X, y, seeds=None, **fit_kw):
    """Fit `seeds` models that differ only in random_state and export their average.

    A HistGradientBoosting prediction is `baseline + sum(leaf values hit)`, so the mean of N
    models is exactly `mean(baseline) + sum over all N models' trees of (leaf value / N)`.
    Concatenating the trees with their leaves scaled by 1/N therefore reproduces the average
    with no change to the runtime evaluator -- it just walks more trees.
    """

    n = META_SEEDS if seeds is None else seeds
    baselines, merged = [], []
    for index in range(n):
        model = factory(11 + 1000 * index)
        model.fit(X, y, **fit_kw)
        baseline, trees = _export_model(model)
        baselines.append(baseline)
        if n == 1:
            merged.extend(trees)
            continue
        for tree in trees:
            merged.append([
                [node[0], node[1] / n if node[0] else node[1], node[2], node[3], node[4], node[5]]
                for node in tree
            ])
    return sum(baselines) / len(baselines), merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-outcomes", type=Path, required=True)
    parser.add_argument("--dev-input", type=Path, required=True)
    parser.add_argument("--dev-outcomes", type=Path, required=True)
    args = parser.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor

    policy = load_bundled_policy()
    train_inputs = load_input(args.train_input)
    train_outcomes = load_outcomes(args.train_outcomes)
    artifact = learned_router.load_artifact(args.artifact)
    if artifact.augmentation is None:
        raise SystemExit("augmentation 블록이 먼저 필요합니다 (build_router_augmentation.py)")
    augmentation = artifact.augmentation

    targets = _targets(train_inputs, train_outcomes, policy)
    print("[meta] building train meta-features", flush=True)
    train_texts = [episode_text(e) for e in train_inputs.episodes]
    dense_fam_legacy_linear = []
    for episode, text in zip(train_inputs.episodes, train_texts):
        _s, _c, learned_row, legacy_row, raw_dense = (
            learned_router._predict_with_components(episode, artifact, text)
        )
        onehot = [0.0] * len(similarity.FAMILY_NAMES)
        onehot[similarity.FAMILY_NAMES.index(similarity.classify_family(text))] = 1.0
        dense_fam_legacy_linear.append(
            (list(raw_dense), onehot, list(legacy_row), list(learned_row))
        )
    print("[meta] OOF ridge", flush=True)
    oof = _oof_linear(train_inputs, artifact, targets)
    print("[meta] LOO kNN", flush=True)
    knn_rows, global_mean = _loo_knn(train_inputs, augmentation, targets)
    legacy_meta = None
    if LEGACY_OOF_META:
        print("[meta] OOF legacy head", flush=True)
        legacy_meta = _oof_legacy(train_inputs, targets)

    prior_rows = [
        list(learned_router.prior_features(
            text, similarity.classify_family(text), artifact.prior_lookup))
        for text in train_texts
    ]
    X_train = np.asarray([
        dense + fam + (list(legacy_meta[i]) if legacy_meta is not None else legacy)
        + list(oof[i]) + list(knn_rows[i]) + prior_rows[i]
        for i, (dense, fam, legacy, _linear) in enumerate(dense_fam_legacy_linear)
    ])
    if artifact.prior_lookup is not None:
        for index, column in enumerate(artifact.prior_lookup.columns):
            offset = index * learned_router.PRIOR_COLUMN_FEATURES
            hit = float(np.mean([r[offset] for r in prior_rows]))
            print(f"[meta] prior column '{column.tag}' present for {hit:.1%} of training rows",
                  flush=True)
    print(f"[meta] X_train {X_train.shape}", flush=True)

    baselines, all_trees = [], []
    for k in range(targets.shape[1]):
        baseline, trees = _fit_export(
            lambda seed: HistGradientBoostingRegressor(**{**GBM_PARAMS, "random_state": seed}),
            X_train, targets[:, k])
        baselines.append(baseline)
        all_trees.append(trees)
        print(f"[meta] head {k}: {len(trees)} trees", flush=True)
    delta_targets = np.column_stack([
        targets[:, 1] - targets[:, 0],
        targets[:, 2] - targets[:, 1],
    ])
    delta_baselines, delta_trees = [], []
    for k in range(2):
        baseline, trees = _fit_export(
            lambda seed: HistGradientBoostingRegressor(**{**GBM_PARAMS, "random_state": seed}),
            X_train, delta_targets[:, k])
        delta_baselines.append(baseline)
        delta_trees.append(trees)
        print(f"[meta] gain head {k}: {len(trees)} trees", flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier

    true_scores = targets[:, : len(MODEL_IDS)]
    ordinal_baselines, ordinal_trees = [], []
    for model_index in range(len(MODEL_IDS)):
        for threshold in ORDINAL_THRESHOLDS:
            y = (true_scores[:, model_index] >= threshold).astype(int)
            if y.min() == y.max():
                probability = min(max(float(y.min()), 1e-6), 1 - 1e-6)
                ordinal_baselines.append(math.log(probability / (1 - probability)))
                ordinal_trees.append([])
                print(f"[meta] ordinal m{model_index} t{threshold}: constant", flush=True)
                continue
            baseline, trees = _fit_export(
                lambda seed: HistGradientBoostingClassifier(**{**GBM_PARAMS, "random_state": seed}),
                X_train, y)
            ordinal_baselines.append(baseline)
            ordinal_trees.append(trees)
            print(f"[meta] ordinal m{model_index} t{threshold}: {len(trees)} trees", flush=True)

    # E27: rank-transformed efficiency heads (allocator-aligned targets)
    from scipy.stats import rankdata

    true_costs = np.exp(targets[:, len(MODEL_IDS):])
    rank_grid = np.linspace(0.0, 1.0, RANK_LUT_NODES)
    rank_baselines, rank_trees, rank_luts, rank_floors = [], [], [], []
    for g, (low, high) in enumerate([(0, 1), (1, 2)]):
        score_delta = true_scores[:, high] - true_scores[:, low]
        cost_delta = true_costs[:, high] - true_costs[:, low]
        floor = max(float(np.quantile(cost_delta, RANK_FLOOR_QUANTILE)), 1e-9)
        efficiency = score_delta / np.maximum(cost_delta, floor)
        ranks = rankdata(efficiency, method="average") / max(len(efficiency) - 1, 1)
        baseline, trees = _fit_export(
            lambda seed: HistGradientBoostingRegressor(**{**GBM_PARAMS, "random_state": seed}),
            X_train, ranks)
        rank_baselines.append(baseline)
        rank_trees.append(trees)
        rank_luts.append([float(v) for v in np.quantile(efficiency, rank_grid)])
        rank_floors.append(floor)
        print(f"[meta] rank head {g}: {len(trees)} trees, floor {floor:.6f}", flush=True)

    raw = json.loads(args.artifact.read_text(encoding="utf-8"))
    raw["meta_gbm"] = {
        "blend_weight": dict(BLEND_WEIGHTS),
        "feature_layout": (
            "raw_dense30 + family_onehot9 + legacy(score3,logcost3) "
            "+ linear(score3,logcost3) + knn(score3,logcost3,top1)"
            + (" + prior7" if artifact.prior_lookup is not None else "")
        ),
        "baselines": baselines,
        "trees": all_trees,
        "gain_alpha": GAIN_ALPHA,
        "delta_baselines": delta_baselines,
        "delta_trees": delta_trees,
        "ordinal_thresholds": list(ORDINAL_THRESHOLDS),
        "ordinal_baselines": ordinal_baselines,
        "ordinal_trees": ordinal_trees,
        "rank_beta": RANK_BETA,
        "rank_baselines": rank_baselines,
        "rank_trees": rank_trees,
        "rank_luts": rank_luts,
        "rank_floors": rank_floors,
        "knn_fallback_row": [float(v) for v in global_mean],
    }
    raw["tier_safety_ratios"] = dict(SAFETY_RATIOS)
    raw.pop("public_lookup", None)
    args.artifact.write_text(
        json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    # dev verification through the actual runtime path
    print("[meta] dev verification", flush=True)
    from ossp_router.protocol import Decision, Submission
    from ossp_router.scoring import score_submissions

    dev_inputs = load_input(args.dev_input)
    dev_outcomes = load_outcomes(args.dev_outcomes)
    reloaded = learned_router.parse_artifact(raw)
    submissions = []
    for tier in TIERS:
        predictions = [
            learned_router.predict_episode_augmented(e, reloaded, tier)
            for e in dev_inputs.episodes
        ]
        selected, _ratio = learned_router.select_models(
            [p[0] for p in predictions],
            [p[1] for p in predictions],
            budget_multiplier=float(policy.tiers[tier].budget_multiplier),
            safety_ratio=SAFETY_RATIOS[tier],
        )
        submissions.append(Submission(
            schema_version=dev_inputs.schema_version,
            challenge_id=dev_inputs.challenge_id,
            policy_id=policy.policy_id,
            split=dev_inputs.split,
            tier=tier,
            decisions=tuple(
                Decision(e.episode_id, m)
                for e, m in zip(dev_inputs.episodes, selected)
            ),
        ))
    report = score_submissions(dev_inputs, dev_outcomes, submissions, policy)
    print(f"dev final={report['final_score']}")
    for tier in TIERS:
        t = report["tiers"][tier]
        print(f"  {tier}: score={t['tier_score']} ratio={t['budget_ratio']} passed={t['budget_passed']}")
    print("OK: meta_gbm 블록과 안전계수를 기록했습니다. build_public_lookup.py를 다시 실행하십시오.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
