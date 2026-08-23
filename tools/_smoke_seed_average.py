# SPDX-License-Identifier: Apache-2.0
"""Prove the seed-average merge is exact, not approximate.

`_fit_export` claims that concatenating N models' trees with leaves scaled by 1/N, and averaging
the baselines, reproduces the mean of the N models' predictions under the runtime's
`similarity.evaluate_trees`.  This checks that against sklearn directly, and also checks that
META_SEEDS=1 is byte-identical to the old single-fit path.
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

N = 5
os.environ["ROUTER_META_SEEDS"] = str(N)
import build_meta_gbm as B  # noqa: E402
from ossp_router.similarity import evaluate_trees  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 12))
y = X[:, 0] * 2 + np.sin(X[:, 3]) + rng.normal(scale=0.3, size=400)
Xt = rng.normal(size=(40, 12))

# ---- regressor ----
baseline, trees = B._fit_export(
    lambda seed: HistGradientBoostingRegressor(**{**B.GBM_PARAMS, "random_state": seed}), X, y)
merged = np.array([evaluate_trees(baseline, trees, row) for row in Xt])

direct = []
for i in range(N):
    m = HistGradientBoostingRegressor(**{**B.GBM_PARAMS, "random_state": 11 + 1000 * i}).fit(X, y)
    direct.append(m.predict(Xt))
expected = np.mean(direct, axis=0)
err = float(np.max(np.abs(merged - expected)))
print(f"[smoke] regressor: {len(trees)} merged trees, max |merged - mean(sklearn)| = {err:.3e}")
assert err < 1e-9, f"seed average is not exact: {err}"

spread = float(np.mean(np.std(direct, axis=0)))
print(f"[smoke]   per-seed prediction sd {spread:.4f} -> averaging removes ~{spread*(1-1/np.sqrt(N)):.4f} of it")

# ---- classifier (ordinal heads use raw scores, sigmoid is applied by the runtime) ----
yc = (y > np.median(y)).astype(int)
cb, ct = B._fit_export(
    lambda seed: HistGradientBoostingClassifier(**{**B.GBM_PARAMS, "random_state": seed}), X, yc)
cm = np.array([evaluate_trees(cb, ct, row) for row in Xt])
cd = np.mean([
    HistGradientBoostingClassifier(**{**B.GBM_PARAMS, "random_state": 11 + 1000 * i})
    .fit(X, yc).decision_function(Xt) for i in range(N)], axis=0)
cerr = float(np.max(np.abs(cm - cd)))
print(f"[smoke] classifier: {len(ct)} merged trees, max |merged - mean(sklearn)| = {cerr:.3e}")
assert cerr < 1e-9, f"classifier seed average is not exact: {cerr}"

# ---- META_SEEDS=1 must reproduce the old path exactly ----
B.META_SEEDS = 1
b1, t1 = B._fit_export(
    lambda seed: HistGradientBoostingRegressor(**{**B.GBM_PARAMS, "random_state": seed}), X, y)
old = HistGradientBoostingRegressor(**B.GBM_PARAMS).fit(X, y)
ob, ot = B._export_model(old)
same = (b1 == ob and t1 == ot)
print(f"[smoke] META_SEEDS=1 identical to the previous single fit: {same}")
assert same, "META_SEEDS=1 changed the old behaviour"
print("[smoke] PASS")
