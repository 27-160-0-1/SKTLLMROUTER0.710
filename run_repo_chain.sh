#!/usr/bin/env bash
# The repository's own build chain (GPU cupy linear head), Train-only, Dev scored once.
# This is the chain that produced the published 0.705568 -- no CPU substitution anywhere.
#
#   $1 = baseline | replace | append | only
#        baseline -> the shipped 2-column prior, untouched (reproduces 0.705568)
#        replace  -> [A, C]     C = the real skt/A.X-3.1 34B column
#        append   -> [A, B, C]
#        only     -> [C]
#
# Constants are E55's shipped values; DEPLOY_TRAIN_ONLY semantics of tools/deploy_v2.ps1.
set -u
MODE="${1:-baseline}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY="${PY:-python}"
export PYTHONPATH="$ROOT/src"
export PYTHONUTF8=1
export OMP_NUM_THREADS=2          # b04: thread count alone moved the dev premium ratio 3.47->3.80
export ROUTER_FAM_W=0.15 ROUTER_CONF_SCALE=0.25
export ROUTER_BLEND_FAST=0.6 ROUTER_BLEND_BALANCED=0.45 ROUTER_BLEND_PREMIUM=0.3
export ROUTER_RIDGE_ALPHA=10 ROUTER_GAIN_ALPHA=0.5 ROUTER_RANK_BETA=0.4
export ROUTER_SAFETY_FAST=0.94 ROUTER_SAFETY_BALANCED=0.80 ROUTER_SAFETY_PREMIUM=0.73

# CHAIN_OUT lets a second configuration (e.g. ROUTER_META_SEEDS=1) run without
# clobbering the first one's artifact.
OUT="${CHAIN_OUT:-$ROOT/reports/repo_$MODE}"
mkdir -p "$OUT"
ART="$OUT/learned-router.v1.json"
REF="$ROOT/src/ossp_router/resources/learned-router.v1.json"
TRAIN_IN=data/materialized/train/inputs.json
TRAIN_OUT=data/train/outcomes.json
DEV_IN=data/materialized/dev/inputs.json
DEV_OUT=data/dev/outcomes.json

echo "=== [$MODE] 1/4 linear head on the GPU (Train only) ==="
"$PY" -X utf8 tools/train_learned_router_gpu.py \
  --input $TRAIN_IN --outcomes $TRAIN_OUT \
  --validation-input $DEV_IN --validation-outcomes $DEV_OUT \
  --artifact "$ART" --report "$OUT/gpu-report.json" \
  --word-bins 8192 --char-bins 8192 --alphas 10 --blend-weights 0.9 --context-limits 1000000 || exit 1
grep -o '"training_backend": *"[^"]*"' "$ART" | head -1
grep -o '"solver": *"[^"]*"' "$ART" | head -1

echo "=== [$MODE] 2/4 augmentation (Train only) ==="
"$PY" -X utf8 tools/build_router_augmentation.py --artifact "$ART" \
  --train-input $TRAIN_IN --train-outcomes $TRAIN_OUT \
  --dev-input $DEV_IN --dev-outcomes $DEV_OUT || exit 1

echo "=== [$MODE] 3/4 prior ==="
if [ "$MODE" = "baseline" ]; then
  "$PY" -X utf8 tools/inject_prior.py --artifact "$ART" --reference "$REF" || exit 1
else
  "$PY" -X utf8 tools/splice_prior_column.py --artifact "$ART" --reference "$REF" \
    --column-json colab-label/prior_column_c.json --mode "$MODE" || exit 1
  # EXTRA_COLUMN appends one more compiled column on top (E66: the reasoning-model length prior)
  if [ -n "${EXTRA_COLUMN:-}" ]; then
    "$PY" -X utf8 tools/splice_prior_column.py --artifact "$ART" --reference "$ART" \
      --column-json "$EXTRA_COLUMN" --mode append || exit 1
  fi
fi

echo "=== [$MODE] 4/4 meta GBM + held-out Dev ==="
"$PY" -X utf8 tools/build_meta_gbm.py --artifact "$ART" \
  --train-input $TRAIN_IN --train-outcomes $TRAIN_OUT \
  --dev-input $DEV_IN --dev-outcomes $DEV_OUT || exit 1
"$PY" -X utf8 tools/holdout_eval.py --artifact "$ART" --input $DEV_IN --outcomes $DEV_OUT
echo "=== [$MODE] DONE ==="
