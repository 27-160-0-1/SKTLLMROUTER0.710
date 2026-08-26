#!/usr/bin/env bash
# Held-out Dev score with the new 34B mid column spliced in.
#   $1 = append | replace | only   (see tools/splice_prior_column.py)
# Baseline from the same chain (shipped 2-column prior): 0.702727
set -u
MODE="${1:-append}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# 원래는 개발 기계의 venv 절대경로가 박혀 있었고 사용자 이름을 노출했다.
PY="${ROUTER_PY:-python3}"
export PYTHONPATH="$ROOT/src"
export PYTHONUTF8=1
export OMP_NUM_THREADS=2
export ROUTER_FAM_W=0.15 ROUTER_CONF_SCALE=0.25
export ROUTER_BLEND_FAST=0.6 ROUTER_BLEND_BALANCED=0.45 ROUTER_BLEND_PREMIUM=0.3
export ROUTER_RIDGE_ALPHA=10 ROUTER_GAIN_ALPHA=0.5 ROUTER_RANK_BETA=0.4
export ROUTER_SAFETY_FAST=0.94 ROUTER_SAFETY_BALANCED=0.80 ROUTER_SAFETY_PREMIUM=0.73

OUT="$ROOT/reports/e59_$MODE"
mkdir -p "$OUT"
ART="$OUT/learned-router.v1.json"
REF="$ROOT/src/ossp_router/resources/learned-router.v1.json"
TRAIN_IN=data/materialized/train/inputs.json
TRAIN_OUT=data/train/outcomes.json

echo "=== [$MODE] 1/5 linear head (Train only, CPU lsmr) ==="
"$PY" -X utf8 tools/cpu_shim_train.py \
  --input $TRAIN_IN --outcomes $TRAIN_OUT \
  --validation-input data/materialized/dev/inputs.json --validation-outcomes data/dev/outcomes.json \
  --artifact "$ART" --report "$OUT/gpu-report.json" \
  --word-bins 8192 --char-bins 8192 --alphas 10 --blend-weights 0.9 --context-limits 1000000 || exit 1

echo "=== [$MODE] 2/5 augmentation (Train only) ==="
"$PY" -X utf8 tools/build_router_augmentation.py --artifact "$ART" \
  --train-input $TRAIN_IN --train-outcomes $TRAIN_OUT \
  --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json || exit 1

echo "=== [$MODE] 3/5 prior: splice the 34B mid column ==="
"$PY" -X utf8 tools/splice_prior_column.py --artifact "$ART" --reference "$REF" \
  --labels colab-label/out/labels_mid_pool.jsonl colab-label/out/labels_gate.jsonl \
  --items colab-label/bundle/all.jsonl colab-label/bundle/ext.jsonl \
  --mode "$MODE" || exit 1

echo "=== [$MODE] 4/5 meta GBM (Train only) ==="
"$PY" -X utf8 tools/build_meta_gbm.py --artifact "$ART" \
  --train-input $TRAIN_IN --train-outcomes $TRAIN_OUT \
  --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json || exit 1

echo "=== [$MODE] 5/5 held-out Dev score ==="
"$PY" -X utf8 tools/holdout_eval.py --artifact "$ART" \
  --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json
echo "=== [$MODE] DONE ==="
