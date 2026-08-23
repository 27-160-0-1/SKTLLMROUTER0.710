# Local held-out evaluation (this machine's venv). Constants come from env or E43 defaults.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "$root\src"
$env:PYTHONUTF8 = "1"
if (-not $env:ROUTER_FAM_W) { $env:ROUTER_FAM_W = "0.15" }
if (-not $env:ROUTER_CONF_SCALE) { $env:ROUTER_CONF_SCALE = "0.25" }
if (-not $env:ROUTER_BLEND_FAST) { $env:ROUTER_BLEND_FAST = "0.6" }
if (-not $env:ROUTER_BLEND_BALANCED) { $env:ROUTER_BLEND_BALANCED = "0.45" }
if (-not $env:ROUTER_BLEND_PREMIUM) { $env:ROUTER_BLEND_PREMIUM = "0.3" }
if (-not $env:ROUTER_RIDGE_ALPHA) { $env:ROUTER_RIDGE_ALPHA = "10" }
if (-not $env:ROUTER_GAIN_ALPHA) { $env:ROUTER_GAIN_ALPHA = "0.5" }
if (-not $env:ROUTER_RANK_BETA) { $env:ROUTER_RANK_BETA = "0.4" }
$py = "$root\.venv\Scripts\python.exe"
$out = if ($env:HOLDOUT_OUT) { $env:HOLDOUT_OUT } else { "$root\reports\holdout_local" }
New-Item -ItemType Directory -Force $out | Out-Null
$art = "$out\learned-router.v1.json"

Write-Host "[holdout] 1) linear head (Train only)"
& $py tools/train_learned_router_gpu.py `
  --input data/materialized/train/inputs.json --outcomes data/train/outcomes.json `
  --validation-input data/materialized/dev/inputs.json --validation-outcomes data/dev/outcomes.json `
  --artifact $art --report "$out\gpu-report.json" `
  --word-bins 8192 --char-bins 8192 --alphas $env:ROUTER_RIDGE_ALPHA --blend-weights 0.9 --context-limits 1000000
if ($LASTEXITCODE -ne 0) { throw "linear stage failed" }

Write-Host "[holdout] 2) augmentation (Train only)"
& $py tools/build_router_augmentation.py --artifact $art `
  --train-input data/materialized/train/inputs.json --train-outcomes data/train/outcomes.json `
  --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json
if ($LASTEXITCODE -ne 0) { throw "augmentation stage failed" }

Write-Host "[holdout] 3) meta GBM (Train only)"
& $py tools/build_meta_gbm.py --artifact $art `
  --train-input data/materialized/train/inputs.json --train-outcomes data/train/outcomes.json `
  --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json
if ($LASTEXITCODE -ne 0) { throw "meta stage failed" }

Write-Host "[holdout] 4) held-out Dev score (no lookup)"
& $py tools/holdout_eval.py --artifact $art --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json
