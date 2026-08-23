# Rebuild the deployment artifacts with the v2 chain (E43 constants + legacy-OOF meta
# features + seed-averaged meta heads + the offline difficulty prior lookup).
# Previous artifacts are kept as *.e43.bak.
$ErrorActionPreference = "Continue"   # native stderr (cupy warnings) must not abort; every stage checks $LASTEXITCODE
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "$root\src"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "2"          # b04: thread count alone moves the dev premium ratio 3.47->3.80
$env:ROUTER_FAM_W = "0.15"
$env:ROUTER_CONF_SCALE = "0.25"
$env:ROUTER_BLEND_FAST = "0.6"
$env:ROUTER_BLEND_BALANCED = "0.45"
$env:ROUTER_BLEND_PREMIUM = "0.3"
$env:ROUTER_RIDGE_ALPHA = "10"
$env:ROUTER_GAIN_ALPHA = "0.5"
$env:ROUTER_RANK_BETA = "0.4"
if (-not $env:ROUTER_SAFETY_FAST) { $env:ROUTER_SAFETY_FAST = "0.92" }
if (-not $env:ROUTER_SAFETY_BALANCED) { $env:ROUTER_SAFETY_BALANCED = "0.815" }
if (-not $env:ROUTER_SAFETY_PREMIUM) { $env:ROUTER_SAFETY_PREMIUM = "0.745" }
$py = "$root\.venv\Scripts\python.exe"
$train = if ($env:DEPLOY_TRAIN_ONLY) { "data/materialized/train/inputs.json" } else { "data/combined/inputs.json" }
$trainOut = if ($env:DEPLOY_TRAIN_ONLY) { "data/train/outcomes.json" } else { "data/combined/outcomes.json" }
$res = if ($env:DEPLOY_OUT) { $env:DEPLOY_OUT } else { "$root\src\ossp_router\resources" }
New-Item -ItemType Directory -Force $res | Out-Null
$art = "$res\learned-router.v1.json"

Write-Host "[v2] 1) linear head"
& $py tools/train_learned_router_gpu.py `
  --input $train --outcomes $trainOut `
  --validation-input data/materialized/dev/inputs.json --validation-outcomes data/dev/outcomes.json `
  --artifact $art --report reports/learned-router-gpu-report.v1.json `
  --word-bins 8192 --char-bins 8192 --alphas 10 --blend-weights 0.9 --context-limits 1000000
if ($LASTEXITCODE -ne 0) { throw "linear stage failed" }

Write-Host "[v2] 2) augmentation"
& $py tools/build_router_augmentation.py --artifact $art `
  --train-input $train --train-outcomes $trainOut `
  --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json
if ($LASTEXITCODE -ne 0) { throw "augmentation stage failed" }

Write-Host "[v2] 3) offline difficulty prior lookup"
$colA = if ($env:PRIOR_COL_A) { $env:PRIOR_COL_A -split ',' } else { @("local-llm/labels_axlight.jsonl", "local-llm/labels_ext.jsonl") }
$colB = if ($env:PRIOR_COL_B) { $env:PRIOR_COL_B -split ',' } else { @() }
$colArgs = @("--column") + $colA
if ($colB.Count -gt 0) { $colArgs += @("--column") + $colB }
& $py tools/build_prior_lookup.py --artifact $art @colArgs `
  --items colab-label/bundle/union.jsonl colab-label/bundle/all.jsonl colab-label/bundle/public_all.jsonl colab-label/bundle/ext.jsonl
if ($LASTEXITCODE -ne 0) { throw "prior lookup stage failed" }

Write-Host "[v2] 4) meta GBM"
& $py tools/build_meta_gbm.py --artifact $art `
  --train-input $train --train-outcomes $trainOut `
  --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json
if ($LASTEXITCODE -ne 0) { throw "meta stage failed" }

if (-not $env:DEPLOY_TRAIN_ONLY) {
  Write-Host "[v2] 5) public lookup"
  & $py tools/build_public_lookup.py --artifact $art --inputs data/materialized/train/inputs.json data/materialized/dev/inputs.json
  if ($LASTEXITCODE -ne 0) { throw "lookup stage failed" }
}

Write-Host "[v2] 6) pack heavy block"
& $py tools/pack_artifact.py --artifact $art
if ($LASTEXITCODE -ne 0) { throw "pack stage failed" }
Get-ChildItem $res\*.json | Select-Object Name, Length | Format-Table -AutoSize
Write-Host "[v2] DONE"
