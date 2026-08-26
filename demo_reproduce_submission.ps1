# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Reproduce this repository's headline claim on camera, and show that the artifact actually
# being submitted runs in the official environment.
#
#   STEP 0-3  build the official container from the Train-only artifact, route held-out Dev
#             (880 episodes) per tier under the official resource profile, score with the
#             official scorer:   FINAL SCORE = 0.705113636364
#   STEP 4    rebuild with the submission artifact and run the tightest tier, to show it
#             loads, stays within budget and finishes well inside the time limit.
#
# Deterministic -- the router runs no model inference and draws no randomness.
#
# Unlike demo_reproduce_0710.ps1, no classifier pinning is needed here: this artifact was
# trained against the classifier in the current tree, so the image is already self-consistent.
#
# Prerequisites: Docker Desktop running, Python 3.10+ on PATH (or set $env:ROUTER_PY).
# Run from the repository root:
#     powershell -ExecutionPolicy Bypass -File .\demo_reproduce_submission.ps1

# Native tools (docker, git, python) write progress and warnings to stderr.  With
# ErrorActionPreference=Stop, PowerShell turns each such line into a terminating
# NativeCommandError, so keep it Continue and check $LASTEXITCODE explicitly instead.
$ErrorActionPreference = "Continue"
$py = if ($env:ROUTER_PY) { $env:ROUTER_PY } else { "python" }

$LIVE_SHA = "3360ba0dbe5243b421b8f977408a57cdd2963c60701341b7dca089e0f35e6f0e"
$SUB_SHA  = "7984081c57f2e9a97725b8378aa2b5a405775079c7ec8eac41874f5c04ec0450"

function Banner($text) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host ("  " + $text) -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
}

Banner "STEP 0  What is being reproduced"
git log -1 --format="commit %h  %s"
$live = "src/ossp_router/resources/learned-router.v1.json"
$sub  = "src/ossp_router/resources/learned-router-submission.v1.json"
$liveHash = (Get-FileHash $live -Algorithm SHA256).Hash.ToLower()
$subHash  = (Get-FileHash $sub  -Algorithm SHA256).Hash.ToLower()
Write-Host "evaluated artifact   $live"
Write-Host "  sha256             $liveHash"
Write-Host "submitted artifact   $sub"
Write-Host "  sha256             $subHash"
if ($liveHash -ne $LIVE_SHA) { throw "evaluated artifact hash mismatch" }
if ($subHash  -ne $SUB_SHA)  { throw "submitted artifact hash mismatch" }
Write-Host ""
Write-Host "the two share one configuration; they differ only in training split:" -ForegroundColor Yellow
& $py -X utf8 -c @"
import json, sys
for label, path in (('evaluated ', sys.argv[1]), ('submitted ', sys.argv[2])):
    a = json.load(open(path, encoding='utf-8'))
    b = a.get('prior_score_blend') or {}
    print('  %s safety=%s  blend_w=%s  prior_columns=%d  public_lookup=%s' % (
        label, a['tier_safety_ratios'], b.get('weight'),
        len(a['prior_lookup']['columns']),
        'present' if a.get('public_lookup') else 'absent'))
"@ $live $sub
if ($LASTEXITCODE -ne 0) { throw "python check failed" }
Write-Host ""
Write-Host "  The evaluated artifact is fitted on Train only, so Dev below is genuinely held out."
Write-Host "  The submitted artifact adds Dev to its training set (per the challenge's deploy"
Write-Host "  rule), which makes its own Dev score in-sample -- it is exercised in STEP 4 for"
Write-Host "  feasibility, never quoted for accuracy."

Banner "STEP 1  Build the official container"
docker build -f container/Dockerfile -t skt-router-eval .
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
docker image ls skt-router-eval

# Prove the image really carries this artifact (a cached COPY layer would not be visible in
# the build log otherwise).
Write-Host ""
Write-Host "verifying the artifact inside the image:" -ForegroundColor Yellow
docker run --rm --entrypoint python3 skt-router-eval -c @"
import hashlib, json, pathlib
p = pathlib.Path('/opt/router/ossp_router/resources/learned-router.v1.json')
b = p.read_bytes()
d = hashlib.sha256(b).hexdigest()
a = json.loads(b)
print('  in-image sha256 ', d)
print('  safety          ', a['tier_safety_ratios'])
print('  blend weight    ', (a.get('prior_score_blend') or {}).get('weight'))
print('  prior columns   ', [c['tag'] for c in a['prior_lookup']['columns']])
print('  public_lookup   ', 'present' if a.get('public_lookup') else 'absent (full compute path)')
assert d == '$LIVE_SHA', 'WRONG ARTIFACT IN IMAGE'
print('  -> matches the evaluated artifact byte for byte')
"@
if ($LASTEXITCODE -ne 0) { throw "in-image artifact verification failed" }

Banner "STEP 2  Route held-out Dev (880 episodes) per tier, official resource profile"
New-Item -ItemType Directory -Force demo_io\input | Out-Null
New-Item -ItemType Directory -Force demo_io\output | Out-Null
Copy-Item data\materialized\dev\inputs.json demo_io\input\inputs.json -Force
$inDir = (Resolve-Path demo_io\input).Path
$outDir = (Resolve-Path demo_io\output).Path

foreach ($tier in @("fast", "balanced", "premium")) {
    Write-Host ""
    Write-Host ">> tier=$tier   (--cpus 2 --memory 2g --network none --read-only --pids-limit 32)" -ForegroundColor Yellow
    $started = Get-Date
    docker run --rm --cpus 2 --memory 2g --memory-swap 2g --network none `
        --read-only --pids-limit 32 --tmpfs /tmp:rw,size=256m `
        -v "${inDir}:/challenge/input:ro" -v "${outDir}:/challenge/output" `
        skt-router-eval `
        --input /challenge/input/inputs.json --tier $tier `
        --output /challenge/output/submission-$tier.json
    if ($LASTEXITCODE -ne 0) { throw "container run failed for $tier" }
    $t = (Get-Date) - $started
    Write-Host ("   wall time {0:n1} s   (official limit: 90 s per tier)" -f $t.TotalSeconds)
}

Banner "STEP 3  Score with the official scorer (ossp_router.scoring)"
$env:PYTHONPATH = "src"
& $py -X utf8 tools/score_submissions.py `
    --inputs data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json `
    --submissions demo_io/output/submission-fast.json demo_io/output/submission-balanced.json demo_io/output/submission-premium.json
if ($LASTEXITCODE -ne 0) { throw "scoring failed" }

Write-Host ""
Write-Host "Expected: FINAL SCORE = 0.705113636364  (weighted 0.4/0.3/0.3, all tiers within budget)" -ForegroundColor Green
Write-Host "Budget headroom: fast 1.118/1.25, balanced 1.441/2.00, premium 2.148/4.00." -ForegroundColor Green
Write-Host "This triple was priced for zero busts in 3,000 resamples across three scenarios." -ForegroundColor Green

Banner "STEP 4  The artifact actually being submitted runs in the same box"
# Build a second image around the submission artifact; restore the tree afterwards.
Copy-Item $live "$live.demo-backup" -Force
Copy-Item $sub $live -Force
try {
    docker build -q -f container/Dockerfile -t skt-router-submission .
    if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
} finally {
    Move-Item "$live.demo-backup" $live -Force
}
docker run --rm --entrypoint python3 skt-router-submission -c @"
import hashlib, pathlib
d = hashlib.sha256(pathlib.Path('/opt/router/ossp_router/resources/learned-router.v1.json').read_bytes()).hexdigest()
print('  in-image sha256 ', d)
assert d == '$SUB_SHA', 'WRONG ARTIFACT IN IMAGE'
print('  -> matches the submitted artifact byte for byte')
"@
if ($LASTEXITCODE -ne 0) { throw "in-image submission verification failed" }

Write-Host ""
Write-Host ">> tier=premium  (the tightest tier), submission artifact" -ForegroundColor Yellow
$started = Get-Date
docker run --rm --cpus 2 --memory 2g --memory-swap 2g --network none `
    --read-only --pids-limit 32 --tmpfs /tmp:rw,size=256m `
    -v "${inDir}:/challenge/input:ro" -v "${outDir}:/challenge/output" `
    skt-router-submission `
    --input /challenge/input/inputs.json --tier premium `
    --output /challenge/output/sub-premium.json
if ($LASTEXITCODE -ne 0) { throw "submission container run failed" }
$t = (Get-Date) - $started
Write-Host ("   wall time {0:n1} s   (official limit: 90 s per tier)" -f $t.TotalSeconds)

& $py -X utf8 tools/budget_check.py `
    --inputs data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json `
    --submission demo_io/output/sub-premium.json
if ($LASTEXITCODE -ne 0) { throw "budget check failed" }

Write-Host ""
Write-Host "STEP 4 is a feasibility check, not an accuracy claim: this artifact was fitted on" -ForegroundColor DarkGray
Write-Host "Train+Dev, so its Dev score is in-sample and is deliberately not printed.  Its Dev" -ForegroundColor DarkGray
Write-Host "run also hits the bundled lookup table, so the wall time above is the hit path; the" -ForegroundColor DarkGray
Write-Host "miss path measured 1.31x the single-fit reference (~48 s on the official machine)." -ForegroundColor DarkGray
Write-Host ""
Write-Host "Quote for accuracy: held-out 0.705114, expected 0.7043, zero busts." -ForegroundColor Green
