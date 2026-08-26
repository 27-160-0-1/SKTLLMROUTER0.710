# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Reproduce the 0.7100 build on camera: build the official container from this repository,
# route the held-out Dev split (880 episodes) per tier under the official resource profile,
# and score the three submissions with the official scorer.  Deterministic -- the router runs
# no model inference and draws no randomness, so the final line is always:
#
#     FINAL SCORE = 0.709971590909...
#
# Prerequisites: Docker Desktop running, Python 3.10+ on PATH (or set $env:ROUTER_PY).
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File .\demo_reproduce_0710.ps1

# Native tools (docker, git, python) write progress and warnings to stderr.  With
# ErrorActionPreference=Stop, PowerShell turns each such line into a terminating
# NativeCommandError, so keep it Continue and check $LASTEXITCODE explicitly instead.
$ErrorActionPreference = "Continue"
$py = if ($env:ROUTER_PY) { $env:ROUTER_PY } else { "python" }

function Banner($text) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host ("  " + $text) -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
}

Banner "STEP 0  What is being reproduced"
git log -1 --format="commit %h  %s"
$artifact = "src/ossp_router/resources/learned-router-0710.v1.json"
$sha = (Get-FileHash $artifact -Algorithm SHA256).Hash.ToLower()
Write-Host "artifact  $artifact"
Write-Host "sha256    $sha"
if ($sha -ne "f279218a7f7a02ce32ce75e94d21b6250054b7aebaa28baadd97a60a54bccf9a") {
    throw "artifact hash mismatch -- this is not the 0.7100 build"
}
& $py -c "import json; a=json.load(open(r'$artifact',encoding='utf-8')); print('safety   ', a['tier_safety_ratios']); print('columns  ', [c['tag'] for c in a['prior_lookup']['columns']])"
if ($LASTEXITCODE -ne 0) { throw "python check failed" }

Banner "STEP 1  Build the official container with the 0.7100 artifact"
# The image bundles whatever sits at resources/learned-router.v1.json; swap in the 0710
# artifact for the build, restore afterwards.  The swap is shown on purpose.
$live = "src/ossp_router/resources/learned-router.v1.json"
Copy-Item $live "$live.demo-backup" -Force
Copy-Item $artifact $live -Force
# The runtime must be the one this artifact was trained against.  `similarity.classify_family`
# was rebuilt after 0.7100 shipped (E67, 91.4 % -> 99.85 % against the true source); feeding the
# new labels to the old artifact's meta GBM shifts the score to 0.710199.  Pin the classifier to
# the artifact's own commit so the image is internally consistent.
$sim = "src/ossp_router/similarity.py"
Copy-Item $sim "$sim.demo-backup" -Force
# Byte-exact: piping `git show` through PowerShell decodes with the console codepage and
# mangles the Korean regex literals in this file.  Redirect inside cmd instead, then verify.
cmd /c "git show f0b29e3:src/ossp_router/similarity.py > `"$sim`""
if ($LASTEXITCODE -ne 0) { throw "git show failed" }
$blobSha = git rev-parse "f0b29e3:src/ossp_router/similarity.py"
$fileSha = git hash-object $sim
if ($blobSha -ne $fileSha) { throw "pinned classifier does not match the blob ($fileSha vs $blobSha)" }
Write-Host "runtime classifier pinned to f0b29e3 (git blob $($blobSha.Substring(0,12)), byte-exact)"
try {
    docker build -f container/Dockerfile -t skt-router-0710 .
    if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
} finally {
    Move-Item "$live.demo-backup" $live -Force
    Move-Item "$sim.demo-backup" $sim -Force
}
docker image ls skt-router-0710

# Prove the image really carries the 0.7100 artifact (a cached COPY layer would not be visible
# in the build log otherwise).
Write-Host ""
Write-Host "verifying the artifact inside the image:" -ForegroundColor Yellow
docker run --rm --entrypoint python3 skt-router-0710 -c @"
import hashlib, json, pathlib
p = pathlib.Path('/opt/router/ossp_router/resources/learned-router.v1.json')
b = p.read_bytes()
d = hashlib.sha256(b).hexdigest()
a = json.loads(b)
print('  in-image sha256 ', d)
print('  safety          ', a['tier_safety_ratios'])
print('  prior columns   ', [c['tag'] for c in a['prior_lookup']['columns']])
print('  public_lookup   ', 'present' if a.get('public_lookup') else 'absent (full compute path)')
assert d == 'f279218a7f7a02ce32ce75e94d21b6250054b7aebaa28baadd97a60a54bccf9a', 'WRONG ARTIFACT IN IMAGE'
print('  -> matches the 0.7100 artifact byte for byte')
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
        skt-router-0710 `
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
Write-Host "Expected: FINAL SCORE = 0.709971590909  (weighted 0.4/0.3/0.3, all tiers within budget)" -ForegroundColor Green
Write-Host ""
Write-Host "Artifact and runtime are both from commit f0b29e3.  The same artifact against the" -ForegroundColor DarkGray
Write-Host "current tree's classifier (E67) scores 0.710198863636 instead - a code pairing, not" -ForegroundColor DarkGray
Write-Host "noise; the container's picks match this Windows host on all 2,640 decisions." -ForegroundColor DarkGray
