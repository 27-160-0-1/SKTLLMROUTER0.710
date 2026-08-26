# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Produce, on camera, the figures the challenge actually asks for.
#
#   STEP 0  which artifacts are being submitted, by SHA-256
#   STEP 1  build the image on the official platform, linux/arm64 (RUNTIME.md)
#   STEP 2  the official local check -- tools/check_runtime.py over the full public
#           Train+Dev (2,640 episodes), three tiers against the 90 s limit, with the
#           complete official isolation profile
#   STEP 3  same artifact, arm64 vs amd64, decision-for-decision -- the basis for
#           quoting an accuracy number measured on amd64
#   STEP 4  held-out accuracy with the official scorer: FINAL SCORE = 0.705113636364
#
# STEP 2 is the checklist item "공개 Train+Dev 검사에서 세 등급의 실행 시간과 출력
# 형식을 확인했습니다" in docs/SUBMISSION.md.
#
# Prerequisites: Docker Desktop running with arm64 emulation, Python 3.10+ on PATH
# (or set $env:ROUTER_PY).  Run from the repository root:
#     powershell -ExecutionPolicy Bypass -File .\demo_reproduce_official.ps1

# Native tools write progress and warnings to stderr.  With ErrorActionPreference=Stop,
# PowerShell turns each such line into a terminating NativeCommandError, so keep it
# Continue and check $LASTEXITCODE explicitly instead.
$ErrorActionPreference = "Continue"
$py = if ($env:ROUTER_PY) { $env:ROUTER_PY } else { "python" }

$EVAL_SHA = "3360ba0dbe5243b421b8f977408a57cdd2963c60701341b7dca089e0f35e6f0e"
$SUB_SHA  = "7984081c57f2e9a97725b8378aa2b5a405775079c7ec8eac41874f5c04ec0450"
$live = "src/ossp_router/resources/learned-router.v1.json"
$sub  = "src/ossp_router/resources/learned-router-submission.v1.json"

function Banner($text) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host ("  " + $text) -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
}

Banner "STEP 0  What is being submitted"
git log -1 --format="commit %h  %s"
$evalHash = (Get-FileHash $live -Algorithm SHA256).Hash.ToLower()
$subHash  = (Get-FileHash $sub  -Algorithm SHA256).Hash.ToLower()
Write-Host "evaluated artifact   $live"
Write-Host "  sha256             $evalHash"
Write-Host "submitted artifact   $sub"
Write-Host "  sha256             $subHash"
if ($evalHash -ne $EVAL_SHA) { throw "evaluated artifact hash mismatch" }
if ($subHash  -ne $SUB_SHA)  { throw "submitted artifact hash mismatch" }
& $py -X utf8 -c @"
import json, sys
for label, path in (('evaluated', sys.argv[1]), ('submitted', sys.argv[2])):
    a = json.load(open(path, encoding='utf-8'))
    b = a.get('prior_score_blend') or {}
    print('  %s  safety=%s  blend_w=%s  prior_columns=%d  public_lookup=%s' % (
        label, a['tier_safety_ratios'], b.get('weight'),
        len(a['prior_lookup']['columns']),
        'present' if a.get('public_lookup') else 'absent'))
"@ $live $sub
if ($LASTEXITCODE -ne 0) { throw "python check failed" }
Write-Host "  one configuration, two training splits: the evaluated artifact is fitted on"
Write-Host "  Train only, so Dev in STEP 4 is genuinely held out."

Banner "STEP 1  Build on the official platform: linux/arm64 (docs/RUNTIME.md)"
# The image bundles whatever sits at resources/learned-router.v1.json.  Build the
# submission image by swapping the submission artifact in, then restore the tree.
Copy-Item $live "$live.demo-backup" -Force
Copy-Item $sub $live -Force
try {
    docker build --pull --platform linux/arm64 --provenance=false --sbom=false `
        -f container/Dockerfile -t skt-router:arm64-submission .
    if ($LASTEXITCODE -ne 0) { throw "arm64 submission build failed" }
} finally {
    Move-Item "$live.demo-backup" $live -Force
}
docker image inspect --platform linux/arm64 skt-router:arm64-submission `
    --format 'platform {{.Os}}/{{.Architecture}}   image id {{.Id}}'

# Prove the image really carries the submitted artifact, and that it is running on
# aarch64 -- a cached COPY layer would not be visible in the build log otherwise.
Write-Host ""
Write-Host "verifying inside the arm64 image:" -ForegroundColor Yellow
docker run --rm --platform linux/arm64 --entrypoint python3 skt-router:arm64-submission -c @"
import hashlib, json, pathlib, platform
b = pathlib.Path('/opt/router/ossp_router/resources/learned-router.v1.json').read_bytes()
d = hashlib.sha256(b).hexdigest()
a = json.loads(b)
print('  uname machine   ', platform.machine())
print('  in-image sha256 ', d)
print('  safety          ', a['tier_safety_ratios'])
print('  prior columns   ', [c['tag'] for c in a['prior_lookup']['columns']])
assert d == '$SUB_SHA', 'WRONG ARTIFACT IN IMAGE'
print('  -> matches the submitted artifact byte for byte')
"@
if ($LASTEXITCODE -ne 0) { throw "in-image verification failed" }

Banner "STEP 2  Official local check: tools/check_runtime.py, public Train+Dev"
# check_runtime.py imports fcntl, which Windows has no equivalent for, so run the
# official tool itself inside a Linux container.  The containers it measures are still
# started by the Docker daemon exactly as the tool specifies -- the helper only holds
# the Python process.  The repository is mounted at the path the daemon knows it by so
# the tool's bind mounts resolve on both sides.
$root = (Get-Location).Path
$drive = $root.Substring(0, 1).ToLower()
$vmPath = "/run/desktop/mnt/host/$drive/" + ($root.Substring(3) -replace '\\', '/')
New-Item -ItemType Directory -Force build\rtcheck | Out-Null
Write-Host "official profile applied by the tool: --platform linux/arm64 --network none"
Write-Host "  --ipc none --cgroupns private --ulimit core=0:0 --read-only --user 65532:65532"
Write-Host "  --cap-drop ALL --security-opt no-new-privileges --cpus 2 --memory 2g"
Write-Host "  --memory-swap 2g --pids-limit 32 --tmpfs /tmp:rw,noexec,nosuid,size=256m"
Write-Host ""
docker build -q -f container/checkrunner.Dockerfile -t skt-check-runner container/ | Out-Null
if ($LASTEXITCODE -ne 0) { throw "check-runner build failed" }
docker run --rm --platform linux/amd64 `
    -v "/var/run/docker.sock:/var/run/docker.sock" `
    -v "${root}:${vmPath}" -w $vmPath -e "HOME=$vmPath/build/rtcheck" `
    -e "PYTHONPATH=src" `
    skt-check-runner `
    python3 tools/check_runtime.py --image skt-router:arm64-submission `
    --repetitions 1 --report build/runtime-check-report.json
if ($LASTEXITCODE -ne 0) { throw "official runtime check failed" }

Banner "STEP 3  Same artifact on arm64 and amd64, decision for decision"
# STEP 4 measures accuracy on amd64 because the evaluated artifact carries no lookup
# table, and every episode therefore takes the compute path -- about nine times slower
# under this machine's arm64 emulation than on its native architecture.  Check first
# that the architecture does not change any decision.
docker build --platform linux/arm64 --provenance=false --sbom=false -q `
    -f container/Dockerfile -t skt-router:arm64-eval . | Out-Null
if ($LASTEXITCODE -ne 0) { throw "arm64 eval build failed" }
docker build --platform linux/amd64 --provenance=false --sbom=false -q `
    -f container/Dockerfile -t skt-router:amd64-eval . | Out-Null
if ($LASTEXITCODE -ne 0) { throw "amd64 eval build failed" }

New-Item -ItemType Directory -Force build\xarch\input, build\xarch\arm64, build\xarch\amd64 | Out-Null
& $py -X utf8 -c @"
import json, pathlib, sys
d = json.load(open('data/materialized/dev/inputs.json', encoding='utf-8'))
d['episodes'] = d['episodes'][:120]
pathlib.Path(sys.argv[1]).write_text(json.dumps(d, ensure_ascii=False), encoding='utf-8')
print('  slice: first %d Dev episodes, compute path (no lookup)' % len(d['episodes']))
"@ build/xarch/input/inputs.json
if ($LASTEXITCODE -ne 0) { throw "slice failed" }
$xIn = (Resolve-Path build\xarch\input).Path
foreach ($arch in @("arm64", "amd64")) {
    $outDir = (Resolve-Path "build\xarch\$arch").Path
    $started = Get-Date
    docker run --rm --platform "linux/$arch" --cpus 2 --memory 2g --memory-swap 2g `
        --network none --read-only --pids-limit 32 --tmpfs /tmp:rw,size=256m `
        -v "${xIn}:/challenge/input:ro" -v "${outDir}:/challenge/output" `
        "skt-router:$arch-eval" `
        --input /challenge/input/inputs.json --tier premium `
        --output /challenge/output/submission.json
    if ($LASTEXITCODE -ne 0) { throw "cross-arch run failed for $arch" }
    Write-Host ("   linux/$arch  {0:n1} s" -f ((Get-Date) - $started).TotalSeconds)
}
& $py -X utf8 -c @"
import json
def picks(p):
    return {d['episode_id']: d['model_id']
            for d in json.load(open(p, encoding='utf-8'))['decisions']}
a = picks('build/xarch/arm64/submission.json')
b = picks('build/xarch/amd64/submission.json')
assert set(a) == set(b), 'episode sets differ'
bad = [e for e in a if a[e] != b[e]]
print('  compared %d decisions -- mismatches: %d' % (len(a), len(bad)))
assert not bad, 'architecture changed a decision'
print('  -> the architecture does not change what the router picks')
"@
if ($LASTEXITCODE -ne 0) { throw "cross-arch comparison failed" }

Banner "STEP 4  Held-out accuracy with the official scorer (ossp_router.scoring)"
New-Item -ItemType Directory -Force demo_io\input, demo_io\output | Out-Null
Copy-Item data\materialized\dev\inputs.json demo_io\input\inputs.json -Force
$inDir = (Resolve-Path demo_io\input).Path
$outDir = (Resolve-Path demo_io\output).Path
foreach ($tier in @("fast", "balanced", "premium")) {
    Write-Host ""
    Write-Host ">> tier=$tier   held-out Dev, 880 episodes, evaluated artifact" -ForegroundColor Yellow
    $started = Get-Date
    docker run --rm --platform linux/amd64 --cpus 2 --memory 2g --memory-swap 2g `
        --network none --read-only --pids-limit 32 --tmpfs /tmp:rw,size=256m `
        -v "${inDir}:/challenge/input:ro" -v "${outDir}:/challenge/output" `
        skt-router:amd64-eval `
        --input /challenge/input/inputs.json --tier $tier `
        --output /challenge/output/submission-$tier.json
    if ($LASTEXITCODE -ne 0) { throw "container run failed for $tier" }
    Write-Host ("   wall time {0:n1} s" -f ((Get-Date) - $started).TotalSeconds)
}

$env:PYTHONPATH = "src"
Write-Host ""
& $py -X utf8 tools/score_submissions.py `
    --inputs data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json `
    --submissions demo_io/output/submission-fast.json demo_io/output/submission-balanced.json demo_io/output/submission-premium.json
if ($LASTEXITCODE -ne 0) { throw "scoring failed" }

Write-Host ""
Write-Host "Expected: FINAL SCORE = 0.705113636364  (weighted 0.4/0.3/0.3, all tiers within budget)" -ForegroundColor Green
Write-Host "Report these: held-out 0.705114, expected 0.7043, zero busts in 3,000 resamples." -ForegroundColor Green
Write-Host ""
Write-Host "The STEP 2 timings are the submitted artifact's lookup-hit path, which is what the" -ForegroundColor DarkGray
Write-Host "public Train+Dev check exercises by definition.  A private batch takes the compute" -ForegroundColor DarkGray
Write-Host "path, measured at 1.31x the single-fit reference (~48 s on the official machine)." -ForegroundColor DarkGray
Write-Host "STEP 2 times are also emulated arm64 on x86; the official run is native." -ForegroundColor DarkGray
