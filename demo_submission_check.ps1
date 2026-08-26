# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# 제출물 시연: 주최측이 제공한 명령만 사용해 규격 적합성과 점수를 재현한다.
#
#   S1 무엇을 제출하는가      저장소·커밋·환경
#   S2 이미지가 실을 산출물   SHA-256 과 구성
#   S3 공식 빌드 명령         README.md / docs/RUNTIME.md 원문 그대로
#   S4 이미지 내부            그 산출물이 실제로 들어갔는가, 무엇이 함께 들어갔는가
#   S5 접수 전 사전검증       docs/RUNTIME.md 의 거부 조건
#   S6 공식 검사              tools/check_runtime.py — 공개 Train+Dev 2,640문항 x 3등급 x 90초
#   S7 레지스트리 왕복        다이제스트로 push -> 로컬 삭제 -> 다이제스트로 pull
#   S8 3등급 실행 + 공식 채점 공식 격리 프로필 + ossp_router.cli self-check
#   S9 기술 제출 정보 파일    tools/validate_technical_submission.py
#
# 실행:  powershell -ExecutionPolicy Bypass -File .\demo_submission_check.ps1

# 콘솔 코드페이지를 UTF-8 로 먼저 고정한다. 이 스크립트의 한글 출력과, python·docker 가
# 내보내는 UTF-8 바이트를 같은 인코딩으로 맞춰야 터미널 녹화(TSV)가 깨지지 않는다.
# 첫 출력보다 앞에 있어야 한다 -- 한 줄이라도 찍힌 뒤에는 writer 가 이미 만들어져 효과가 없다.
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch { }
$OutputEncoding = [Console]::OutputEncoding

$ErrorActionPreference = "Continue"
$py = if ($env:ROUTER_PY) { $env:ROUTER_PY } else { "python" }
$SUB_SHA = "7984081c57f2e9a97725b8378aa2b5a405775079c7ec8eac41874f5c04ec0450"
$REG = "localhost:5000"
$REPO = "$REG/skt/router"

function Banner($t) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host ("  " + $t) -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
}

Banner "S1  What is being submitted"
git log -1 --format="commit  %H"
git log -1 --format="        %s"
Write-Host "working tree:"
$st = git status --short
if ($st) { $st } else { Write-Host "  clean" }
docker version --format "docker  client {{.Client.Version}} / server {{.Server.Version}} ({{.Server.Os}}/{{.Server.Arch}})"
& $py --version

Banner "S2  이미지가 실을 학습 산출물"
$live = "src\ossp_router\resources\learned-router.v1.json"
$h = (Get-FileHash $live -Algorithm SHA256).Hash.ToLower()
Write-Host "$live"
Write-Host "  sha256  $h"
if ($h -ne $SUB_SHA) { throw "제출 아티팩트가 아닙니다" }
& $py -X utf8 -c @"
import json
a = json.load(open('src/ossp_router/resources/learned-router.v1.json', encoding='utf-8'))
print('  safety        ', a['tier_safety_ratios'])
print('  blend weight  ', a['prior_score_blend']['weight'])
print('  prior columns ', [c['tag'] for c in a['prior_lookup']['columns']])
print('  public lookup  %d entries' % len(a['public_lookup']['entries']))
"@
Write-Host "  -> 공개 자료로 미리 만든 조회표와 계수뿐. 실행 중 모델 호출 없음." -ForegroundColor DarkGray

Banner "S3  공식 빌드 명령 (README.md / docs/RUNTIME.md 원문)"
Write-Host "docker build --pull --platform linux/arm64 --file container/Dockerfile --tag my-router:check ." -ForegroundColor Yellow
docker build --pull --platform linux/arm64 --file container/Dockerfile --tag my-router:check .
if ($LASTEXITCODE -ne 0) { throw "빌드 실패" }

Banner "S4  이미지 안에 무엇이 들어갔는가"
docker run --rm --platform linux/arm64 --entrypoint python3 my-router:check -c @"
import hashlib, pathlib, platform
b = pathlib.Path('/opt/router/ossp_router/resources/learned-router.v1.json').read_bytes()
d = hashlib.sha256(b).hexdigest()
print('  uname machine  ', platform.machine())
print('  in-image sha256', d)
assert d == '$SUB_SHA', 'WRONG ARTIFACT IN IMAGE'
print('  -> matches the submitted artifact byte for byte')
"@
if ($LASTEXITCODE -ne 0) { throw "이미지 내부 검증 실패" }
Write-Host ""
Write-Host "이미지에 들어간 파일 전체:" -ForegroundColor Yellow
docker run --rm --platform linux/arm64 --entrypoint sh my-router:check -c "find /opt/router -type f | sort"
Write-Host "  -> 데이터도, 실험 코드도, 네트워크 의존도 없음" -ForegroundColor DarkGray

Banner "S5  접수 전 사전검증 (docs/RUNTIME.md)"
docker image inspect --platform linux/arm64 my-router:check --format 'platform    {{.Os}}/{{.Architecture}}'
# docker 가 내보내는 바이트는 콘솔 코드페이지를 거치므로 서식 문자열은 ASCII 로 둔다.
docker image inspect --platform linux/arm64 my-router:check --format 'VOLUME      {{if .Config.Volumes}}declared -> rejected at intake{{else}}none -> OK{{end}}'
docker image inspect --platform linux/arm64 my-router:check --format 'user        {{.Config.User}}'
docker image inspect --platform linux/arm64 my-router:check --format 'entrypoint  {{json .Config.Entrypoint}}'

Banner "S6  공식 검사: tools/check_runtime.py (공개 Train+Dev 2,640문항)"
Write-Host "PYTHONPATH=src python3 tools/check_runtime.py --image my-router:check --report build/runtime-check-report.json" -ForegroundColor Yellow
Write-Host "(Windows 에는 fcntl 이 없어 이 도구만 Linux 헬퍼 컨테이너 안에서 실행)" -ForegroundColor DarkGray
$root = (Get-Location).Path
$vm = "/run/desktop/mnt/host/" + $root.Substring(0,1).ToLower() + "/" + ($root.Substring(3) -replace '\\','/')
New-Item -ItemType Directory -Force build\rtcheck | Out-Null
docker build -q -f container/checkrunner.Dockerfile -t skt-check-runner container/ | Out-Null
docker run --rm --platform linux/amd64 -v "/var/run/docker.sock:/var/run/docker.sock" -v "${root}:$vm" -w $vm -e "HOME=$vm/build/rtcheck" -e "PYTHONPATH=src" skt-check-runner python3 tools/check_runtime.py --image my-router:check --repetitions 1 --report build/runtime-check-report.json
if ($LASTEXITCODE -ne 0) { throw "공식 검사 실패" }

Banner "S7  레지스트리 왕복: 다이제스트로 push, 로컬 삭제, 다이제스트로 pull"
if (-not (docker ps --filter "name=localreg" --format "{{.Names}}")) {
    docker run -d -p 5000:5000 --name localreg registry:2 | Out-Null
    Start-Sleep -Seconds 3
}
docker tag my-router:check "${REPO}:sub"
docker push "${REPO}:sub" | Select-Object -Last 1
$dg = ((docker image inspect "${REPO}:sub" --format '{{range .RepoDigests}}{{println .}}{{end}}') | Where-Object { $_ -like "$REG*" } | Select-Object -First 1).Trim()
Write-Host ""
Write-Host "제출 다이제스트  $dg" -ForegroundColor Green
docker rmi -f "${REPO}:sub" my-router:check | Out-Null
Write-Host "로컬 이미지를 지우고 다이제스트로만 다시 받는다:" -ForegroundColor Yellow
docker pull --platform linux/arm64 $dg | Select-Object -Last 1
if ($LASTEXITCODE -ne 0) { throw "pull 실패" }

Banner "S8  공식 자원 프로필로 3등급 실행 + 공식 채점기"
New-Item -ItemType Directory -Force demo_io\in, demo_io\score | Out-Null
Copy-Item data\materialized\dev\inputs.json demo_io\in\inputs.json -Force
$inDir = (Resolve-Path demo_io\in).Path
foreach ($t in @("fast","balanced","premium")) {
    New-Item -ItemType Directory -Force "demo_io\out\$t" | Out-Null
    $outDir = (Resolve-Path "demo_io\out\$t").Path
    $s = Get-Date
    docker run --rm --platform linux/arm64 --network none --read-only --user 65532:65532 --cap-drop ALL --security-opt no-new-privileges --ipc none --cgroupns private --ulimit core=0:0 --cpus 2 --memory 2g --memory-swap 2g --pids-limit 32 --tmpfs /tmp:rw,noexec,nosuid,size=256m -v "${inDir}:/challenge/input:ro" -v "${outDir}:/challenge/output" $dg --input /challenge/input/inputs.json --tier $t --output /challenge/output/submission.json
    if ($LASTEXITCODE -ne 0) { throw "$t 실행 실패" }
    $n = (Get-ChildItem "demo_io\out\$t").Count
    Write-Host ("   {0,-9} {1,5:n1} s   출력 볼륨 파일 {2}개 (규격: submission.json 하나만)" -f $t, ((Get-Date)-$s).TotalSeconds, $n)
    Copy-Item "demo_io\out\$t\submission.json" "demo_io\score\$t.json" -Force
}
Write-Host ""
$env:PYTHONPATH = "src"
& $py -X utf8 -m ossp_router.cli self-check --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json --submissions demo_io/score --report demo_io/report.json
if ($LASTEXITCODE -ne 0) { throw "채점 실패" }
& $py -X utf8 -c @"
import json
r = json.load(open('demo_io/report.json', encoding='utf-8'))
for t in ('fast', 'balanced', 'premium'):
    x = r['tiers'][t]
    print('  %-9s score=%s  budget_ratio=%s  passed=%s' % (t, x['tier_score'], x['budget_ratio'], x['budget_passed']))
print('  FINAL SCORE = %s' % r['final_score'])
"@
Write-Host ""
Write-Host "이 이미지의 산출물은 Train+Dev 로 학습했으므로 위 Dev 점수는 in-sample 이다." -ForegroundColor DarkGray
Write-Host "성능 주장에 쓰는 홀드아웃 수치는 Train 만으로 학습한 산출물의 0.705113636364 이며" -ForegroundColor DarkGray
Write-Host "같은 공식 채점기로 별도 측정했다 (demo_reproduce_official.ps1 STEP 4)." -ForegroundColor DarkGray

Banner "S9  기술 제출 정보 파일"
Write-Host "python tools/validate_technical_submission.py" -ForegroundColor Yellow
& $py tools\validate_technical_submission.py
Write-Host ""
Write-Host "제출 순서: 코드 커밋 -> 그 커밋에서 arm64 빌드 -> 공개 레지스트리 push ->" -ForegroundColor DarkGray
Write-Host "다이제스트 확인 -> submission-ossp-skt.json 만 추가한 별도 커밋 -> 그 커밋 tree URL 을 보고서에 기재" -ForegroundColor DarkGray
Write-Host ""
