<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-3.1-Light 자체 라벨링 (우선순위 1)

목표: 공개 Train/Dev(2,640)에 없는 **새 문항 + 새 outcome**을 만들어 light 예측기(fast tier의 핵심)에 보조
학습 데이터로 넣는다. 로컬 RTX 2050(4GB)은 2.5 tok/s라 규모 라벨링이 불가능하므로 Colab GPU(L4/A100)에서 vLLM으로 돌린다.

## 파일

| 파일 | 역할 | 어디서 |
|---|---|---|
| `build_pool.py` | 고정 공개 출처(GSM8K·Belebele·CRUXEval·RuleTaker·TruthfulQA·BABILong·HRMCR)에서 **주최측 템플릿 그대로** 새 문항 렌더링. `--verify`로 템플릿 정확도 검증(전 family 일치 확인 완료). 출력 `bundle/pilot.jsonl`(공개 1,951 + 주최 라벨) / `bundle/pool.jsonl`(신규 6,961) | 로컬 (완료) |
| `judge.py` | 답 추출·채점 (`\boxed{}`, `Answer:`/`정답:`, 마지막 숫자·글자·리터럴, HRMCR 날짜/띠 리스트). pilot·Colab 공용 | 양쪽 |
| `run_labels.py` | vLLM 생성기. `--stage pilot`(일치율 측정) / `pool`(라벨링) / `report`(재채점). 재개 지원 | Colab |
| `label_colab.ipynb` | 위를 순서대로 실행하는 노트북 | Colab |
| `make_zip.py` | `router_label_bundle.zip` 생성 (완료, 10.7MB) | 로컬 |
| `ingest_labels.py` | Colab 결과 zip → `data/aux/light-labels.v1.json` + `light-calibration.v1.json` | 로컬 |
| `../experiments/e41_aux_light.py` | aux 라벨을 light 헤드에 넣었을 때 CV EV 변화 (E27/E40 동일 하니스) | 로컬 |

## 사용자가 할 일 (Colab)

1. `official-router/colab-label/router_label_bundle.zip`을 Google Drive `MyDrive/`에 올린다 (또는 노트북 ①에서 직접 업로드).
2. `label_colab.ipynb`를 Colab에서 열고 **런타임 유형 → L4 또는 A100** 선택.
3. 셀 ①~⑤ 순서대로 실행. ③(pilot) 끝에 나오는 표에서
   - `within.25` ≥ 0.85, `outlen corr` ≥ 0.8 → ④ 계속
   - `within.25` < 0.75 → 중단, 표를 Claude에게 붙여넣기 (라벨 품질이 낮아 투입 무의미)
4. ⑤에서 받은 `router_label_out.zip`을 `official-router/colab-label/`에 둔다.

### pilot 표 읽는 법
- `within.25`(문항별 |주최 점수 − 우리 점수| ≤ 0.25 비율)와 `corr`이 핵심. 로컬 Q4 pilot은 0.63/0.58이었는데
  이는 양자화 모델의 오답 때문이라 bf16에서는 올라가야 정상.
- `outlen` 열(우리 출력 길이 / 주최 출력 길이 중앙값 비율)이 family마다 크게 다르면(로컬: belebele 3.6×, code 8.7×,
  gsm8k 0.3×) 비용 라벨은 못 쓰고 점수 라벨만 쓴다 → E41을 `SCOREONLY=1`로 실행 (아래).
- 두 온도 중 `within.25`가 높은 쪽을 노트북이 자동으로 pool에 사용.

## 그 다음 (로컬, Claude가 실행)

```bash
python colab-label/ingest_labels.py colab-label/router_label_out.zip
python experiments/e41_aux_light.py 0.5 ridge_gbm 7      # 가중 0.5, ridge+GBM light 헤드, 시드 7
python experiments/e41_aux_light.py 0.5 ridge_gbm 7 1 - 1  # 같은 설정, 점수 라벨만 사용(SCOREONLY)
```
기준선(W=0)은 s7 0.6980. 채택 기준: 3시드(7/17/23) 평균 +0.0012 이상, W에 대해 단봉.

## 로컬 pilot (완료, 참고용)

`tools/pilot_local_light.py`(llama-server Q4_K_M, 3 tok/s) 결과 — `reports/pilot_light_{raw,instr}_summary.md`:
raw 프롬프트는 이진 일치 0.80(code 0.53·ruletaker 0.40), **지시문 부착은 0.876**(aime/gsm8k 1.00, belebele/hrmcr/
ruletaker 0.93, truthfulqa 0.87, code 0.47). 출력 길이는 추론 family 0.5×, belebele 2.3×, code 7.6×로 재현 불가.
→ 노트북 ③은 raw와 `--instruct v1`을 둘 다 돌리고 within.25 높은 쪽을 pool에 쓴다(예상: v1). code는 주최측
판정과 어긋나므로(우리 정답을 주최는 0점) aux에서 code family는 가중을 낮추거나 제외를 검토.

## 규칙 검토

- 라우터 이미지에는 학습된 계수만 들어간다(aux 문항·라벨은 이미지 미포함) — "공개 자료로 미리 만든 학습 파일"에 해당.
- 출처는 모두 공식 `data/sources/source-pins.v1.json`에 고정된 공개 데이터셋 + Apache-2.0 공개 모델(`skt/A.X-3.1-Light`).
- 제출 저장소 문서에 학습 데이터 출처(라이선스 포함)를 적어야 하므로 `data/aux/*.json`의 `model`/`source` 필드를 그대로 인용하면 된다.

---

# E59 — 진짜 `skt/A.X-3.1`(34B) 중간 prior 컬럼 (Colab A100)

E56 시점의 prior는 컬럼 A(`A.X-3.1-Light`, 대상 `ax31-light`에 corr 0.739)와 컬럼 B
(`Qwen2.5-14B`, 대상 `ax31`에 corr **0.612**)로 되어 있다. EXPERIMENT_LOG의 "What would move the
score further"가 지목한 대로, **컬럼 B를 진짜 34B로 교체**하는 것이 남은 최대 레버다
(E51 시뮬레이션: 컬럼 A 수준 충실도의 3컬럼 → dev 0.722~0.724, 현재 0.7055).

| 파일 | 역할 |
|---|---|
| `e59_mid_column_colab.ipynb` | Colab A100 노트북. ④풀 재생성+digest 자가검증 → ⑤2,640 라벨링 → ⑥**충실도 게이트** → ⑦(통과 시)대량 라벨링 → ⑧회수 |
| `make_e59_zip.py` | `e59_colab_bundle.zip` 생성 (런타임 패키지 + 배포 아티팩트 + 공개 Train/Dev + 풀 빌더/라벨러, 8.3MB) |
| `to_prior_labels.py` | `run_labels.py` 출력 → `build_prior_lookup.py`가 먹는 행 형식(자기일관성 `sc`, `out_tokens_total` 포함) |
| `prior_column_report.py` | 충실도 게이트. 공개 2,640에 조인해 대상 모델별 corr/within.25, family별 분해, digest 겹침 보고 |
| `_smoke_prior_scripts.py` | 위 두 스크립트의 엔드투엔드 스모크(컬럼 A를 재생해 corr 0.736 복원 확인) |

**게이트 규칙**: `corr(새 컬럼, ax31)`이 컬럼 B의 **0.612 미만이면 대량 라벨링에 들어가지 않는다.**
2,640문항만 먼저 돌려 1시간 안에 판정하도록 셀 순서를 짰다.

**결과 반영(로컬)**: `e59_out.zip`을 풀고
```
python tools/build_prior_lookup.py --artifact <art> \
    --column local-llm/labels_axlight.jsonl local-llm/labels_ext.jsonl \
    --column colab-label/out/labels_mid_pool.jsonl colab-label/out/labels_gate.jsonl \
    --items colab-label/bundle/all.jsonl colab-label/bundle/ext.jsonl
```
컬럼 순서는 약→강이어야 한다(교차컬럼 delta가 업그레이드 이득의 대리이므로). 기존 Qwen 컬럼을
남길지(A,B,C) 교체할지(A,C)는 둘 다 재학습해 held-out으로 비교할 것.
