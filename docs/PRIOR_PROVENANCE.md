<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# 오프라인 prior 조회표 — 출처·라이선스·변환 기록

`SUBMISSION.md`의 기록 의무("이름·용도·공개 업스트림 URL / 고정 리비전과 SHA-256 / 라이선스
근거 / 변환 기록")에 따른 문서다. 제출 아티팩트의 `prior_lookup`은 4개 컬럼을 담으며, 각각
**가중치가 공개된 모델**을 **오프라인에서** 공개 자료 문항에 돌려 얻은 (점수, 로그 출력길이,
자기일관성)을 **프롬프트 SHA-256 키**로 저장한 것이다. 아티팩트에는 해시와 수치만 들어 있고
프롬프트 원문·생성 텍스트는 포함하지 않는다. 평가 실행 중 모델 호출은 없다
(`CHALLENGE_RULES.md` "사용할 수 있는 정보"의 조회표·프롬프트 해시 조회 허용 조항,
"공개 자료로 미리 만든 학습 파일" 조항에 근거).

## 컬럼별 기록

| 컬럼 태그 | 모델 (업스트림) | 라이선스 | 항목 수 | 생성 조건 |
|---|---|---|---:|---|
| `axlight-q6k-v2` | `skt/A.X-3.1-Light` — huggingface.co/skt/A.X-3.1-Light | Apache-2.0 | 38,330 | llama.cpp Q6_K, n=4, 이전 릴리스 라인에서 생성·계승 |
| `qwen2.5-14b-q4km` | `Qwen/Qwen2.5-14B-Instruct` — huggingface.co/Qwen/Qwen2.5-14B-Instruct | Apache-2.0 | 8,861 | llama.cpp Q4_K_M, n=4, 이전 릴리스 라인에서 생성·계승 |
| `ax31-34b-v1` | `skt/A.X-3.1` — huggingface.co/skt/A.X-3.1 | Apache-2.0 | 39,507 | vLLM + bitsandbytes NF4, Colab A100-40GB, n=4, T=0.7, family별 형식 지시문 v1, max_tokens 384(AIME 2048), 2026-08-21~22 실행 |
| `r1-distill-qwen-14b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` — huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | MIT | 2,640 | vLLM + bitsandbytes NF4, Colab A100-40GB, n=2, T=0.6, max_tokens 4096, 지시문 없음, **길이·자기일관성만 저장(점수 미사용)**, 2026-08-23~24 실행 |

컬럼 A·B는 이전 릴리스 라인(`LLM-ROUTE` 계열 제출 저장소)이 생성해 제출 아티팩트에 실었던
것을 컴파일된 형태로 계승했다(원 라벨 파일은 해당 라인에 있음). 네 모델 모두 상업적 이용·
재배포·변형을 허용하는 라이선스로, "모델·토크나이저·학습 파일은 상업적 이용, 이미지 안의
재배포, 필요한 변형과 평가 목적 사용을 허용해야" 요건을 충족한다.

리비전 참고: 컬럼 C·D의 모델 가중치는 실행 시점(위 날짜)의 HF `main` 리비전으로 내려받았고
별도 커밋 해시로 고정하지 않았다. 두 저장소 모두 해당 시점 이후 가중치 파일 변경이 없음을
모델 카드에서 확인할 수 있으며, 조회표 자체가 결과의 고정본이므로 재현 대상은 가중치가 아니라
아래 SHA-256의 컴파일 산출물이다.

## 라벨링 대상 문항의 출처

- 공개 출처 렌더링 풀: `colab-label/build_pool.py` · `build_pool_ext.py` — 업스트림 데이터셋과
  리비전·파일 SHA-256은 `data/sources/source-pins.v1.json`에 고정, 라이선스는
  `DATA_LICENSES.md` 참조.
- AIME 문항: `colab-label/build_pool_aime.py` — `HuggingFaceH4/aime_2024`, `math-ai/aime25`,
  `allenai/aime-2022-2025`, `AI-MO/aimo-validation-aime`에서 원문 그대로 렌더링. 대회 공개
  분할의 AIME 36문항 중 35문항이 프롬프트 SHA-256 단위로 정확히 재현됨을 스크립트가 검증한다.
- 공개 Train/Dev 프롬프트 자체: `colab-label/build_public_all.py` (규정의 "정확한 프롬프트나
  프롬프트 해시를 사용하는 공개 자료 조회" 허용 범위).
- 채점 기준: `colab-label/judge.py`가 공개 업스트림의 정답으로 채점. `data/gold/gold-answers.v1.json`
  은 SKT 배포물이 아니라 공개 업스트림에서 로컬로 재구성한 매칭 산출물이며, 이미지에는 그
  텍스트가 아닌 수치 결과만 들어간다.

## 변환 기록과 SHA-256

라벨(JSONL) → 컬럼(JSON) 변환은 `colab-label/to_prior_labels.py`(컬럼 D는 `--length-only`)와
`tools/splice_prior_column.py`(내부적으로 `tools/build_prior_lookup.py`의 `_load_column`,
소수 4자리 반올림)로 수행했다.

| 파일 | SHA-256 |
|---|---|
| `colab-label/prior_column_c.json` (3.5 MB) | `b18956ca6e86b4c86c2b568b9bf70d0c3667b2be3c51ad6b825de065620479e5` |
| `colab-label/prior_column_d_reason.json` (0.2 MB) | `34ca7b76420418aaf0ae7ec78c6f143745ddb27764c2e139424dde3168ca647c` |
| `src/ossp_router/resources/learned-router-submission.v1.json` (27.8 MB) | `7984081c57f2e9a97725b8378aa2b5a405775079c7ec8eac41874f5c04ec0450` |
| `src/ossp_router/resources/learned-router.v1.json` (25.1 MB) | `3360ba0dbe5243b421b8f977408a57cdd2963c60701341b7dca089e0f35e6f0e` |

## 런타임에서의 사용

- 특징: `prior_features()`가 컬럼별 11특징 + 인접 컬럼 델타를 메타 GBM 입력에 공급 (miss는 전부 0).
- 결정층 혼합(E69): 아티팩트 필드 `prior_score_blend` `{weight: 0.25, columns: {ax31-light: axlight-q6k-v2, ax31: ax31-34b-v1}}` —
  채점된 조회 적중 시 최종 점수행에 컬럼 실측 점수를 가중 혼합. 순수 표준 라이브러리(sha256 1회 + dict 조회).
- 공개 조회표(`public_lookup`)는 이 blend 필드 설정 **이후에** 생성되었고, 빌드 기계에서
  계산 경로와의 동일성(120문항 × 3 tier, 최대 오차 0)을 `tools/verify_lookup_consistency.py`로 검증했다.
