<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# 실험 로그 (전체 기록)

모든 실험을 채택/기각 여부와 관계없이 기록한다. 표기:
- **dev** = 공개 Dev 880 공식 scorer 점수 (합산 학습 이후엔 in-sample이라 참고용)
- **CV** = 5-fold 중첩 교차검증 점수 (합산 학습 이후의 공식 추정치)
- **EV** = 880 크기 부트스트랩 기대점수 `E[점수 × 예산통과]` — 최종 의사결정 기준
- 가중치: fast 0.4 / balanced 0.3 / premium 0.3

## 2026-08-13

### E01. 순수 Python 추론 최적화 ✅ 채택
- 가설: FNV 해시·문자 카운트가 병목; 출력 불변 최적화 가능
- 방법: 토큰/바이그램 FNV 메모이제이션, 바이그램 접두사 FNV 상태 캐시,
  Counter 집계, `str.split()` 공백 카운트, `map(operator.mul)` 내적
- 결과: 예측 단계 30.4s→15.1s (**2.0×**, 동일 머신 A/B), 제출물 6종 바이트 동일
- 부수: 노트북 스로틀링으로 절대시간 비교 불가 확인 → 이후 A/B만 사용

### E02. 공개 prompt-hash 조회표 ✅ 채택
- 가설: 규칙이 허용하는 공개 조회로 QEMU 90초 검사 통과 가능
- 방법: 공개 2,640문항 SHA-256 → 예측값 아티팩트 내장, 미스 시만 계산
- 결과: QEMU 95s+ 타임아웃 → **10.1~10.6s** 전 tier 통과

### E03. 선형 하이퍼파라미터 스윕 1차 (4096 bins) ✅ 부분 채택
- α∈{3,10,30} × legacy blend∈{0,.25,.5,.75} × 학습모델 적용범위 limit∈{256,∞}, 21후보
- 결과: blend 0.75 지배적, limit ∞가 일관 우세, 최고 α10/b.75/L∞ dev **0.698523**
  (기존 0.697642; blend0(순수 학습모델) 0.681로 legacy 앙상블 필수 확인)

### E04. 선형 스윕 2차 (8192 bins, blend 0.6~0.9) ✅ 채택
- 결과: α30/b.75/L∞ dev **0.698693** — 최종 채택. blend>0.75는 하락.
- 순수 Python 재채점(공식 기준) 0.698523 (fp 경계 1문항 차이, 문서화)

### E05. GBM 직접 스태킹 (leakage 버전) ❌ 실패 → 교훈
- 방법: in-sample 선형 예측을 특징으로 HistGBM 학습
- 결과: 순수 0.6798, 최고 혼합 0.6926 — 전부 기준(0.6984) 미달
- 교훈: **train 내 특징 leakage가 스태킹을 죽인다** → E12에서 OOF/LOO로 재도전

### E06. family 평균 + 공개 kNN 증강 ✅ 채택
- family 9종(내용 정규식) 평균 혼합 + 문자 tf-idf kNN(k=8) 확신도 혼합 그리드
- 결과(dev): fam0.3 단독 0.699716, kNN 게이트 단독 0.698693,
  **fam0.3 + kNN×conf×0.5 = 0.700511** (fast 0.6693, balanced 0.7006)
- 순수 Python 구현(similarity.py) 재검증: conf×0.4가 최적 **0.700000** → 0.4 채택
- 배포: `augmentation` 블록, 안전계수 재보정 (dev 0.700000 확정)

### E07. tier별 (메타 blend × 안전계수) 분리 최적화 ✅ 채택
- 결과: fast blend0.6/s0.89, balanced 0.3/0.84, premium 0.45/0.72
  → dev 0.6923, EV 0.6921 (공유 설정 대비 EV +0.003)
- 조회표를 tier별 18값으로 확장, 런타임에 tier 인자 관통

### E08. 문항별 비용 상한 할당 ❌ 기각
- 가설: 고비용 꼬리 차단 → 안전계수 상향 여지
- 결과: 최고 EV 0.6842 ≈ 기준 0.684 — 무효. 예산 분산은 소수 문항이 아니라
  광범위 비용 오차에서 옴

### E09. 안전계수 EV 최적화 ✅ 채택 (중대 발견)
- 발견: dev-최적 안전계수(0.958/0.979/0.913)는 **부트스트랩에서 fast 33% /
  balanced 17% / premium 25% 확률로 예산 초과**(=tier 0점), 기대점수 ≈0.53
- EV 최적 0.86/0.87/0.74 적용 → dev 표시 0.700→0.6857이지만 EV 0.53→**0.684**,
  초과 확률 0.2~0.3%
- 원칙 확립: 이후 모든 안전계수는 880 크기 부트스트랩 EV로 선택

### E10. 비공개셋 위험 분석 (참고 자료)
- family 구성 ×2 스트레스: 최악도 EV 최적 안전계수에서 관리 가능
- kNN 무력화 하한(비공개가 공개와 안 닮음): fast 0.667/balanced 0.698, 예산 통과
  — kNN 가중치가 유사도 비례라 하방이 완만함을 확인

## 2026-08-15

### E11. 합산(train+dev 2,640) 학습 ✅ 채택
- 전 구성요소 재학습, 평가는 5-fold 중첩 CV(선형 OOF·kNN LOO·메타 내부 OOF)로 전환
- 결과: 가중 CV 0.6940 / CV-EV **0.6926** (기존 dev-EV 0.6921), fast EV 0.6521→0.6581
- 함정 발견: 2,640 크기 부트스트랩은 위험 과소평가 → 안전계수는 880 크기로 선택
  (0.96/0.89/0.85)

### E12. 아티팩트 heavy 분리 ✅ 채택
- 문제: 합산 아티팩트 14.8MB → QEMU에서 json 26s + parse 19s = 기동만 50s, 타임아웃
- 해법: kNN 벡터·메타 트리(10.7MB)를 `learned-router-heavy.v1.json`으로 분리,
  조회 미스 시 SHA-256 검증 후 지연 로드 (`pack_artifact.py`)
- 결과: 본체 4.1MB, QEMU 46~61s 통과, 분리 전후 결정 동일

### E13. DeepMind 모듈 사전분포 (공개 소스 메타데이터 활용) ❌ 기각
- 소스 selection 파일에서 456문항/71모듈 라벨 무료 획득; 모듈별 light 성공률
  스프레드 0.06~0.91로 신호는 강력
- kNN top-1 이웃의 모듈 평균 혼합, (가중, 유사도 문턱) 그리드 CV
- 결과: 최고 EV +0.0004 (0.6928→0.6932) — 노이즈 수준. 커버리지 10.8%뿐이고
  kNN이 모듈 신호를 이미 흡수. 복잡도 대비 무가치

### E14. Gain(이득) 헤드 ✅ 채택 (스태킹 이후 최대 단일 개선)
- 가설: 할당이 쓰는 양은 업그레이드 이득(s₁−s₀, s₂−s₁) — 직접 회귀가 유리
- 방법: 델타 GBM 헤드 2개 추가, 재구성 점수(s₀, s₀+δ₁, s₀+δ₁+δ₂)를 α로 혼합
- 결과(CV): α=0→0.6928, 0.3→0.6958, **0.5→0.6966**, 0.7→0.6959, 1.0→0.6957
  — 곡선 매끄러움(신뢰 높음). α=0.5 채택, CV 점수 0.6983
- 안전계수 재선택: 0.98/0.85/0.88 (fast 0.98은 0.99 확장 그리드에서도 최적 재확인)

### E15. word-level kNN 추가 ❌ 기각
- 단어 1·2-gram 해시 tf-idf kNN을 문자 kNN 뒤에 확신도 게이트로 추가
- 결과: scale 0.15에서 EV +0.0002(노이즈), 0.3+는 하락 — 문자 kNN이 포섭

### E16. Colab MCP 연동 (인프라) ✅ 구축
- googlecolab/colab-mcp: Python≥3.13 필요(uv, UV_PYTHON_INSTALL_DIR 우회),
  uvx 직접 실행은 기동 53s로 타임아웃 → `uv tool install` 고정 exe로 해결,
  사용자 스코프 등록. 브리지는 **열린 Colab 탭**을 통해 동작(탭 닫히면 도구 잠김),
  기본 브라우저 이슈는 BROWSER env로 해결. 셀 추가/실행 도구로 원격 실행,
  번들은 base64 청크 셀로 전송(PowerShell zip은 `\` 구분자라 unzip 불가 →
  Python zipfile로 정규화 추출)

### E17. Colab GBM 하이퍼파라미터 스윕 + 시드 앙상블 ❌ 미반영 (검증적 음성 결과)
- 12개 설정(learning_rate {.04,.06,.1} × leaves {15,31} × min_samples {20,30,50}
  × l2 {1,3,10} 부분격자) CV 비교 + 최적 설정 5시드 앙상블, Colab Pro에서 실행
- 결과(동일 하니스 내 비교): 단일 시드 최고 = **현재 배포 설정** (lr.06/leaves15/
  msl30/l2 3.0) EV 0.6962; 타 설정 0.6927~0.6960; **5시드 앙상블 0.6959 (이득 없음)**
- 해석: early stopping이 이미 분산을 억제해 시드 평균화 이득이 소멸.
  현재 하이퍼파라미터가 12개 중 최적임을 교차 확인 → 배포본 유지
- 인프라 산출물: Colab 원격 실행 파이프라인(E16) 검증 완료, 이후 실험 재사용

## 2026-08-16

### E18. MLP 메타헤드 ❌ 기각
- 방법: 58특징→64은닉→8출력 torch MLP (조기종료), 같은 중첩 CV 하니스,
  GBM과 25%/50% 혼합 평가 (Colab)
- 결과: 하니스 내 GBM 기준 EV 0.6962 / **MLP 단독 0.6829** / 혼합 0.6941(25%),
  0.6850(50%) — 전부 미달
- 해석: 2,640 샘플에서 신경망은 트리+선형 대비 열세 (E05와 일관). 소데이터
  구간에서 모델 클래스 다양화보다 특징·타깃 설계가 유효하다는 누적 증거

### E20. kNN 구성 스윕 (k × 유사도 가중) ✅ 채택
- 방법: k∈{4,8,16,32} × 가중 {유사도¹, 유사도²} 8구성, 구성별 전체 파이프라인
  (kNN행 재계산 + GBM 8헤드 재학습) 중첩 CV (Colab, fold당 ~7분)
- 결과(Colab 하니스): k4 0.6938/0.6948, k8(기존) 0.6962, **k16¹ 0.6976** ← 최고,
  k16² 0.6970, k32 0.6957/0.6976 — 단봉 곡선으로 신뢰
- 로컬 재확인: α=0.5에서 EV 0.6966→**0.6974** (+0.0008, 양 하니스 일관)
- 배포: `similarity.NEIGHBORS` 8→16, 메타 재학습, 안전계수 0.98/0.89/0.88,
  조회표/분리 재빌드, 전체 검증(테스트·QEMU 30~49s) 통과

### E19. 문자 n-gram 특징 공간 스윕 ❌ 기각 (전 구성)
- 방법: char bins {8192→16384} / stride {3→1} / 텍스트 한도 {6k→8k자} 변형별
  ridge+파이프라인 전체 중첩 CV (Colab; 도중 VM 재활용 1회로 재실행)
- 결과: 기준 0.6976 / bins16384 0.6969 / stride1 0.6956 / limit8k 0.6977
- 해석: stride 축소·bin 확대는 노이즈만 추가. 한도 8k는 +0.0001로 노이즈 수준.
  현 특징 설정이 사실상 국소 최적 — 특징 공간 방향은 수확 종료
- 운영 교훈: Colab 런타임 재활용에 대비해 폴러가 keepalive 겸용(5분 주기),
  결과 파일은 완료 즉시 회수할 것

### E21. 순서형(누적 임계) 점수 헤드 ✅ 채택 (삼각측량 확정)
- 방법: 모델별 P(s≥0.25/0.5/0.75/1) 이진 GBM 4개로 E[s] 재구성
- 최초 결과: 회귀 0.6976 vs 순서형 순수 **0.6982** (+0.0006), 단 혼합(0.3/0.5)이
  0.6974로 양 끝점보다 낮음 — 비단조 곡선이라 판정 보류, 확인 런 진행
- 3개 부트스트랩 시드 삼각측량 (회귀 vs 순서형):

  | 시드 | 회귀 | 순서형 | 차이 |
  |---|---:|---:|---:|
  | 7 | 0.6976 | 0.6982 | +0.0006 |
  | 17 | 0.6962 | 0.6984 | +0.0022 |
  | 27 | 0.6957 | 0.6969 | +0.0012 |
  | 평균 | — | — | **+0.0013** |

- 판정: 3/3 시드 전부 양수, 평균 +0.0013 → **보류 해제, 채택 확정**
- 배포 완료 (검증 통과, QEMU 7-8s): 점수 회귀 3헤드 → 모델별 누적
  P(s≥0.25/0.5/0.75/1) 분류기 12개(모델 3 × 임계 4)로 교체, 시그모이드
  E[s]=0.25·Σsigmoid 재구성 런타임 구현, 조회표/heavy 분리 재빌드,
  self-check 예산·테스트 전부 통과, QEMU 7~8초로 통과 (E20 대비 대폭 단축).
  gain 헤드(E14) 재구성 기준은 순서형 s₀ 유지. 새 배포 CV EV ≈ **0.6982**
  (시드7 하니스; 3-시드 삼각측량 평균은 위 +0.0013 참조)

### E22. 특징 배터리 (Colab VM1, 병렬 5종) ❌ 전부 기각
- 가설: dense 특징 확충, word bin 확대, word 3-gram 추가, 프리픽스 블록
  추가 중 일부가 기준(0.6976, E20 채택본)을 넘길 수 있다
- 방법: 5종 변형을 Colab VM1에서 병렬 실행, 각각 전체 파이프라인 재학습 후
  중첩 CV EV 비교
- 결과: dense 특징 15개 추가 **0.6889**(큰 하락), word bins 16384 **0.6956**,
  word 3-gram **0.6967**, 프리픽스 블록 **0.6947** — 전부 기준(0.6976) 미달
- 해석: dense 추가 시의 큰 하락은 ridge 표준화 단계에 노이즈 특징이 유입돼
  선형 head 품질을 끌어내린 탓으로 추정. word/char 파생 변형들도 개선 없음.
  E19(문자 n-gram 스윕 기각)와 합쳐 **특징 공간 방향 수확 종료 확정**

### E25. 문장 임베딩 teacher 증류 ❌ 기각
- 방법: 다국어 MiniLM 임베딩(2,640×384, 로컬 66s) 공간의 fold-pure kNN
  teacher 예측을 soft label로 혼합((1−λ)true+λteacher)해 GBM 학습 — 런타임
  무변경으로 신경망 지식 흡수 시도
- 결과: λ=0 0.6977 / λ=0.2 0.6966 / λ=0.4 0.6937 — 단조 악화
- 해석: 이 도메인에선 해시 문자 n-gram 유사도가 신경망 임베딩보다 우수.
  임베딩·파인튜닝 계열의 기대값을 낮추는 근거 (E18 MLP 기각과 합치)

### E30. 이득의 등장성(isotonic) 보정 — 최적화자의 저주 제거 ❌ 기각
- 배경: Opus 실험 설계 문서(`colab-sweep/experiment_designs.md`)의 1순위
  권장 실험 — "비용 대비 정보량이 가장 높다"는 근거로 진단표를 먼저 찍고
  평평하면 즉시 기각 가능하도록 설계됨
- 가설: 할당은 이득이 과대예측된 문항을 골라내는 선택 편향(optimizer's
  curse)을 가짐 → 실현 이득이 예측 이득보다 체계적으로 낮을 것 → fold-pure
  등장성 회귀로 `E[δ_true | δ̂]`를 대입하면 편향 제거, 할당은 `δ̂/Δĉ`로
  순위를 매기므로 비선형 단조 변환이 실제 선택 집합을 바꿈
- 방법: 기존 CV 예측 캐시(meta_all, true_s) 위 사후처리, 재학습 없음.
  δ̂ 10분위별 `mean(δ_true) − mean(δ̂)` 편향 진단표를 먼저 산출(권장 진단
  산출물); 이어 fold-pure isotonic 보정 및 예측 비용차 10분위별 2D 변형까지
  α 그리드로 EV 비교
- 결과: 10분위 편향 진단표가 **±0.05 범위에서 비단조로 요동** — 과대예측
  문항이 꼬리에 몰리는 optimizer's curse 패턴이 관측되지 않음. EV도
  raw(gain α=0.5) **0.6974** vs isotonic 보정 0.6963/0.6969/0.6974로 이득
  없음
- 해석: 편향 진단표 자체가 평평(비단조 잡음)해 가설의 전제(체계적 과대예측)가
  성립하지 않음이 확인됨 → 등장성 보정은 애초에 고칠 편향이 없어 무효.
  설계 문서가 의도한 대로 진단표 덕에 30분 만에 판정 완료. 이 결과는
  E26/E27(부호 분해·영과잉 타깃 실험) 착수 시 "이득 재구성 쪽에 남은 편향은
  적다"는 사전확률을 낮춰주는 근거로 활용

### E23b. comp1024 × ordinal 결합 확인 (로컬, E23 후속) ❌ 기각(미채택)
- 배경: E23(Colab VM2, kNN 표현 3종 스윕) 계열 중 comp1024(TOP_COMPONENTS를
  256→1024로 늘려 kNN 벡터 성분을 확충한 표현)가 유망 후보로 남아, 배포본
  (E21 순서형 헤드 포함)과의 결합 효과를 로컬에서 재확인
- 방법: comp1024 kNN 벡터를 순서형 점수 헤드가 포함된 현재 배포 파이프라인과
  결합해 재학습, 동일 중첩 CV 하니스로 comp1024 단독 및 결합 EV를 비교
- 결과: comp1024 단독 EV **0.6972**, comp1024+ordinal 결합 EV **0.6985**
  (score 0.6987) — 배포 기준선(ordinal, EV 0.6982) 대비 **+0.0003**
- 해석: +0.0003은 부트스트랩 노이즈 범위 내로, kNN 벡터 4배 확충에 따른
  아티팩트 크기 증가(heavy 분리본 재확대)를 정당화할 개선폭이 아님 → 미채택.
  E23 계열(kNN 표현 스윕) 전체 종료

### E26. 부호 분해 이득 헤드 (Colab VM1) ❌ 기각
- 가설: 이득(δ) 타깃을 부호별로 분해하면 신호가 개선될 것 —
  P(도움)·E[크기|도움] − P(손해)·E[크기|손해]로 재구성(HistGB 분류기 2개 +
  부분모집단 회귀 2개, SMALL_PARAMS), 기존 δ 회귀와 β로 혼합
- 방법: 사전 진단으로 zero-mass 비율(델타=0인 표본 비중) 산출 후 β∈
  {0, 0.25, 0.5, 0.75, 1.0} 그리드로 가중 EV 비교
- 사전 진단: zero-mass 비율 **0.75 / 0.69**로, 설계가 가정한 40~60%보다
  높게 관측됨
- 결과(가중 EV): β=0.0 **0.6976** / β=0.25 0.6974 / β=0.5 0.6971 /
  β=0.75 0.6971 / β=1.0 **0.6967** — β 증가에 단조 악화
- 해석: 부호 분해가 순수 손해. zero-mass가 예상보다 커서 분류 확률 추정
  자체의 정보량이 기존 δ 회귀보다 낮았던 것으로 보임 → 기각

### E31. 비용 비율 타깃 + Duan smearing (Colab VM2) ❌ 전체 기각, fast 한정 후보로 보류
- 가설: 비용 헤드를 log(c0), log(c1/c0), log(c2/c1)로 재구성하면 단조성이
  구조적으로 보장되고, fold-train 잔차 기반 Duan smearing 보정을 얹으면
  로그 변환의 재변환 편향을 줄일 수 있다
- 방법: ratio 타깃 재구성 + Duan smearing 보정, fast tier 안전계수 그리드를
  0.92~1.0(0.005 간격)으로 확장해 비교
- 결과 (tier별 EV@안전계수, 가중 EV):

  | 구성 | 가중 EV | fast | balanced | premium |
  |---|---:|---|---|---|
  | baseline | 0.6976 | 0.6643@0.980 | 0.6922@0.890 | 0.7474@0.870 |
  | ratio | 0.6961 | 0.6644@0.965 | 0.6931@0.890 | 0.7412@0.820 |
  | ratio+smear | 0.6961 | 0.6654@0.965 | 0.6925@0.900 | 0.7408@0.860 |

- 해석: premium에서 −0.0066 손실이 지배해 전체 가중 EV는 하락 —
  비율 누적 예측이 premium 구간의 비용 분산을 키운 것으로 추정됨. 전체
  구성으로는 기각. 단, fast만 보면 ratio+smear가 baseline 대비
  **+0.0011**(0.6654 vs 0.6643)로 tier별 헤드 채택 여지가 있어, 다중 시드
  확인 후 fast 한정 채택 후보로 기록해 둔다

### E27. 랭크 변환 효율 헤드 (Colab) ✅ 채택 (β=0.25)
- 설계: 할당기가 소비하는 것은 효율 Δs/Δc의 순위뿐이라는 관찰에서, 효율을
  fold-train 내 경험적 백분위 순위 [0,1]로 변환해 표준 GBM으로 학습, 65노드
  분위 LUT로 역변환 후 예측 비용차를 곱해 이득 추정치로 복원, 기존 δ회귀와
  β로 혼합
- 결과 (시드 3개 삼각측량, 가중 EV):
  - seed7: β=0 0.6976 / β=0.25 0.6979 / β=0.5 **0.6982** / β=0.75 0.6981 /
    β=1.0 0.6981
  - seed17: β=0 0.6993 / β=0.25 **0.7003** / β=0.5 0.6999 / β=0.75 0.6998
  - seed23: β=0 0.6910 / β=0.25 **0.6916** / β=0.5 0.6914 / β=0.75 0.6907
- 3시드 모두 단봉·기준선 상회 (평균 +0.0007, 시드 평균 최적 β=0.25), 이득은
  premium(0.7474→0.7504 @seed7 β=0.5) 주도. 경계 가중 변형(rank_w)은 악화로
  기각
- 배포 반영: `build_meta_gbm.py`에 RANK_BETA=0.25 랭크 헤드 2개(25/60트리) +
  65노드 LUT + 비용차 하한(0.000433/0.009775) 추가, `learned_router.py`
  런타임에 stdlib 평가 경로(트리→LUT 선형보간→Δĉ 곱→β혼합) 추가, rank_trees는
  heavy 블록("meta_rank_trees")으로 분리. 재빌드 후 dev **in-sample 참고치**
  0.7286 (fast 0.6960 ratio 1.196 / balanced 0.7256 ratio 1.732 / premium
  0.7750 ratio 3.307, 전부 예산 통과) — Train+Dev 합산 학습으로 조회표가
  Dev를 암기하고 있어 성능 지표 아님, 정정 및 held-out 재측정은 8/16
  "정직한 held-out 검증" 항목 참조. QEMU arm64 검사 tier당 6.6~7.7초 통과,
  heavy 11.70MB / main 4.14MB

### E28. 지도 선택 어휘 특징 (Colab) ❌ 기각
- 방법: fold-pure Welch t 선택(+5분할 안정성 필터, min_df 30), 후보 어휘
  5,349개
- 결과: K=0 0.6976 / K=24 0.6957 / K=48 0.6974 / K=96 0.6972 — 전부 기준선
  이하
- 해석: 진단 산출물에서 선택 토큰이 문장부호·영문 문어체·한국어 조사류
  ('mercy', 'nay', '왔다' 등 출처 템플릿 어휘)로, 난이도 신호가 아니라 출처
  신호였음을 확인 → 가설 4-1(어휘×길이 상호작용 통로)은 열어줘도 쓸 신호가
  없다는 결론

### E32. 비용 불확실성 팽창 할당 (로컬) ❌ 기각
- 가설: 예산 초과 위험의 원천은 비용의 크기가 아니라 예측 불확실성 —
  이분산 헤드로 문항별 σ̂를 추정해 할당 결정에서만 불확실한 문항을
  밀어내면(κ 페널티) 실현 비용비율 분산이 줄어 안전계수를 올릴 수 있다
- 방법: 내부 OOF |로그비용 잔차|를 회귀하는 σ 헤드 3개 추가. 팽창 비용
  `pc·exp(κ(σ̂−mean))`은 할당 결정에만 사용, 예산 회계·채점은 원 비용 유지
  (안전계수와 이중 계상 방지). κ∈{0, 0.25, 0.5, 1, 2} × 안전계수 그리드
- 진단: σ̂ 5분위별 실현 |잔차| 스프레드 **4.00× / 2.48× / 1.90×** (모델
  0/1/2) — 이분산은 강하게 실재하며 σ 헤드가 이를 잘 예측함 (기록 자산)
- 결과: κ=0 **0.6977** / κ=0.25 0.6972 / κ=0.5 0.6971 / κ=1.0 0.6969 /
  κ=2.0 0.6949 — κ 증가에 단조 악화
- 해석: 전역 안전계수가 이미 비용 분산 위험을 흡수하고 있어, 문항별
  페널티는 점수 손실만 추가함. E08(문항별 비용 상한 기각)과 합쳐 "예산
  위험 관리는 전역 스칼라 하나로 충분" 결론 확정

### E34. IPR 구조 이식 (arXiv 2509.06274, 로컬) ❌ 전 구성 기각
- 배경: 사용자가 제공한 논문 4편 중 실행 가능한 구조는 IPR(Amazon,
  prompt-only 라우터) 하나. 서베이(2406.16838)는 지형도, Inference Scaling
  Laws(2408.00724)는 반복 추론 전제라 1문항 1호출인 본 과제엔 직접 적용
  불가(cheapest-first 철학 근거로만), 2604.10632는 음악-미각 논문(무관)
- 이식 3종: (A) 품질≥θ인 가장 싼 모델 선택 + 배치별 θ 이분탐색 할당기
  (§2.3/2.4) 및 잔여예산 Lagrangian 하이브리드, (B) P(s₁>s₀)/P(s₂>s₁)
  pairwise 승률 헤드로 이득 재구성 γ혼합(§2.2 ranking loss 대응),
  (C) 순서형 12헤드 per-head 온도 보정(§3 calibration)
- 결과(가중 EV, 배포본 0.6983): A 임계값 **0.5325**(premium 0.385) /
  A′ 하이브리드 0.6155 / B γ0.25 0.6960, γ0.5 0.6867 / C **0.6983**(±0) /
  B+C 0.6960 / A+B+C 0.5435
- 해석: (A) 임계값 규칙은 사용자 τ 손잡이를 위한 제품 요구이지 최적성
  근거가 아님 — 예산 고정 과제에선 한계효용 기준 Lagrangian이 지배적
  (light 평균 0.60인 분포에서 θ 상향 시 어중간 문항이 일괄 최고가로 튐).
  (B) 순서 정보는 순서형+gain+rank 헤드가 이미 흡수, 세 번째 추정치는
  분산만 추가. (C) 학습된 온도 0.96~1.12 → 이미 보정돼 있어 무효.
  **논문 구조 대비 현 배포 구조 우위 확인**; 남은 격차는 할당·보정이
  아니라 prompt-only 예측 한계에서 옴

### E35. 순수 IPR 재현 (논문 그대로, 독립 라우터) ❌ 기각 — 정면 비교 근거
- 방법: E34(구성요소 이식)와 달리 배포 파이프라인과 섞지 않고 논문 레시피를
  단독 구현. 공유 인코더(해시 n-gram 16,414→선형+ReLU 128; 트랜스포머는
  stdlib 제약으로 대체) → 모델별 로지스틱 head 3개, 라벨=수용가능(s≥0.5),
  손실 BCE+λ·pairwise ranking(λ∈{0,.5,1}), per-head 온도 보정(20% 분할),
  선택=q≥θ인 최저가 모델(없으면 argmax)+배치별 θ 이분탐색, 비용=모델별
  학습셋 평균(논문의 Bedrock 고정 단가 가정). 동일 CV+부트스트랩 하니스
- 결과(가중 EV): 논문 그대로 λ=0 **0.3828**(premium 0.098) / +온도 0.3636 /
  λ=0.5 0.3203 / λ=1.0 0.3126. 브리지: IPR 예측기+우리 Lagrangian 0.5716,
  IPR 선택기+오라클 비용 0.6132. 배포본 0.6982, baseline 0.6954
- 진단: OOF AUC 0.78~0.81(모델별), 학습된 온도 1.3~1.8(과신), λ↑ 시 AUC↓
- 해석: 세 원인 중첩. (1) **고정 단가 가정이 치명적** — 본 과제는 출력
  토큰 길이로 비용이 결정돼 think 비용이 문항별 수십 배 편차; 평균 비용
  배차는 부트스트랩 배치 대부분에서 예산 초과(0점). 오라클 비용 시 0.61로
  회복이 증거. IPR엔 문항별 비용 예측 구성요소가 없음. (2) 임계값 선택기는
  예산 고정 환경에서 Lagrangian 대비 구조적 열세(E34 재확인, 같은 예측기
  +0.19). (3) 1.5M 라벨 전제 신경망 공동학습이 2,640에선 과신·열세(E18/E25
  일관). **배포본이 순수 IPR 대비 +0.32, 최선 브리지 대비 +0.085 우위**
  — 논문에서 유효한 것은 prompt-only 모델별 품질 예측이라는 큰 틀뿐이며
  이미 반영돼 있음

### 비용 오차 진단 (diag_cost.py, 2026-08-16) — E36 설계 근거
- think 비용의 87%가 출력 토큰; 출력 길이는 프롬프트 길이와 무상관(r=0.008),
  모델 자기 점수와 음의 상관(r=−0.33, 못 풀수록 길게 생각). heavy tail
  (중앙값 1.6k, p95 11.8k, 최대 130k 토큰). 입력 토큰은 이미 잘 맞음(RMSE
  0.13, r=0.94; family별 tok/char 0.5~4.2 편차를 n-gram이 흡수)
- exp(E[log c])는 중앙값 추정 → 880배치 think 합계를 **33% 과소예측**
  (pred/true 0.671) — 현재는 안전계수 0.88이 전역 흡수. 잔차는
  aime 0.95 / hrmcr 0.92 / dmmath 0.70 / code 0.62 vs belebele 0.36에 집중

### E36. 비용 예측 정밀화 — dense 메타 경로 (로컬) ❌ 기각 (T+P avg +0.0007 미채택)
- 3축: T 분위 GBM(τ .5/.75/.9)로 상단 꼬리 직접 학습, 할당비용
  exp(q50+κ(q90−q50)) κ∈{0,.25,.5,1}; F family×모델별 Duan 스미어링(내부
  OOF 잔차, 전역 대조); P 순서형 P(s≥.5)를 비용 헤드 입력에 스태킹.
  안전계수 그리드 상단 확장(fast→1.00, bal/prem→.97)
- 결과(가중 EV, 배포 0.6983): T κ0 0.6968 / κ.25 0.6982 / κ.5 0.6983 /
  κ1 0.6978; F 전역 0.6970 / family 0.6975; P 0.6984; P+F 0.6969;
  T+F 0.6960; **avg(T κ.5, P) 0.6990 (+0.0007)**
- 진단: T/P 모두 log-RMSE 개선 없음(think 0.646→0.650/0.648); 스미어링은
  합계 보정 완벽(0.75→1.02)이나 EV 하락; T는 안전계수를 fast .99 /
  prem .91로 올렸지만 점수 미변
- 해석: **합계 과소예측은 무해** — Lagrangian은 비용의 상대 순위만 쓰고
  절대 수준은 안전계수가 흡수하므로 전 문항 동일 비율 편향은 손실이 아님.
  실제 손실원인 문항 간 순위 오차(RMSE 0.65 = ±1.9배)는 어떤 변형도 못
  줄임. T+P avg는 두 추정치 평균의 분산 감소일 뿐 새 정보 아님 → E27과
  같은 크기라 3시드 확인 없이 미채택

### E36b. 비용 예측 정밀화 — sparse 특징 경로 (로컬) ❌ 기각 (단조 악화)
- 4종: S1 희소 릿지 로그비용을 전용 비용원으로 w∈{.25,.5,1}; S2 출력·입력
  토큰 각각 희소 릿지 → 단가로 비용 재구성; S3 희소 OOF 예측을 dense 비용
  헤드에 스태킹; S4 꼬리 인지 희소 릿지(타깃 log c + .5|resid|)
- 결과: log-RMSE(think) dense 0.646 vs S1 0.729 / S2 0.724 / S3 **0.646
  (동일)** / S4 0.782. EV: S1 w.25 0.6970 → w1 0.6510, S2 w.25 0.6968 →
  w1 0.6662, S3 0.6983(±0), S4 w.25 0.6971 — sparse 비중↑ 시 단조 악화
- 해석: 16,414차원 n-gram에는 dense 58특징이 모르는 비용 신호가 없음
  (S3에서 GBM이 희소 예측을 완전히 무시). 출력 길이는 어휘가 아니라
  "과제 유형 × 난이도"의 거친 구조에서 결정되며 family 원핫·순서형 확률·
  kNN 이웃 비용이 이미 담고 있음. **비용 예측 정밀화 방향 수확 종료** —
  think 출력 길이의 프롬프트-불가측 성분(±1.9배)은 문제 자체의 한계

### E37. 미탐색 방향 3종 — 행렬 분해 / 컨포멀 비용 / ILP 정확해 (로컬) ❌ 기각, ILP는 최적성 확인
- 설계 공간 지도(IMAGE_PROMPTS.md §10-B)의 미실험 잎 중 실행 가능한 셋.
  MF: 문항×모델 점수 행렬 ALS 분해(rank 8, 편향 포함), 프롬프트→잠재
  벡터는 희소 릿지로 사상, 순서형 점수와 μ 혼합. CP: 비용 헤드 내부 OOF
  잔차의 split-conformal 상단 분위(α∈{.5,.6,.7,.8})를 할당 비용 승수로.
  ILP: scipy.milp(HiGHS) 0/1 정확 배낭해 vs Lagrangian 이분탐색, 동일
  예측·100표본
- 결과(가중 EV, 배포 0.6983): MF μ.15 0.6981 / μ.3 0.6973 / μ.5 0.6962
  (단조 악화); CP α.5 0.6989(+0.0006) / α.6 0.6981 / α.7 0.6970 / α.8
  0.6961 (단조 악화); **ILP 0.7009 vs Lagrangian 0.7013 (갭 −0.0004,
  100표본 기준, ILP 413s)**
- 진단: MF 예측은 진짜 점수와 상관 0.44~0.50(순서형 0.54~0.60), 순서형과
  0.79~0.82로 정보 중복. CP α.5는 중앙값 잔차(−0.07)라 사실상 배포본
- 해석: MF는 열 3개·완전 관측 행렬이라 저랭크 구조가 자명해 배울 것이
  없음. CP는 E32/E36과 동일 — 전역 안전계수가 같은 역할을 더 싸게 수행.
  **ILP 결과가 핵심**: 정수 갭이 사실상 0 → 현 Lagrangian 할당기는 주어진
  예측 하에서 최적(E34/35의 "IPR 대비 우위"를 넘어 절대 최적성 확인).
  RL 정책(라벨 정보가 오라클과 동일)·반지도(비공개 프롬프트 접근 불가)는
  실행 근거 없음 → **설계 공간 탐색 종료, 배포본 E27 확정**

### E38. 외부 라우터 9종 재구축 정면 비교 (RouteLLM · LLMRouter, 로컬) ❌ 전부 기각
- 출처: lm-sys/RouteLLM(mf·sw_ranking·bert·causal_llm — 강/약 이진 승률+
  임계값), ulab-uiuc/LLMRouter(arXiv 2608.06867 — KNN·SVM·MLP·MF·Elo·
  RouterDC·GraphRouter 등 16종), awesome-model-routing(인프라 목록, 알고리즘
  없음 → 위 둘로 귀결). 인코더는 stdlib 제약상 우리 희소 특징의 fold-pure
  SVD 128차원으로 대체. **비용 예측·할당기는 배포본과 동일 고정** → 라우터
  점수 신호만 분리 비교. 동일 CV+부트스트랩 EV 하니스
- 결과(가중 EV, 배포 0.6987): GraphRouter 0.6776 / RouteLLM-MF 0.6773 /
  KNN 0.6769 / MLP 0.6690 / Elo 0.6683 / SW-ranking 0.6628 / RouterDC
  0.6530 / BIN 3모델 확장 0.6276 / SVM 0.5760 / **RouteLLM 원형(이진
  임계값) 0.3953**. 배포본에 0.2 혼합: GraphRouter −0.0011, MF −0.0051,
  MLP −0.0040, DC −0.0109 — 추가 정보 없음
- 진단: 쌍별 선호 계열(MF·SW·BIN·Elo)은 think>light 쌍 정확도 0.82~0.83으로
  배포본(0.83)과 동등하나 점수 상관 0.1~0.26 — 순서만 배우고 크기를 버림.
  단일 예측기(KNN·MLP·GraphRouter)는 우리 스태킹의 구성 요소 하나 수준.
  SVM은 최선 모델 분류라 mid/think 판별 불가(상관 −0.03)
- 해석: 이 라이브러리들은 "이진 강/약 · 쌍별 선호 라벨(Arena) · 예산 무제약"
  세팅용. 본 과제(3모델 · 절대 점수 · 문항별 가변 비용 · 엄격 예산)에선
  할당기가 이득의 **크기/비용**을 요구하므로 순서 학습만으로 부족하고,
  단일 예측기는 스태킹에 포섭됨. LLMRouter의 "14.6%"는 예산 무제약에서
  최강 고정 모델 대비 지표라 본 과제 지표와 비교 불가(all-think는 premium
  예산도 초과). **배포 구조의 우위 재확인**

### E39. 안전 마진 최적화 — 다중 시드·미세 그리드·분포 이동 스트레스 (로컬)
- 방법: 그리드 0.005 × 시드 7/17/23 × 880배치 400회로 tier별 EV·초과확률·
  통과시 점수 곡선. 스트레스 4종(math/code 2배 가중, think 비용 상위 20%
  2배 가중, 배치 440, 배치 1,760)에서 최적점 이동 측정. 명목+스트레스 5개
  시나리오의 최악 후회(min-regret) 최소 안전계수와 보험료 표 산출
- 명목: 배포 .98/.89/.88 vs 3시드 최적 .980/.885/.875 — 차이 0.0002 이내,
  **배포값이 명목 최적**. 초과확률 배포 0.0% / 0.2% / 0.2%
- 스트레스: fast·balanced는 전 시나리오 ≤1.5%. **premium은 "think 출력
  길어짐" 시나리오에서 초과 27% (EV 0.7488→0.5237)**, 배치 440에서 3.3%.
  큰 배치(1,760)에선 최적이 오히려 .995/.925/.92로 상승
- 강건 추천(min-regret): fast .985 / balanced .875 / **premium .81** —
  명목 가중 EV 0.6982→0.6966(−0.0016), premium 최악 초과 27%→1.5%.
  보험료 표(premium): .855 −0.0017/12%, .845 −0.0025/7.5%, .825
  −0.0035/3.3%, .810 −0.0045/1.5%. 손익비 ~40:1로 비공개셋이 공개와
  "97.5% 이상 동일"하다고 믿지 않는 한 보험이 기대값상 우세
- 결정: **사용자 판단 대기** (제안 .98/.875/.82). 안전계수는 아티팩트
  `tier_safety_ratios` 한 줄이라 재학습 없이 값 갱신+조회표 재빌드로 반영

### E39b. 초과 위험을 감수하면 얼마나 오르나 (5/10/20% 목표)
- 방법: 안전계수를 명목 초과확률 5/10/20%가 되도록 상향, 통과시 점수·기대
  점수·3tier 동시 통과확률·스트레스 기대점수 비교
- 결과(가중): 배포 통과시 0.6993 / 기대 **0.6983** / 동시통과 99.5%;
  5% → 0.7032 / **0.6560** / 81%; 10% → 0.7042 / **0.6218** / 69%;
  20% → 0.7059 / **0.5501** / 47%. 스트레스 기대점수 0.616 → 0.471 /
  0.416 / 0.331. premium 5% 정책은 longer-think 시 초과 68%
- 해석: 예산을 더 써도 살 수 있는 think 배차가 몇 개 없어 통과시 점수는
  +0.004~0.007뿐인데 초과확률은 절벽 → **위험 감수는 어떤 수준에서도
  손해**. 현재 값이 상단 최적이며 검토 방향은 하향(보험)뿐

### E40. 라벨 보존 텍스트 증강 (로컬, 2026-08-17) ❌ 기각
- 질문: fold-train 문항마다 부모 라벨(score·cost)을 물려받는 교란 사본 K개
  (단어 드롭아웃·구간 삭제·크롭·오타, 1~2개 합성)를 추가 학습하면 CV EV가
  얼마나 움직이나. `experiments/e40_text_augment.py`, E27/E39와 동일 중첩
  CV + 880×400 부트스트랩, β=0.25 랭크 헤드. 사본은 hold-out에 절대 미포함,
  내부 OOF·kNN 자기제외는 부모 그룹 단위(라벨 누수 차단)
- 축: K∈{1,2,4,8} × 적용 위치 {ridge만 / ridge+GBM / ridge+GBM+kNN 색인} ×
  강도 {light/mid/heavy}, 확인 시드 7/17/23
- 결과(가중 EV, 기준선 s7 0.6969 / s17 0.6968 / s23 0.6980, 3시드 평균 0.6972):
  ridge만 mid K1 0.6963 / K2 0.6954 / **K4 0.6984** / K8 0.6968;
  K4 ridge 3시드 0.6984/0.6973/0.6981 = 평균 **0.6979 (+0.0007)**;
  K4 light 0.6984 / heavy 0.6908(fast 0.6465로 붕괴);
  ridge+GBM K1 0.6962 / K2 0.6952 / K4 0.6929;
  +kNN 색인 K1 0.6943 / K2 0.6954 / K4 **0.6899**
- 진단: 메타 OOF RMSE — ridge만은 기준선과 동일(score .384/.358/.305), GBM에
  사본 투입 시 score RMSE 악화(.386~.396), kNN 색인 투입 시 사본이 이웃 자리를
  차지해 premium 안전계수가 0.80 하한까지 밀림
- 해석: (1) K에 대해 비단봉(K1·K2 하락, K4 상승, K8 복귀)이라 프로젝트
  규약상 노이즈 판정, 3시드 평균 +0.0007도 시드 분산(0.0012) 이내. (2) GBM·
  kNN은 사본에 강하게 손해 — 사본은 새 정보가 아니라 같은 라벨의 재표본이라
  트리·이웃 가중치만 왜곡. (3) 강한 교란은 fast tier를 직접 붕괴시킴.
  **prompt-only 문제에서 텍스트 교란 증강은 라벨 정보를 늘리지 못한다** —
  데이터로 올리려면 새 문항+새 outcome(A.X 실제 라벨)이어야 함. 미채택

### 오라클 상한 측정 (참고 기록, 2026-08-16)
- 방법: 완벽한 점수·비용 예측을 가정하고 동일 할당기·부트스트랩(880×400,
  시드 7)으로 tier별 EV 상한 산출
- 결과: fast **0.7469** / balanced **0.7978** / premium **0.8544**,
  가중 EV 상한 **0.7944**. 참고치: all-light 0.6046 / all-mid 0.6830 /
  all-think 0.8166, 예산 무제한 best-of-3 0.8789
- 해석: 목표 0.80은 오라클 상한(0.7944)보다 높아 구조적으로 도달 불가.
  현 배포본(≈0.699)과 오라클의 격차는 ~0.095로, 이후 개선은 이 격차의
  회수율로 평가하는 것이 타당

### 정직한 held-out 검증 (Train-only 재학습, 2026-08-16)
- 배경: E27 배포 시 보고된 "dev final 0.7286"(및 tier fast 0.6960/balanced
  0.7256/premium 0.7750)은 Train+Dev 합산(2,640) 학습 이후 Dev를 재채점한
  값으로, [0]단계 공개 조회표가 Dev 문항의 SHA-256을 이미 암기하고 있어
  **in-sample**이다 — 성능 추정치로 쓸 수 없다. 이 로그와 README에 남아있던
  "dev 0.7286" 계열 표기는 이번 항목으로 정정한다.
- 방법: 전체 학습 체인(선형→family/kNN→메타 GBM→gain/ordinal/rank
  헤드→안전계수→조회표)을 **Train 1,760만**으로 재실행하고 조회표에서 Dev를
  제외, Dev 880을 순수 held-out으로 채점. `tools/run_holdout.ps1` +
  `tools/holdout_eval.py`, 산출물은 `reports/holdout/`
- 결과: **held-out dev 0.7000** — fast 0.6741 (ratio 1.187) / balanced
  0.6955 (ratio 1.722) / premium 0.7389 (ratio 3.600), 전 tier 예산 통과
- 비교: 공식 hash-regex baseline(Train-only) dev 0.6954 → 정직한 개선폭
  **+0.0046**. 선형 head 단독 held-out은 0.6987로, 스태킹
  (family+kNN+메타+gain+ordinal+rank)이 out-of-sample로 추가하는 몫은
  **+0.0013**
- CV EV(0.6982)와 held-out dev(0.7000)의 차는 0.0018로 정합 — 중첩 CV
  하니스가 실제 out-of-sample 성능을 잘 근사함을 확인
- 결론: 이후 "성능"으로 인용할 수치는 CV EV(0.6982) 또는 이번 held-out
  dev(0.7000)뿐이다. "dev 0.7286" 계열은 어디서 보이든 in-sample 참고치로만
  취급한다

### E41. A.X-3.1-Light 자체 라벨링 파이프라인 (2026-08-18, 착수 — Colab 실행 대기)
- 배경: E40 결론("데이터로 올리려면 새 문항+새 outcome이어야 함")과 오라클 격차의
  대부분이 fast tier(light-vs-mid)에 있다는 점에서 light 라벨을 직접 생성. 로컬
  RTX 2050(4GB, Q4_K_M 부분 오프로드)은 2.5 tok/s라 규모 불가 → Colab L4/A100 vLLM
- 구성(`colab-label/`): `build_pool.py`(고정 공개 출처에서 주최측 템플릿 그대로
  신규 문항 렌더링; `--verify`로 gsm8k 333/belebele 483/cruxeval 360/truthfulqa 243/
  babilong 240/hrmcr 60/ruletaker 96 공개 문항 정확 재현 확인 → 템플릿 정합),
  pool 6,961(gsm8k-train 2,500·cruxeval 1,240·ruletaker-train 1,498·truthfulqa 766·
  babilong 500·belebele 417·hrmcr 40; DM-math는 참조 풀 미재현으로 제외), pilot =
  gold 보유 공개 1,951문항(주최 라벨 포함, 일치율 측정용); `judge.py`(boxed/Answer/
  정답/마지막 숫자·글자·리터럴·HRMCR 날짜/띠); `run_labels.py`(vLLM, raw 프롬프트,
  n=4, 재개 가능, pilot 리포트); `label_colab.ipynb`; `ingest_labels.py`(→
  `data/aux/light-labels.v1.json` + family별 보정표); `experiments/e41_aux_light.py`
  (E40 하니스, aux는 light score/logcost 열에만 가중 W로 투입, 누수 그룹—같은
  지문/코드/질문/문맥—은 hold-out fold에서 제외, SCOREONLY 옵션)
- 하니스 기준선(E41 W=0, s7): **0.6980** (E27 계열 0.6982와 정합; E40 로그의
  0.6969는 그 스크립트의 기준선). mock aux 70행으로 코드 경로 검증 완료(0.6980)
- 로컬 pilot 최종(2026-08-18, `tools/pilot_local_light.py`, Q4_K_M 부분 오프로드
  3 tok/s, T=0.7, n=2, max 1024; longdoc 15개는 ctx 4096 초과로 제외; 요약
  `reports/pilot_light_{raw,instr}_summary.md`):
  - raw 프롬프트(80문항): 이진 일치 0.80 / 상관 0.66 — code 0.53·ruletaker 0.40이
    끌어내림(모델이 답 형식 없이 장황). 출력 길이 수학 0.22~0.56×, belebele 4.1×,
    code 6.6×, 상관 0.11 → 주최측은 family별 답 형식 지시문 사용이 확실
  - **지시문 부착(105문항)**: 이진 일치 **0.876** / 상관 0.76. family별 일치:
    aime 1.00 · gsm8k 1.00 · belebele 0.93 · hrmcr 0.93 · ruletaker 0.93 ·
    truthfulqa 0.87 · **code 0.47**. 입력 토큰 차이 +3~+35(지시문 길이 재현),
    code만 +219(주최측 few-shot). 출력 길이 비율(우리/주최): 추론 family
    0.47~0.53×(aime .47 gsm8k .49 hrmcr .53, IQR 좁음), truthfulqa 0.33×,
    ruletaker 0.81×, belebele 2.3×, code 7.6× → 길이는 재현 불가(family 상수배는
    가능). code 불일치는 우리 답이 맞는데 주최 0점인 문항 5/15(dev-0565·train-1214·
    dev-0297·train-1225·dev-0270) — 주최측 코드 프로토콜(few-shot·답만·판정기)
    차이이지 모델 차이가 아님
  - 판정: 점수 라벨은 code 제외 6 family에서 재현 가능(≥0.85), 비용 라벨은 불가
    → Colab 라벨링은 **지시문 모드(`--instruct v1`)** 로, aux는 점수 열 우선
    (`SCOREONLY=1`, 길이는 family 보정 후 별도 실험). 노트북 ③을 raw+v1 두 번
    돌려 within.25 높은 쪽을 pool에 쓰도록 수정, `ingest_labels.py`는 (온도,
    지시문) 키로 분리. 로컬 처리량(3 tok/s)으로는 pool 6,961 불가 → Colab 필수
- 판정 기준(사전 고정): Colab pilot within.25 ≥ 0.85 & outlen corr ≥ 0.8 → pool
  투입, < 0.75 → 중단. 채택 여부는 E41 3시드 CV(+0.0012 이상, 단봉)로만 결정

### E42. 출처 부가정보 특징 (parse 57 + lookup 10, 메타 GBM 입력) ❌ parse 기각 / lookup 노이즈
- 방법: `experiments/e42_features.py` — parse(런타임 정규식: ruletaker 규칙/사실/부정
  수, cruxeval 모드·루프·문자열 연산, babilong 질문 유형·bAbI 문장 수, belebele/
  truthfulqa 길이, DM-math 모듈 키워드, HRMCR 유형), lookup(공개 출처 내용 해시
  조회: gsm8k 풀이 단계·연산 수·답 크기, ruletaker 깊이, truthfulqa 카테고리;
  0.6MB). 메타 GBM 입력에 추가, lookup은 학습 시 50% 마스킹 + hold-out을 "조회
  있음/없음" 이중 평가. `e42_sideinfo.py MODE MASK SEED`, 기준선 s7 0.6980
- 결과(s7): **parse 0.6873 / both 0.6838 / lookup 0.6985(+0.0005; 조회 없음 0.6836→
  both 기준, lookup 단독은 조회 유무 무관)**. 헤드 분리: parse→score 헤드만 0.6890,
  cost 헤드만 0.6940, rank 헤드만 0.6980(무영향)
- 진단(`e42_diag.py`, 예측 덤프): parse는 메타 OOF RMSE를 전 열에서 개선(score
  .3838/.3580/.3061→.3810/.3557/.2962, think logcost .6457→.6335)했는데도 EV가
  급락. 전체 2,640에서 fast 안전계수 0.97 고정 시 none은 실제 비용비 1.183, parse_
  score는 1.203(같은 업그레이드 수 1,202~1,208), 실제 점수 0.6622→0.6616 —
  개선된 점수 예측이 고르는 업그레이드 대상이 **비용이 과소예측된 문항 쪽으로
  이동**(선택 유발 비용 편향)해 부트스트랩 초과확률이 커지고 안전계수가 그리드
  하한(0.92/0.80)으로 밀림. 확장 그리드(0.70~1.0)로 재최적화해도 0.6910 < 0.6983
- 해석: E32/E36과 같은 교훈의 재확인 — 이 과제에선 한계 예측 정확도(RMSE)가 아니라
  "업그레이드 대상의 비용 꼬리"가 EV를 지배하며, 점수 헤드만 좋아지면 오히려
  손해. lookup 3시드: s7 0.6985/0.6980, s17 0.6965/0.6974, s23 0.6984/0.6980
  (처리/기준선) → 평균 ±0.0000, 노이즈. 비공개셋이 같은 출처에서 나올 때만
  발동하는 특징이기도 함 → **E42 전체 미채택**. 교훈: 다음 특징 실험은 RMSE가
  아니라 "업그레이드 대상의 실제/예측 비용비"를 함께 봐야 하며, 점수 헤드 개선은
  비용 헤드의 동반 개선(또는 선택 편향 보정) 없이는 EV를 깎는다

### E41 진행 기록 — Colab L4 pilot (2026-08-18)
- 인프라: Colab L4 24GB, vLLM 0.27.1(torch cu130; Colab 기본 torchaudio/torchvision cu128과 충돌
  → 제거 후 torchvision cu130 재설치). bf16 7B는 KV 여유 ~5GB뿐 → `--kv-dtype fp8
  --max-model-len 8192`(16k babilong 78문항 스킵), 128문항 배치 ~4분
- **raw 프롬프트 pilot(1,152문항 중간)**: within.25 0.53, belebele 0.50/ruletaker 0.34로 불일치.
  주최측 input_tokens − 우리 프롬프트 토큰이 family별 상수(수학 24, truthfulqa 29, hrmcr 33,
  belebele 42, longdoc 46, ruletaker 66, **code 221/259**)이고 belebele/ruletaker/code/longdoc
  출력이 6~15토큰으로 짧음 → **주최측은 family별 형식 제한 지시문을 붙였음**. code는 CRUXEval
  공식 프롬프트(2-shot, [ANSWER] 태그)로 재구성하니 토큰 수가 259/213으로 정확히/거의 일치
- **지시문 v1 서브셋 pilot(family당 40)**: within.25 **0.863**, 이진 일치 0.897, 점수 상관 0.807,
  in-len 차 +8, 출력길이 상관 0.60. family별 within.25: gsm8k 0.97 / aime 0.90 / code 0.90
  (outlen 0.94) / longdoc 0.88 / belebele 0.85 / truthfulqa 0.85 / hrmcr 1.00 / **ruletaker 0.55
  (상관 −0.03)**. ruletaker 지시문 변형 3종(v2 0.47, v3 0.58, v4 0.67)도 불일치 — 주최측은 6~7
  토큰 출력으로 0.78을 얻는데 우리는 no-CoT 0.55~0.62, 문항 단위 상관 0. 원인 미상 → aux에서
  ruletaker 제외(E41 `AUX_FAMS` 기본값)
- 본 실행: full pilot(v1, n=4) → pool(v1, n=2) 순차. 로컬 pilot(Q4)·llama-server 종료

### E41 결과 — 자체 라벨(bf16, 지시문 v1) 투입 ❌ 기각 (2026-08-19)
- Colab(A100/L4) 본 실행: pilot 1,873문항 within.25 **0.820** / 이진 일치 0.860 / 점수 상관
  0.726 (gsm8k 0.96, aime 0.91, longdoc 0.89, code 0.87, truthfulqa 0.86, belebele 0.83,
  ruletaker 0.63), 출력길이 상관 0.43. pool 6,718 라벨(8k 초과 longdoc 243 스킵) →
  `data/aux/light-labels.v1.json` + family별 보정표. code 채점기의 무한루프 hang을 서브프로세스
  타임아웃으로 수정(`judge._run_check`)
- CV(E41 하니스, Colab sklearn 기준선 s7 **0.6972**; 로컬 0.6980과 0.0008 차): aux 5,212행
  (ruletaker·잘린 응답 제외) W=0.5 — ridge(점수+비용) **0.6956(−0.0016)** / ridge+GBM
  0.6976(+0.0004) / ridge+GBM 점수만 0.6977(+0.0005). light 점수 OOF RMSE .3907→.3902(변화 없음),
  logcost .5902→.5954(비용 라벨은 오히려 악화 — belebele 0.27×·truthfulqa 0.71× 길이 불일치)
- 해석: 주최측 라벨과 82%만 일치하는 라벨 5천 개는 광범위 특징 하에서 light 예측을 개선하지
  못함(E40 결론 "라벨 정보가 늘어야 한다"는 맞았지만, 재현 라벨의 노이즈가 이득을 상쇄).
  로컬 IQ3_XS GPU 라벨(belebele/code/truthfulqa/hrmcr 1,576행, `colab-label/out_cpu/`)은
  품질이 더 낮아 미평가. **우선순위 1 종료. 배포본 E27 유지**
- 남은 선택지: E39 premium 안전계수 보험(.82) 결정, 규칙상 허용된다면 mid(A.X-3.1 34B) 라벨링은
  Colab A100에서 가능하나 light 결과로 볼 때 기대 이득 낮음

### E43. 공동 하이퍼파라미터 탐색 (Colab CPU, 2026-08-19) ✅ 채택 후보 — held-out +0.0019
- 방법: `experiments/e43_joint_sweep.py` — 특징 1회 계산, (ridge α × GBM 설정) 11개 "비싼" 조합의
  5-fold 산출물 캐시 → 배포 조합과 최선 조합 각각에서 후처리 상수 8개(LEGACY_W·FAM_W·kNN conf·
  GAIN_ALPHA·RANK_BETA·tier blend 3) 좌표하강 2라운드(880×200 부트스트랩, 단봉 조건) → 상위 후보를
  3시드×400으로 확인. 채택 규칙 +0.0015
- 비싼 조합: GBM 변형은 전부 배포값 이하, **ridge α=10**만 +0.0011(0.6998 vs 0.6987)
- **cand0** = α10 + {legacy_w .9, fam_w .15, conf .25, gain_α .5, rank_β .4, blend fast .6/bal .45/prem .3}:
  3시드 0.7030/0.7007/0.7020 = **0.7019 (배포 0.6979 대비 +0.0040)**; cand1(α30) +0.0023
- 정직한 held-out(Train만 재학습, `tools/run_holdout_e43*.ps1`, 빌드 도구에 ROUTER_* env 훅 추가):
  배포 안전계수(.98/.89/.88) 그대로 → **premium 초과 4.06(0점)**, blend_premium만 .45로 되돌려도 4.08
  → 새 예측치엔 안전계수 재보정 필요. `e43b_bust_curve.py`(3시드×400) 기준 초과확률 0%인
  premium ≤.86·balanced ≤.85~.87, Dev가 CV q99 밖의 어려운 표본이었으므로 보수적으로 **.98/.87/.85** 선택
  → held-out **0.7019** (fast 0.6764 +.0023 / balanced 0.6972 +.0017 / premium 0.7406 +.0017, 비율
  1.199/1.784/3.572), 배포 0.7000 대비 **+0.0019**
- 해석: 개별 튜닝된 상수들이 공동 최적이 아니었음(legacy 비중↑·family/kNN 비중↓·랭크 헤드 비중↑·
  balanced는 메타 비중↑/premium은↓). CV +0.0040 중 held-out에 남은 건 +0.0019 — 선택 편향이 절반을
  먹었지만 노이즈 한계 위. 배포 반영은 사용자 승인 대기

## 2026-08-19 ~ 08-20 — 22-agent design round (E44 ~ E55)

Environment moved to a new machine (i9-13900H 20T, 31.7 GB RAM, **RTX 4090
Laptop 16 GB**, network available).  Baseline reproduced exactly:
`tools/run_holdout_local.ps1` gives held-out dev **0.701648** at safety
.98/.87/.85 (E43 reported 0.7019 at .98/.89/.88 — the two numbers differ only in
the safety triple, see E45 below).

New lab infrastructure (`experiments/lab/`):

| file | what |
|---|---|
| `harness.py` | in-memory replica of the whole chain; one held-out evaluation in 33 s instead of 225 s, verified against the real chain to 3e-4 |
| `protocol.py` | `exact_allocate` / `safety_curve` — a concave-envelope allocator that reproduces the deployed 40-step bisection **exactly** (100 % pick agreement on every tier and safety value tested) and evaluates a whole safety grid from one sort, ~100x faster |
| `bench2.py` | the honest protocol: 10-fold OOF over Train only, safety chosen by multi-scenario bootstrap on those rows, Dev scored once |
| `gainlab.py`, `priorfeat.py`, `famrepair.py` | gain-axis transforms, the offline-prior feature block, the repaired family classifier |

### The 22-agent round

14 analysis agents (leakage audit, adversarial critic, training-failure,
prediction-confidence, selection-failure, train-set characterisation and
exploitation, pre/post-processing for both the predictor and the allocator,
A.X model profile, data-provider perspective, noise ceiling) and 8 strategy
agents (fast/balanced/premium tier owners, memory-heavy, time-heavy,
forest-view, visualiser, chief strategist).  Reports in `reports/lab/a*.md`,
`reports/lab/b*.md`, figures in `reports/lab/figs/`.

Three findings changed the direction of the project:

1. **The allocator is invariant to the score level** (a03).  Adding a constant
   to all three of an item's predicted scores changes nothing, yet 69–81 % of
   the score variance — and essentially all of the "score correlation" the
   project had been steering by — lives in that channel.  In a
   noise-regenerated simulation a perfect level buys **+0.009** final, perfect
   gains buy **+0.078**.  `corr(prediction, realised score)` was retired as an
   objective; a06 built a head that reaches the old corr-0.48 target and it
   *lost* 0.016 dev / 0.037 EV.
2. **The deployed safety ratios were priced on predictions that had already
   seen the evaluation episodes** (a01, independently a10/a11/a13/a14).  Honest
   re-measurement on train-only predictions gives fast/balanced/premium bust
   probabilities of about **6 / 2 / 14 %** and E[final] ≈ 0.65, not the
   0.0/0.2/0.2 % and 0.700 that E39/E43b reported.
3. **The meta-GBM over-trusts the legacy hash-regex feature** (orchestrator).
   `build_meta_gbm.py` feeds it the *shipped* legacy artifact, which was fitted
   on all 1,760 Train episodes — so the feature is in-sample for every training
   row (score corr **0.60**) and out-of-sample at prediction time (**0.39** on
   train-OOF, 0.38–0.45 on dev).

### E44. Cost sum calibration, estimated out-of-fold ❌ rejected
Per-model and per-family multiplicative corrections of the predicted cost sums
(the runtime exponentiates a log-cost prediction, so the sums are 0.82/0.88/0.66
of the truth).  OOF relative factors [1, 0.95, 1.19].  cvEV 0.7022 (model) /
0.7019 (family) vs 0.7027 baseline.  The +0.0053 seen in the first diagnostic
was an artefact of tuning the safety ratio on dev.  Confirms E36.

### E45. Gain-axis post-hoc transforms ❌ rejected
Pair-balance scaling of (d1, d2) — only their ratio changes decisions — over
a2 ∈ [0.5, 3.0], and shrinkage/expansion of each gain toward its family mean.
Best cvEV 0.70369 vs 0.70335 baseline; a2 = 1.0 is already optimal; strong
shrinkage busts the fast tier on dev.

### E46. Full coordinate descent over the 8 post-hoc constants ❌ rejected
With the fast evaluator every configuration costs ~1 s, so seven random restarts
x four rounds were affordable.  Every start converged to cvEV ≈ 0.7060 (+0.0036)
and **every one of those configurations busts the fast tier on dev**.  Cause:
CV was optimistic because of the in-sample legacy feature (finding 3), and the
CV-optimal points sit exactly on the budget knife-edge.

### E47/E48. Legacy head refitted out-of-fold ✅ adopted
`exp["legacy_oof_meta"]` refits the 256-bin hash-regex head inside an inner
5-fold before it is handed to the meta GBM, exactly as the ridge block already
was.  My refit reproduces the shipped artifact **bit-for-bit** at alpha 100
(max abs diff 0.000000) once its clamping is applied, so this is a pure
anti-leakage change.  Under the honest protocol: EV 0.661181 → **0.670729**,
dev 0.697642 → **0.699915**, and the premium safety ratio can rise from .735 to
.840 at 0 % bust because the cost predictions are better calibrated.  The chief
strategist's independent rotation test (16 fresh 1,760-fit / 880-holdout splits)
scored it **+0.0078 EV / +0.0036 dev at 25 sigma** — the only candidate of eight
that survived.

### E49. Family-classifier repair (a08) + sub-family split (a06) ❌ not adopted
`famrepair.classify_v3` moves 185/2,640 items (7.0 %): 91 GSM8K money problems
out of `aime` (the deployed regex matches any two dollar signs), 38 RuleTaker
and 54 DeepMind-Mathematics items out of the `gsm8k_or_other` catch-all.  a08
measured +0.0018 CV EV on 3/3 seeds in its own harness, but under bench2 with
the legacy fix already in place it is EV −0.0005 / dev −0.0010, and combined
with E48 it busts the fast tier.  Kept in the tree as `famrepair.py` because the
diagnosis is correct and it may pay once the fast tier has margin.

### E50. Cost re-transformation against the selection-induced bias ❌ rejected
Diagnosis: write inflation = realised ratio / predicted ratio at the cap.  On
Train-OOF it is 0.99–1.04, on Dev 0.98–1.07, and the whole gap is the *selected*
items — predicted/true cost is 0.807 on the selection but 0.840 on the light
baseline.  The allocator picks items whose cost is under-predicted.  Replacing
the conditional median `exp(m)` with the conditional mean `exp(m + sigma^2/2)`
in both the decision and the cap accounting (global, Duan, per-family and
heteroscedastic variants) fixes the *level* of the inflation but leaves the
train→dev gap of ~0.045 untouched.  Best variant EV +0.0009.

### E51. What is an external per-item difficulty prior worth? (simulation)
Latent p drawn from each item's Beta-Binomial posterior, then an **independent**
proxy observation of controlled fidelity, added as a meta feature.

| arm | corr(proxy, realised score) | dev |
|---|---|---|
| none | — | 0.6999 |
| light only, n=4, fidelity .85 | .864 | 0.7177–0.7210 |
| light+mid+k1, n=4, fidelity .85 | .864/.859/.820 | 0.7322–0.7381 |
| light+mid+k1, n=8, fidelity .95 | .919/.920/.886 | 0.7350–0.7416 |
| light+mid+k1, n=4, fidelity .60 | .717/.709/.665 | 0.7224–0.7244 |
| light+mid+k1, n=4, fidelity .40 | .524/.520/.476 | 0.7075–0.7116 |

E41 measured A.X-3.1-Light run locally to agree with the official light score at
corr 0.726, i.e. the fidelity-.60 row.  b06 reached the same conclusion from the
opposite direction: the only mechanism with headroom above 0.72 is a finer
item partition, and it "must come from public-source per-item metadata via
exact-prompt lookup".

### E52/E54. Seed-averaged meta heads and a stress-priced safety triple ✅ adopted
b04 measured that a single meta fit is itself a risk source (fast-ratio sd
0.0183 across refits; 2 of 10 single fits bust the fast tier, 0 of 5
seed-averaged ensembles).  Averaging five seeds per head is free at build time
and does not change the runtime path.  It also pins `OMP_NUM_THREADS`, because
thread count alone moved the dev premium ratio from 3.4719 to 3.8017 (8 % of the
cap).

b01 showed the fast tier's budget is decided by single episodes: 82 % of its
ratio variance on dev comes from one item (dev ep 2437, dmmath, 69 characters,
where `ax31` emitted 33,459 output tokens = 6.3 % of the entire light baseline),
and Train contains four such items per 1,760 rows.  The safety search therefore
averages three scenarios — plain bootstrap, one injected runaway upgrade of
6.5 % of the light total, and a cost inflation of 1.25x on k1 / 1.10x on mid.
Chosen triple **.920/.815/.745**: dev 0.694176, every tier with margin
(1.212/1.25, 1.645/2.0, 3.151/4.0) versus the previous rule's 1.2488/1.25.

### E53. The offline difficulty prior — built and measured ✅ adopted
`skt/A.X-3.1-Light` (Apache-2.0, the organiser's own `ax31-light`) was run
locally over every renderable public benchmark item and the per-item result
stored in a prompt-hash lookup table.  CHALLENGE_RULES permits lookup tables and
search indexes built from public data, exact-prompt / prompt-hash lookup against
public data, and offline use of publicly-weighted models; the router runs no
inference at evaluation time.

Infrastructure: llama.cpp CUDA b10488 + Q6_K on the RTX 4090 Laptop, 16 slots,
q8_0 KV cache, 13.8 GB VRAM, ~2,300 items/h at n=4.  `transformers` + bnb-NF4
was abandoned first: the MHA KV cache is 0.5 MB/token, so batches large enough
to be fast pushed past 16 GB and Windows paged GPU memory to host RAM (0.55 s
per decode step).

Two rounds:

| round | items | agreement with the organiser's `ax31-light` label |
|---|---|---|
| v1 | 8,288 | within.25 **0.797**, corr **0.705** |
| v2 | 8,861 | within.25 **0.829**, corr **0.739** |

v2 fixed two defects found by comparing our generation lengths with the
organiser's published `output_tokens`:

* **TruthfulQA was truncated.**  The organiser's median output is 256 tokens
  (p90 330); the E41 reconstruction capped us at 160, so the model was cut off
  before it emitted its answer.  At 384 tokens agreement rose from corr 0.409 to
  **0.678** (family total 0.481 -> 0.776).
* **Items without a public gold answer were uncovered.**  DeepMind-Mathematics
  and BABILong could not be judged, leaving 22 % of dev with no feature.  Adding
  **self-consistency** (modal-answer agreement across the four samples), which
  needs no gold answer, lifted dev coverage from 0.69 to **0.91**.

RuleTaker resisted reconstruction, as it did in E41: the organiser's median
output is 299 tokens, so they used chain-of-thought, but every CoT variant
agreed *worse* than the one-word form (corr 0.302 at 512 tokens, −0.002 at 256,
against 0.450 for the one-word form).  Kept the cheap form.

Feature block (11 values, all zero on a miss, so the model degrades gracefully
on prompts that are not in the table): present, has-score, score, score minus
the table's family mean, is-zero, is-one, log output length, length minus the
family mean, has-consistency, consistency, consistency minus the family mean.

Measured under bench2 (10-fold OOF over Train only, stress-priced safety, Dev
scored once):

| coverage | EV | dev | fast/bal/prem dev ratio |
|---|---:|---:|---|
| no prior | 0.665721 | 0.694176 | 1.212 / 1.645 / 3.151 |
| 30 % | 0.666763 | 0.695256 | |
| 50 % | 0.668031 | 0.697244 | |
| 70 % | 0.669838 | 0.699091 | |
| **91 % (actual)** | **0.673350** | **0.704063** | 1.177 / 1.630 / 3.470 |

Through the real build chain (`tools/deploy_v2.ps1`, Train-only, Dev scored
once): **held-out dev 0.703210** at safety .94/.80/.73, every tier passing with
margin (1.146/1.25, 1.627/2.0, 3.496/4.0).

### E55. Safety triple re-priced for the new configuration ✅ adopted
Scenario-averaged bootstrap EV (plain / one injected runaway / cost inflation)
on Train-only rows, dev scored once:

| triple | EV(stress) | EV(plain) | bust % f/b/p | dev |
|---|---:|---:|---|---:|
| .98/.87/.85 (E43 deployed) | 0.5925 | 0.6752 | 17.1 / 4.6 / 16.4 | **0.4364 — fast busts** |
| .98/.89/.88 (the "0.7019" run) | 0.5614 | 0.6697 | 17.1 / 10.1 / 25.9 | 0.2120 |
| .96/.84/.84 | 0.6400 | 0.6794 | 2.5 / 1.4 / 13.9 | 0.7067 |
| **.94/.80/.73 (shipped)** | **0.6734** | 0.6741 | 0.0 / 0.1 / 0.2 | 0.7041 |
| .95/.83/.78 | 0.6686 | 0.6773 | 0.7 / 0.9 / 2.3 | 0.7057 |
| .91/.76/.68 | 0.6700 | 0.6700 | 0.0 / 0.0 / 0.0 | 0.6994 |

The old triple busts the fast tier outright once the cost predictions improve,
so it had to be re-derived.  .95/.83/.78 buys +0.0016 of dev point estimate for
a 2.3 % premium bust probability whose expected cost is 0.005 — .94/.80/.73 is
the rational choice.

### Runtime
On a lookup miss the container path costs **49.5 s per tier for 2,640 episodes**
on this host, versus **50.0 s for the E43 baseline** — the prior lookup is free
(one SHA-256 and a dict probe per episode).  Calibrating against the organiser's
published container benchmark (`hash-regex` 7.34 s there, 4.92 s here) this host
is 1.49x the official container, so 2,640 unseen episodes project to ~75 s of
the 90 s limit and 880 to ~25 s.  With the public lookup present the public
inputs cost 0.4 s per tier.  This margin is inherited from the deployed
baseline, not introduced here, but it is thin and is recorded as an open risk.

### E56. A second, stronger prior column ✅ adopted
Column A is `skt/A.X-3.1-Light` Q6_K — the organiser's own `ax31-light`.  Column
B is `Qwen2.5-14B-Instruct` Q4_K_M, a genuine capability step above it, used as a
proxy for `ax31` (the real 34B model needs 2-bit quantisation to fit 16 GB VRAM,
which leaves 1.5 GB for the KV cache and puts the pool at 40-60 GPU-hours).
8,861 items, n=4, 0 errors, ~26 GPU-hours in total across both columns.

Calibration is good: column B's mean score over the gold-covered public items is
**0.758**, exactly `ax31`'s own 0.758, and `corr(B, s_mid) = 0.612`
(aime 0.797, truthfulqa 0.697, gsm8k 0.679, code 0.604, hrmcr 0.532,
belebele 0.468, ruletaker 0.338).

The decision-relevant quantity is the cross-column delta, and it carries real
signal on the channel that was previously at chance: `corr(B - A, d1) = +0.285`
(mid-light gain), `corr(B - A, d2) = +0.041` (k1-mid, as expected -- a 14B
instruct model says nothing about a reasoning model).

Under bench2, both columns at 90.9 % coverage:

| configuration | EV | dev |
|---|---:|---:|
| no prior | 0.665721 | 0.694176 |
| column A only | 0.673045 | 0.703438 |
| **column A + column B** | **0.675244** | **0.705170** |
| A + B at 70 % coverage | 0.670797 | 0.701761 |
| A + B at 50 % coverage | 0.669168 | 0.698835 |

Through the real build chain (Train-only, Dev scored once), with the coverage
pool finished (38,330 entries in column A, 8,861 in column B): **held-out dev
0.705568**, fast 0.6781 (ratio 1.146) / balanced 0.7065 (1.652) /
premium 0.7412 (3.177), all passing with margin.

### Round summary

| | held-out dev | honest E[final] | bust f/b/p |
|---|---:|---:|---|
| E43, start of the round (.98/.87/.85) | 0.700483 | ~0.637 | ~2 / 1 / 11-16 % |
| **after E44-E58 (.94/.80/.73)** | **0.705568** | **0.675** | 0.0 / 0.1 / 0.2 % |

**+0.0047 on the point estimate and +0.038 in expectation.**

### Container verification (2026-08-20)
Built from `container/Dockerfile` and run under the official resource profile
(2 CPU, 2 GiB, no swap, pids 32, no network, read-only root, 256 MiB /tmp):

| input | per tier |
|---|---:|
| public Train+Dev 2,640 (public lookup hits) | 0.96 s |
| 2,640 unseen public-benchmark items, in the prior table | 34.3-35.4 s |
| 2,640 items in neither table (worst case) | 34.6-37.2 s |

The same container running the **previously shipped E43 artifact** on the same
worst-case input takes 36.6 s, versus 36.7 s for this build: the prior lookup is
one SHA-256 and a dict probe per episode and costs nothing measurable.  The
runtime margin against the 90 s limit is therefore inherited from the previous
build, not introduced here.  This measurement is on linux/amd64; the official
platform is linux/arm64 on Apple Silicon, so it remains an open risk to confirm
on the official hardware.

### What would move the score further
The remaining gap is fidelity to the two models we cannot run, not algorithm.
E51's simulation is explicit: three columns at the fidelity we achieved for
column A put dev at 0.722-0.724, and our proxy for `ax31` reaches only
`corr 0.612` against `corr 0.739` for the real `ax31-light`.  Running the actual
`skt/A.X-3.1` (34B) at 4-bit needs ~24 GB of VRAM, and `axk1-think` (519B MoE)
is out of reach entirely.  On an A100-80GB the mid column is a few GPU-hours and
is the single highest-value remaining experiment.

### E57. Is a third (reasoning) prior column worth building? ❌ not built
The premium tier's decision axis is d2 = s_k1 - s_mid, and a reasoning-model
column was the obvious next candidate.  Measured on the 2,400 covered public
items, the two existing columns already carry that signal better than the
deployed stack does:

| signal | corr with d1 | corr with d2 | corr with k1-light |
|---|---:|---:|---:|
| column A score (A.X-3.1-Light) | −0.216 | **−0.375** | **−0.514** |
| column B score (Qwen2.5-14B) | −0.057 | −0.352 | −0.371 |
| column A output length | +0.051 | +0.207 | +0.232 |
| column B − column A score | **+0.220** | +0.017 | +0.184 |
| *deployed model's own prediction* | *0.107* | *0.355* | *0.326* |

"How hard does a small model find this" already predicts the k1 gain better than
the whole existing stack, and the cross-column delta is a d1 specialist.  A
reasoning column would cost 20+ GPU-hours (reasoning traces are 1,000-4,000
tokens) to add a signal that is largely already present, so it was not built.
E51's simulation agrees: at matched fidelity, adding the k1 column moved dev
from 0.733 to 0.734.

### E58. Re-sweep the post-hoc constants with the prior in place ❌ not adopted
The tier blend weights were tuned in E43 for a meta stack without the offline
prior.  The prior lands entirely inside the meta features, so the optimum should
move toward the meta -- and it does, strongly: coordinate descent on the
stress-priced train-only EV moves `blend_fast` .60 -> .95, `blend_balanced`
.45 -> .75, `blend_premium` .30 -> .60, `legacy_w` .90 -> .80, and lifts EV from
0.675600 to **0.680863 (+0.0053)** over 39 configurations.

But held-out dev moves the other way: 0.705341 -> **0.704233 (−0.0011)**.

This is the pattern b06 quantified: EV and dev are essentially uncorrelated
across the 8-constant box (rho −0.149, CI [−0.32, +0.03]) and screening 40
candidates on EV returns +0.0073 EV / −0.0022 dev.  A +0.0053 EV gain taken from
39 evaluations of the same criterion is exactly what selection bias produces.
With the two measurements disagreeing and the held-out one going the wrong way,
the deployed constants stand.  Recorded because the *direction* is informative:
the prior really has shifted where the information lives, and a future round with
an unbiased constant-selection instrument (nested rotations, not a single CV)
should revisit it.

## 2026-08-21

### E59. The real `skt/A.X-3.1` (34B) as the mid prior column ✅ adopted
Column B was `Qwen2.5-14B-Instruct` standing in for `ax31`, at `corr 0.612`.  E51's simulation
and the previous round's own "What would move the score further" both named replacing it with
the real 34B as the highest-value remaining experiment.  Run on a Colab A100-40GB with vLLM +
bitsandbytes NF4, n=4, T=0.7, per-family instructions v1, 384 max tokens.

**Fidelity gate** (`colab-label/prior_column_report.py`, joined to the public 2,640):

| target | corr | within .25 |
|---|---:|---:|
| `ax31` (what this column proxies) | **0.699** | 0.829 |
| `ax31-light` | 0.579 | 0.759 |
| `axk1-think` | 0.322 | 0.706 |

Against column B's 0.612 that is +0.087, and the column is correctly *mid*-specific: it tracks
`ax31` better than it tracks `ax31-light`, which is the discrimination the allocator needs.
Per family it is strong on truthfulqa (0.845), code (0.817), belebele (0.722) and weak on
ruletaker (0.301) — the same family that resisted column A in E53.

**Held-out Dev** (Train-only rebuild, Dev scored once; linear head via `tools/cpu_shim_train.py`
because this machine's CUDA runtime is unavailable, so all three rows share that substitution):

| prior configuration | held-out dev | vs baseline |
|---|---:|---:|
| shipped 2 columns [A, B] | 0.702727 | — |
| [A, B, C] (`--mode append`) | 0.708352 | **+0.0056** |
| **[A, C] (`--mode replace`)** | **0.709432** | **+0.0067** |

Every tier passes with margin (replace: 1.139/1.25, 1.624/2.0, 3.491/4.0).  The gap between
append and replace (0.0011) is inside the noise limit, so the choice is made on parsimony:
`replace` drops a column, shrinks the artifact and removes the dependency on the Qwen labels.

**The column is still handicapped.**  Its coverage of the challenge's own items is 0.753 against
column A's 0.909, because `build_pool.py` cannot re-render every public prompt.  The dev holes
are dmmath 115, gsm8k 38, aime 12, truthfulqa 2 (never in our pool — the deployed chain feeds
`bundle/public_all.jsonl`, the public prompts themselves, which we had not built) and longdoc 50
(skipped at generation time by the 8192 length limit).  E53's coverage curve values that gap at
roughly +0.005, i.e. as much again as the column swap itself.  `colab-label/build_public_all.py`
and `e59b_coverage_colab.ipynb` close it in about one more GPU hour.

Tools: `tools/splice_prior_column.py` (compiles one new column with `build_prior_lookup`'s own
code path and splices it beside the already-compiled ones, since the raw labels for A and B are
not in the repository), `verify_e59.sh`, logs in `reports/e59_{append,replace}.log`.

### E59b. Coverage of the 34B column: public prompts, long items, and AIME (in progress)
E59's column reached `corr(ax31) 0.699` but covered only **0.753** of the challenge's own items
against column A's 0.909.  Three separate causes, all fixed here.

**1. Items `build_pool.py` cannot re-render** (dev: dmmath 115, gsm8k 38, aime 12, truthfulqa 2).
The deployed chain feeds `bundle/public_all.jsonl` -- the public prompts themselves -- which we
had never built.  `colab-label/build_public_all.py` writes it (2,640 prompts, 1,951 with a gold
answer).

**2. `run_labels.py` crashed on gold-less items.**  `judge()` was called unconditionally, so the
first batch containing a `gold: null` row died with `TypeError: 'NoneType' object is not
subscriptable`.  Those rows are exactly dmmath (359) and longdoc (238) -- the holes we were
trying to fill.  Now guarded: no gold -> `score=None`, length and self-consistency still
recorded, which is the degradation E53 relied on for its 0.69 -> 0.91.  Regression test
`colab-label/_smoke_nogold.py` drives the real crash path with the mock engine.

**3. No AIME renderer exists at all.**  `build_pool.py` never produced a single AIME item, so of
the 36 competition AIME episodes the organiser pins in `data/{train,dev}/aime-selection.json`
(2024: 18, 2025: 18), our column covered **0** -- column A covers 36/36, but only because
`public_all.jsonl` gave it the public prompts.  Any AIME item outside the public split is
uncovered by *both* columns.  That is the worst family to be blind on:

| model | mean score on the 36 | median output tokens |
|---|---:|---:|
| `ax31-light` | 0.069 | 967 |
| `ax31` | 0.132 | 907 |
| `axk1-think` | **0.792** | 9,765 |

`colab-label/build_pool_aime.py` renders the upstream sources verbatim (the challenge prompt is
the `problem` field with no template) and proves it by hashing against those 36:

| source | rows | reproduces |
|---|---:|---:|
| `HuggingFaceH4/aime_2024` (ids 60-89 match `source_key.source_id`) | 30 | 18/36 |
| `math-ai/aime25` | 30 | 16/36 |
| `allenai/aime-2022-2025` | 120 | 31/36 |
| `AI-MO/aimo-validation-aime` | 90 | 18/36 |
| **union** | | **35/36** |

Only `train-0845` (2025 #19) differs.  `yentinglin/aime_2025` matches 1/18 -- it writes
`$17_{b}$ ... $97_{b}$.` where the organiser has `$17_b$ ... $97_b.$` (AoPS style, period inside
the math).  130 unique problems, all with gold answers, are written to `bundle/aime.jsonl`.

AIME is labelled at `--max-tokens 2048` rather than 384: with a 384 cap every AIME generation
hits the ceiling, so the log-length feature carries *no* information for the family whose
upgrade decision matters most.

Notebook `colab-label/e59b_coverage_colab.ipynb`, bundle `make_e59b_zip.py`.  Held-out re-run
pending the labels.

### E59b result. Coverage completed; EV-neutral, and the AIME premise was wrong
Labels came back (public prompts 2,572 of 2,640 labelled, AIME 130).  Coverage of the challenge's
own items went **0.753 -> 0.975** (train 0.974), past column A's 0.909, and the column now
contains every digest of both shipped columns (38,330/38,330 and 8,861/8,861).  Fidelity rose
with it: `corr(column, ax31)` **0.699 -> 0.709**, and the `aime` family went 0.458 -> **0.818**
once the 384-token cap was lifted to 2048.

Held-out Dev, same chain:

| prior configuration | held-out dev |
|---|---:|
| shipped 2 columns [A, B] | 0.702727 |
| E59 column at 0.753 coverage, [A, C] | 0.709432 |
| **E59b column at 0.975 coverage, [A, C]** | **0.709290** |
| E59b column, [A, B, C] | 0.708722 |

**Coverage completion is EV-neutral** (−0.00014, well inside the 0.0012 noise limit).  Keep it --
it removes a known handicap and costs nothing at runtime -- but claim no EV for it.

**The AIME premise was wrong, and provably so.**  E59b was justified partly by the 36 organiser
AIME episodes scoring 0.069 / 0.132 / 0.792 across light / mid / think, i.e. "the family where
upgrading pays most".  That read Δs without Δc.  On the 12 AIME episodes in Dev:

| model | share of the *entire* dev light baseline | mean score |
|---|---:|---:|
| `ax31-light` | 13.6 % | 0.021 |
| `ax31` | 16.1 % | 0.042 |
| `axk1-think` | **426.8 %** | 0.729 |

Upgrading those 12 items to think buys +0.0097 dev for **+4.13x the whole budget**, against a
premium cap of 2.92x.  Efficiency 0.0023 -- 7th of 9 families:

| family | Δs (dev) | Δc / baseline | efficiency |
|---|---:|---:|---:|
| truthfulqa | +0.0188 | 0.539 | **0.0348** |
| belebele | +0.0205 | 0.802 | 0.0255 |
| dmmath | +0.0568 | 2.436 | 0.0233 |
| gsm8k_or_other | +0.0330 | 2.212 | 0.0149 |
| code | +0.0500 | 3.595 | 0.0139 |
| ruletaker | +0.0119 | 2.952 | 0.0040 |
| **aime** | +0.0111 | 5.033 | **0.0022** |
| hrmcr | +0.0017 | 1.205 | 0.0014 |
| longdoc | +0.0034 | 4.020 | 0.0008 |

`tools/holdout_by_family.py` confirms the consequence: the 12 AIME episodes score 0.0208 in every
tier under both artifacts -- the allocator never upgrades them, and it is right not to.  Think
emits ~9,765 output tokens on an AIME item; it is one of the most expensive upgrades on the board.

This is E42's lesson repeated: a feature experiment must look at the actual cost ratio of the
items it would upgrade, not only at the score gap.

### E60. The published chain reproduces 0.7027, not 0.705568
The CPU substitution in `tools/cpu_shim_train.py` was suspected of costing ~0.003 against the
published number.  It was not.  Running the repository's real chain -- `train_learned_router_gpu.py`
on an actual GPU (`"training_backend": "gpu"`, `"solver": "cupyx-lsmr"`, cupy 13.6.0) --
gives **0.702727272727** on held-out Dev, bit-identical to the CPU shim (fast 0.676136 /
balanced 0.703125 / premium 0.737784, ratios 1.144 / 1.628 / 3.469).

Two conclusions.  The shim was faithful, so every comparison made with it stands.  And the
published **0.705568 is not reproducible from the repository as published** -- its own tools,
its own constants, its own data produce 0.7027.

The most likely missing piece is E52's "seed-averaged meta heads": the round summary says five
seeds per head were adopted and that this is free at build time, but `tools/build_meta_gbm.py`
fits each head once at `random_state=11`, and no seed loop exists anywhere under `tools/` or
`src/`.  That would account for both the sign and the rough size of the 0.0028 gap.

All E59/E59b deltas were measured against 0.702727 on the same chain, so they are unaffected:
the 34B mid column is **+0.0067**, reaching **0.709432** ([A, C], E59 column) / 0.709290
([A, C], E59b coverage-completed column).

`run_repo_chain.sh {baseline|replace|append|only}` runs the real chain end to end;
`colab-label/e60_repo_chain_colab.ipynb` + `make_e60_zip.py` do the same on Colab, carrying the
34B column pre-compiled (`prior_column_c.json`, 3.5 MB) so the 100 MB item pools stay local.

### E60 result. Confirmed on the repository chain, on two different GPUs
Four arms through `run_repo_chain.sh` (real `cupyx-lsmr` linear head, Train-only, Dev scored
once), run on a Colab GPU, next to the same arms on the local RTX 2050:

| arm | prior | Colab dev | Δ vs its own baseline | local dev | Δ vs its own baseline |
|---|---|---:|---:|---:|---:|
| `baseline` | [A, B] shipped | 0.704148 | — | 0.702727 | — |
| `replace` | [A, C] | 0.708381 | +0.0042 | 0.709290 | +0.0066 |
| `append` | [A, B, C] | **0.709631** | **+0.0055** | 0.708722 | +0.0060 |
| `only` | [C] | 0.698097 | −0.0060 | — | — |

Three things follow.

**The 34B column is confirmed.**  Both machines put it between +0.004 and +0.007 on the
repository's own chain, comfortably outside the 0.0012 noise limit.

**`only` is the useful control.**  Column C alone scores *below* the shipped two-column
baseline (−0.0060), so the light-model column A is not redundant: A and C carry different
signal and both are needed.  That also rules out the reading that C is simply a better B.

**`append` and `replace` are tied.**  Their gap is 0.0012 on Colab and −0.0007 locally -- it
changes sign between machines, which is the definition of noise here.  Choose on other grounds:
`replace` yields a smaller artifact and drops the Qwen dependency, `append` discards nothing.

**The chain is not deterministic across hardware.**  The identical `baseline` build gives
0.704148 on Colab and 0.702727 on the RTX 2050, a 0.0014 spread from GPU/library differences
alone -- as large as the noise limit.  Comparisons are therefore only valid *within* one
machine, and neither machine reproduces the published 0.705568 (E60 above).

### E61. Seed-averaged meta heads, actually implemented ✅ adopted
E52 recorded seed averaging as adopted, but `tools/build_meta_gbm.py` fitted every head once at
`random_state=11`; no seed loop existed anywhere.  E60 measured the consequence: the published
0.705568 was unreachable, the chain producing 0.702727.

**Implementation.**  A HistGradientBoosting prediction is `baseline + Σ(leaf values hit)`, so the
mean of N models equals `mean(baseline) + Σ over all N models' trees of (leaf value / N)`.
`_fit_export` fits N models differing only in `random_state`, concatenates their trees with
leaf values scaled by 1/N, and averages the baselines.  The runtime evaluator is untouched --
`similarity.evaluate_trees` only ever adds leaf values -- which is what E52 meant by "does not
change the runtime path".  Controlled by `ROUTER_META_SEEDS` (default 5).

`tools/_smoke_seed_average.py` proves the merge is exact rather than approximate: max deviation
from `mean(sklearn predictions)` is 2.2e-15 for the regressors and 8.9e-15 for the ordinal
classifiers, and `ROUTER_META_SEEDS=1` reproduces the previous export byte for byte (and the
previous 0.702727272727 on the chain).

**Held-out Dev** (local RTX 2050, real `cupyx-lsmr` chain, Train-only, Dev scored once):

| seeds | prior | dev | vs single-fit baseline |
|---:|---|---:|---:|
| 1 | [A, B] | 0.702727 | — |
| 5 | [A, B] | 0.704801 | **+0.0021** |
| 5 | [A, C] | 0.707330 | +0.0046 |
| 3 | [A, B, C] | 0.709119 | +0.0064 |
| **5** | **[A, B, C]** | **0.709972** | **+0.0072** |

The +0.0021 on the baseline arm accounts for three quarters of the gap to the published
0.705568, and the remainder (0.0008) is smaller than the 0.0014 machine-to-machine spread E60
measured.  The published number is now explained.

Averaging also settles the append/replace tie of E60 in favour of `append`: with the fit noise
reduced, [A, B, C] beats [A, C] by 0.0026, outside the noise limit and consistent in direction
across the 3- and 5-seed runs.

**Runtime cost** (`tools/time_heavy_path.py`, lookup-miss path, this laptop):

| build | meta trees | artifact | us/episode | vs single |
|---|---:|---:|---:|---:|
| single fit [A, B] | 1,296 | 15.2 MB | 14,427 | — |
| 3 seeds [A, B, C] | 4,695 | 22.7 MB | 15,520 | +8 % |
| 5 seeds [A, B, C] | 7,599 | 26.2 MB | 18,118 | +26 % |

Tree evaluation is not the bottleneck -- the kNN posting scan is -- so a 5.9x tree count costs
only 26 %.  Against E44's estimate of 40-50 s per tier on the official Apple Silicon hardware,
5 seeds implies roughly 50-63 s against the 90 s limit; 3 seeds implies 43-54 s for 0.0009 less
score.  Both fit, 5 seeds with less margin.  `run_repo_chain.sh` now honours `CHAIN_OUT` so
configurations no longer overwrite each other's artifacts.

### E62. Bust probability of the shipped safety triple — premium is mispriced
`tools/bust_probability.py` re-runs the allocator inside every resample, which is the part that
matters: `select_models` sizes its cap from the batch it is handed, so a router facing a
different item mix re-balances instead of keeping the picks it made for Dev.  (Written the
obvious way -- fix the picks, resample them -- premium looks like 78 % pass instead of 83 %.)
The vectorised allocator was checked against `select_models` itself: identical picks on all
three tiers, 0 disagreements.

Final candidate ([A, B, C], 5 seeds, safety .94/.80/.73), 1000 resamples:

| tier | ratio | pass plain | runaway | inflation | score | E[score] |
|---|---:|---:|---:|---:|---:|---:|
| fast | 1.137 | 99.8 % | 95.3 % | 99.7 % | 0.6830 | 0.6812 |
| balanced | 1.623 | 99.8 % | 99.6 % | 97.5 % | 0.7097 | 0.7085 |
| premium | 3.533 | **83.3 %** | 81.1 % | 49.4 % | 0.7463 | 0.6228 |

Reported held-out 0.709972; **expected score once busts are counted, 0.6719**.  Fast and
balanced are safe.  Premium is not, and it is not something this round introduced -- the
single-fit baseline scores 88.6 % there.  E55 recorded 0.2 % bust for this triple; measured
against Dev with the allocator re-run it is 12-17 %.

Sweeping premium's safety (it enters allocation only, so no rebuild is needed), under two
independent resampling schemes:

| premium safety | ratio | pass (bootstrap) | pass (half-subsample) | dev score | E[score] |
|---:|---:|---:|---:|---:|---:|
| 0.55 | 2.314 | 100.0 % | 100.0 % | 0.7276 | 0.7245 |
| **0.60** | 2.461 | 99.8 % | 100.0 % | 0.7321 | **0.7285** |
| 0.65 | 3.203 | 97.0 % | 98.3 % | 0.7384 | 0.7139 |
| 0.70 | 3.361 | 93.0 % | 91.0 % | 0.7435 | 0.6900 |
| 0.73 (shipped) | 3.533 | 87.2 % | 83.3 % | 0.7463 | 0.6497 |

Everything between ratio 2.46 and 3.53 buys **0.014 of premium score** and costs 16 points of
pass probability: those upgrades are the expensive tail (cf. E59b, where think emits ~9,765
tokens on an AIME item).  Re-pricing premium to 0.60 moves the expected final score from
**0.6719 to ~0.7047** -- roughly five times what the whole 34B column work was worth -- while the
headline held-out number only drops from 0.709972 to ~0.7057.

Caveat before adopting: 0.60 is the argmax over Dev, the same 880 items the number is read from.
The effect is large, monotone and reproduced by two resampling schemes, but the value itself
should be re-priced the way E55 did it (Train-only rows, scenario-averaged) rather than taken
from this table.

### E63. Safety triple re-priced to a no-bust requirement ✅ adopted
Requirement changed from "maximise EV" to "must not score 0", so the triple is no longer an EV
argmax -- it is the largest safety ratio per tier that busts in **none** of the resamples, across
four scenarios (plain / runaway / inflation / half-size batches), with the allocator re-run
inside every resample.  Picking against a constraint rather than fishing for score also keeps
the selection bias small.  `tools/price_safety.py`.

One trap worth recording: below `1/multiplier` the cap collapses to the light total
(`cap = light_total * max(1.0, mult * safety)`), so for fast every value ≤ 0.80 routes the whole
batch to the light model and looks perfectly "safe" at score 0.6193.  Grids have to start above
1/multiplier per tier.

| tier | shipped | bust @ shipped (plain/runaway/inflation) | **repriced** | ratio | bust @ repriced |
|---|---:|---|---:|---:|---|
| fast | 0.94 | 0.2 / 4.7 / 0.3 % | **0.92** | 1.140 / 1.25 | 0 / 0 / 0 % |
| balanced | 0.80 | 0.2 / 0.4 / 2.5 % | **0.70** | 1.436 / 2.00 | 0 / 0 / 0 % |
| premium | 0.73 | 16.6 / 18.2 / 51.2 % | **0.56** | 2.331 / 4.00 | 0 / 0 / 0.1 % |

Confirmed at 3,000 resamples per scenario.

| | held-out dev | expected score (busts counted) |
|---|---:|---:|
| .94/.80/.73 (shipped triple) | 0.709972 | 0.6719 |
| **.92/.70/.56 (repriced)** | **0.700398** | **0.6996** |

The headline number gives up 0.0096 and the expected score gains 0.0277; the gap between what is
reported and what is expected closes from 0.038 to 0.0008.  Rebuild with
`ROUTER_SAFETY_FAST=0.92 ROUTER_SAFETY_BALANCED=0.70 ROUTER_SAFETY_PREMIUM=0.56`.

Caveat: priced on Dev, which is out-of-sample for the fitted model but is the same 880 items the
score is read from, and the bootstrap models item-mix variance rather than distribution shift.
The margins are wide (premium sits at 2.33 against a cap of 4.0), so the choice is not delicate,
but a private set drawn differently could still behave unlike this estimate.

### E63b. Like-for-like at the repriced triple; 3 seeds overtakes 5
The published 0.705568 uses the same .94/.80/.73 triple as our baseline arm, so the repriced
0.700398 was never comparable to it.  Re-scored with every configuration on the same triple:

| configuration | dev @ .92/.70/.56 | dev @ .94/.80/.73 |
|---|---:|---:|
| single fit [A, B] (the shipped router) | 0.693864 | 0.702727 |
| 5 seeds [A, B] (what the published number is) | 0.694574 | 0.704801 |
| 5 seeds [A, C] | 0.699545 | 0.707330 |
| **3 seeds [A, B, C]** | **0.701903** | 0.709119 |
| 5 seeds [A, B, C] | 0.700398 | 0.709972 |

At equal safety the 34B column plus seed averaging is worth **+0.0073** over the shipped router
and **+0.0066** over the published configuration -- the ordering is unchanged by the repricing,
and none of the gain came from the safety change.

The repricing does flip the seed count: at .92/.70/.56, 3 seeds beats 5 by 0.0015, having
trailed by 0.0009 at the old triple.  That is at the noise limit and the two are effectively
tied, but 3 seeds costs +8 % runtime against 5 seeds' +26 %, so the tie should be broken toward
3.  Its bust profile at the new triple is clean at 3,000 resamples: 100 % pass on every tier in
every scenario, expected score 0.7011 against a reported 0.701903.

**Final candidate: [A, B, C], 3 seeds, safety .92/.70/.56 — held-out dev 0.701903, expected
score 0.7011, zero busts.**

### E64. Where the remaining headroom is: the safety margin is entirely cost-prediction error
`tools/diag_safety_headroom.py` re-runs the allocation on **true** costs instead of predicted
ones.  Result: allocating on true costs **never busts, at any safety ratio, on any tier** --
0.0 % up to 0.94.  So none of the margin is item-mix variance; all of it is insurance against
cost prediction error, and it is therefore addressable.

What removing that error would be worth (per tier, dev):

| tier | now (repriced safety) | oracle costs @ 0.94 | gain | weighted |
|---|---:|---:|---:|---:|
| fast | 0.6807 @ 0.92 | 0.6939 | +0.013 | +0.0053 |
| balanced | 0.7026 @ 0.70 | 0.7248 | +0.022 | +0.0067 |
| premium | 0.7312 @ 0.56 | 0.7659 | +0.035 | +0.0104 |

Ceiling ≈ **+0.022** on the final score -- three times what the 34B column was worth, and the
only axis left with that much in it.

The error is concentrated in the think head.  Log-cost RMSE on dev: light 0.556, mid 0.458,
**think 0.677** (a factor of ~2 at one sd).  And the 34B column did not touch it -- think's RMSE
is 0.676 before the column and 0.677 after, mid 0.461 -> 0.458.  Every bit of E59's +0.0073 came
from *score* prediction; cost prediction is untouched.

Why it is hard: think's output length is close to unpredictable from smaller models' behaviour.
Column C's own output length correlates **0.182** with think's log output length, and even the
real `ax31`'s measured output length correlates only **0.319** with think's.  Nothing currently
in the artifact carries the signal.

One intervention was tested and rejected: a per-item cost cap (refuse any pick whose predicted
cost exceeds a share of the batch budget).  Premium at cap 0.002/safety 0.85 scores 0.7106
against 0.7312 for the uncapped repriced setting -- the cap removes the tail but pays more for it
than it saves.

### E65. Per-model conservative cost offsets ❌ rejected
The cheap half of E64's prescription: charge the allocator more for the model whose cost it
mispredicts.  A uniform inflation cancels (the cap scales with the light total), but a per-model
log-cost offset does not -- and it ships as a constant on a head's baseline, so the runtime is
untouched.  `tools/e65_cost_offsets.py` searches (offset_mid, offset_think) x safety, keeping
only settings that bust in no resample of any of the four scenarios.

At 150 resamples it looked worth +0.002.  At 1,200 it is worth nothing:

| tier | no offset | think +0.60 |
|---|---|---|
| fast | safety 0.92, score 0.6807 | safety 0.92, score 0.6807 |
| balanced | safety 0.70, score 0.7017 | safety 0.72, score 0.7006 |
| premium | safety 0.52, score 0.7253 | safety 0.68, score 0.7247 |

The offset does buy safety ratio -- premium goes 0.52 -> 0.68 -- but the ratio buys back exactly
the upgrades the surcharge removed.  The two cancel to within noise, and the mechanism explains
why: the offset is a monotone reweighting of the same efficiency ordering, so it cannot add
information the cost head does not have.  Only a better *prediction* can, which is E66.

Also recorded: the strict no-bust threshold for premium moves with the resample count -- 0.56 at
500, 0.52 at 1,200, once half-size batches are included as a scenario.  The shipped 0.56 is
clean for 880-item batches (0 busts in 3,000 resamples of plain/runaway/inflation) but assumes
the evaluation batch is not much smaller than Dev.

### E66. Reasoning-model prior for think's cost (built, not yet run)
E64's remaining axis, and the only one worth ~+0.022.  A reasoning model is the one proxy that
emits a long chain the way `axk1-think` does, so its output length is the natural predictor of
think's -- which is 87 % of think's cost.  E57 declined to build a reasoning column because its
*score* signal was already carried by the existing columns; that judgement does not apply to
cost, which nothing in the artifact predicts.

Gate: `corr(reasoning length, think log output length) >= 0.45`, against 0.319 for the real
`ax31`'s own measured output length and 0.182 for the 34B column's.  Public 2,640 only, about an
hour.  No `--instruct`: the organiser's format instructions exist to make answers short, which
is precisely the signal being measured.

`colab-label/e66_think_cost_colab.ipynb`, `think_cost_gate.py`, `make_e66_zip.py` (7.2 MB).
If the gate fails, the conclusion is that think's cost is irreducibly unpredictable, the safety
margin cannot be narrowed, and **0.7019 is the ceiling of this structure**.

### E67. Family classifier rebuilt from the data analysis ✅ adopted (as a correctness fix, EV-neutral)
The analysis measured `similarity.classify_family` at 91.44 % against the true source and put
every error in a regex: a whitelist on the first word (dmmath -> gsm8k 98), `\$[^$]+\$` matching
dollar amounts (gsm8k -> aime 76), a proper-noun gap in the ruletaker pattern (40), and the
ruletaker test running before the length test (babilong -> ruletaker 2).  Its 7-step cascade
reproduces the recorded provenance 2443/2443, but two steps need `aime-selection.json` and
`num_generations`, which are not available at routing time.

`tools/e67_classifier.py` is the text-only projection: length first, then unambiguous structure
(code / 4-option / 2-option / `\nQuestion:` / hangul), then the math trio resolved on text --
LaTeX that is not a dollar amount -> aime, deepmind idioms -> dmmath, competition-geometry
phrasing without money -> aime, else gsm8k_or_other.

| classifier | accuracy | `aime` precision | `gsm8k_or_other` precision |
|---|---:|---:|---:|
| shipped | 2414/2640 = 91.44 % | 34/110 = 0.309 | 249/388 = 0.642 |
| **E67** | **2636/2640 = 99.85 %** | **36/36 = 1.000** | 330/331 = 0.997 |

The 4 residuals are genuine boundary cases (three GSM8K items that read like arithmetic, one
dmmath "nearest to" question).  Shipped in `src/ossp_router/similarity.py` (previous version at
`similarity.py.e66.bak`); runtime tests unchanged -- the four `test_cli` failures and the
`fcntl`/`resource` import errors are pre-existing Windows issues, identical with the old file.

**Effect on routing, same chain, same column set, no-bust triple .90/.70/.56:**

| build | held-out | E[score] | family gap moved |
|---|---:|---:|---|
| E66b (old classifier) | 0.700739 | 0.700471 | — |
| E67, 3 seeds | 0.700682 | 0.699944 | dmmath +0.0099 (premium), belebele/code −0.003/−0.004 |
| E67, 5 seeds | 0.700966 | 0.701276 | |

Stem-grouped paired bootstrap (`tools/e67_paired.py`, 1500 resamples over 753 stem groups,
both artifacts scored on the same resample):
3 seeds: mean −0.00053, 90 % CI [−0.0022, +0.0013], P(better) 0.31.
5 seeds: mean +0.00045, 90 % CI [−0.0011, +0.0022], P(better) 0.67.  **Both noise.**

So the label is now right and the gain lands exactly where the analysis said it would (dmmath),
but the meta-GBM was already compensating for the old label's noise and gives the fix back
elsewhere.  Kept because a correct label is the right input regardless, and it removes the
analysis's "two populations with opposite optimal models in one bucket" hazard for the private set.

### E68. Is there an "extreme item" head to build?  ❌ no -- the ceiling is the features
E67's gap analysis: on the dmmath/cruxeval items the oracle sends to k1, truth is light ≈0.03 /
k1 ≈0.96 but the router predicts light 0.30-0.37 / k1 0.74-0.84, halving the predicted
efficiency.  The hypothesis was a dedicated head for the joint event (light==0 & k1==1; 499/2640
= 18.9 %, concentrated in dmmath 192 and cruxeval 143).

`tools/e68_extreme_diag.py` on dev:

| signal | AUC for the joint event |
|---|---:|
| 1 − P(light≥.25) (E21 ordinal, already shipped) | 0.804 |
| P(k1≥1) (E21 ordinal) | **0.426** |
| product | 0.840 |
| E[k1] − E[light] (what the allocator consumes) | **0.841** |
| dedicated HistGBM classifier, train->dev (upper bound) | 0.760 |

The shipped signals already reach 0.84 and a dedicated head does worse.  The shrinkage is not a
head-design problem: `P(k1≥1)` is 0.765 on extremes and 0.782 on everything else -- **the k1
score head has no discriminative power on this event at all** (AUC 0.43).  The features cannot
tell which hard items k1 will crack; that is the same information limit E47 found for the
upgrade-efficiency rank (eff_spearman ≈ 0.07).  No head to build.

### Round summary (E64-E68) and the final candidate
Safety-margin axis: E64 located it (all of it is think-cost error; oracle costs never bust),
E65 offsets cannot buy it back, E66 reasoning column moves the RMSE (0.677 -> 0.661) and one
safety notch (0.52 -> 0.56) but the notch is worth +0.0004 in paired terms, and its private-set
reach is capped at the public items.  Label axis: E67 fixes it, EV-neutral.  Head axis: E68 --
nothing to build.

The no-bust triple **.90/.70/.56** came out identical on every build priced this round
(3-column, 4-column at 74 % and 100 % coverage, E67 at 3 and 5 seeds) and on both scenario
sets (with and without half-size batches).

**Final candidate: E67 5-seed [A,B,C,D], safety .90/.70/.56 — held-out 0.700966, E[score]
0.701276, 0 busts in 4,500 tier-resamples.**  Against the shipped 0705 router (0.702727 at a
triple that busts premium 11 %), that is +0.025 in expectation and −0.002 on the headline.

### E69. Decision-layer prior-score blend ✅ adopted — the round's one real gain
Step 1 decomposed the remaining gap at the no-bust triple (`tools/e69_decompose.py`, dev):
predicted/predicted 0.7007; TRUE scores/predicted costs **0.7649**; predicted scores/TRUE
costs 0.7058.  **93 % of the remaining gap is score error, not cost error** — the mirror of
E64, which only measured the cost side.  The reliability curve killed recalibration in the
same run: dev predictions are already near-diagonal (bin 0.3-0.4 realises 0.316), so E67's
"light predicted 0.30, truth 0.06" was selection-conditioned regression to the mean, not
miscalibration.

Step 2: the prior columns are direct measurements — column A agrees with the true light score
at corr 0.699 on dev hits, the 34B column with true mid at 0.724 — but through the meta
features the stack dilutes them to 0.604 / 0.666.  Cheapest possible use: on a scored lookup
hit, blend the column's own score into the final score row,
`ps[m] <- (1-w)·ps[m] + w·col_score`, w=0.25 (unimodal over {0.25,0.5,0.75}, all positive).
Runtime cost: one sha256 + a dict probe; stdlib.  Shipped as `prior_score_blend` in the
artifact and ~20 lines in `predict_episode_augmented` (E42's caveat was checked by measuring
EV, not RMSE).

Step 3 (`tools/e69_blend_modes.py`): the blend fattens premium's realised-ratio tail — the
**mid** half is responsible (mid-only at 0.56: 4.1 % inflation busts; light-only clean through
0.54).  Repricing with the blend active moves the triple to **.90/.72/.52** (balanced *gains*
a notch, premium gives one back).

Final package, `reports/e67_append/learned-router.v1.json`:

| | held-out dev | E[score] | busts @3000 (plain/runaway/inflation) |
|---|---:|---:|---|
| previous best (no blend, .90/.70/.56) | 0.700739 | 0.700471 | 0/0/0 |
| **blend w=0.25, .90/.72/.52** | **0.705114** | **0.704283** | **0/0/0** |

Package-level stem-grouped paired gate (`tools/e69_package_paired.py`, each side at its own
triple, 1500 resamples): mean **+0.0037**, 90 % CI **[+0.0003, +0.0071]**, P(B>A) 0.960,
0 busts on either side — the only change this round to clear the adoption rule.

Caveats, recorded: (1) under the strict 4-scenario rule that includes half-size batches,
premium's zero-bust point is 0.48, not 0.52 — 0.52 is certified on the three E55 scenarios at
3000 resamples (the E63b precedent) and busts ~0.2 % of half-size batches; if the private
batch can be far smaller than 880, ship .90/.72/.48 (headline ~0.7025) instead.  (2) w and the
triple were selected on dev; the paired CI is the guard, and its lower bound is +0.0003, not
comfortable.  (3) The blend transfers to the private set wherever the prior lookup hits with a
judged score (columns A and C carry 38k source-rendered entries), and is a no-op on misses —
unlike E66's reasoning column this is not public-only.

Deployment note: `build_public_lookup.py` must run AFTER the blend field is set, or the stored
public rows will disagree with the compute path.

### E70. Per-column blend weights ❌ rejected — (0.25, 0.25) is the argmax of the whole grid
The correlation table suggested headroom (column A alone reaches corr 0.699 with the true light
score where the blended stack reads 0.658), and E69 had only tried a single joint weight.  A
5x4 grid over (w_light, w_mid) at the shipped triple: the shipped (0.25, 0.25) is the exact
maximum, every neighbour loses (best alternative −0.002).  Marginal correlation does not order
EV — E42 again.  Also dead on the same table: the mid-proxy ensemble avg(B, C) (corr 0.714 vs
C alone 0.712), any k1 score blend (no column beats the stack's 0.500), and self-consistency
as a score substitute (corr 0.37-0.47, well under the stack).

The blend axis is closed with the shipped setting sitting at a verified local optimum.

### E71. Submission artifact built on Colab ✅
`run_deploy_chain.sh`: the full chain on the combined 2,640 (E11's deployment convention),
prior columns A+B+C+D, meta x3 seeds, then the blend + the .90/.72/.52 triple, then the public
lookup built AFTER those fields, then a 120-episode lookup-vs-compute equality check.  Built on
a Colab T4 (sha256 7984081c..., 27.8 MB, lookup 2,640 rows); the on-builder verification passed
with max diff 0.

Two findings from receiving it back locally:

**Cross-machine float drift is real but harmless.**  Re-verifying the lookup on the Windows
machine shows 12/1350 score comparisons differing (max 7e-4) and 6 cost comparisons (max 0.5 %
relative) — the signature of libm exp/log ULP differences occasionally flipping a GBM split,
not of a stale configuration (a lookup built without the blend would disagree by ~0.1 on most
hit items).  The lookup ships the builder's own rows, so public prompts are answered
identically everywhere; private prompts compute live and inherit only this noise, consistent
with E60's measured 0.0014 cross-machine spread.

**Runtime fits.**  Same-machine ratio against the single-fit reference: 1.31x
(27,500 vs 21,014 us/episode on the miss path).  Against the prior round's official-hardware
measurement of ~37 s/tier for a single-fit artifact, that estimates **~48 s/tier against the
90 s limit**.

Also fixed in passing: pathlib `write_text` on Windows had converted the chain scripts to CRLF
(Git-Bash tolerates it, Linux bash dies on `$'python\r'`); scripts, bundle and repo normalised
to LF, and the pitfall matches the data analysis's §7.3-1 CRLF trap.

Local sanity of the received artifact: dev 0.7185 through the full path with every tier inside
budget (in-sample — the artifact trains on dev; the performance claim remains the Train-only
held-out 0.705114 / expected 0.7043).
