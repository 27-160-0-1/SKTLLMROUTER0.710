<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# LAB BRIEF — shared context for every analysis/design agent

**Goal:** raise the held-out dev final score from **0.7017** to **0.72**
(prompt-only LLM routing challenge).  Read this before doing anything.

## 0. Ground rules for agents

- Working dir: `C:\portable\skt_LLM1\LLM-ROUTE-0.7000`.  Python:
  `.venv\Scripts\python.exe` (numpy/scipy/sklearn/cupy/torch-cu124 available).
- **Read-only on the repo** except: your own report at
  `reports/lab/<your-id>.md` and your own scripts at
  `experiments/lab/<your-id>_*.py`.  Never touch
  `src/ossp_router/resources/*.json`, `tools/*`, or another agent's files.
- Use `experiments/lab/labdata.py` (numpy view of the data + an exact copy of
  the deployed allocator + scorer).  `reports/lab/dev_preds_e43.npz` holds the
  deployed pipeline's per-episode predicted scores/costs on dev
  (`score_<tier>`, `cost_<tier>`, each (880,3), model order
  `ax31-light, ax31, axk1-think`).
- Prior work is in `EXPERIMENT_LOG.md` (E01..E43, 43 experiments, 4 adopted).
  **Read the entries relevant to your topic before proposing anything** — most
  obvious ideas have already been tried and rejected, and the log records why.
- Judgement standard used by this project: 5-fold nested CV + 880-item
  bootstrap EV, 3 seeds, unimodal curve.  Gains < 0.0005 are noise.

## 1. The task

Per episode, pick one of `ax31-light` / `ax31` / `axk1-think` from the prompt
text alone.  Three tiers are scored independently and summed with weights:

| tier | budget multiplier | weight |
|---|---:|---:|
| fast | 1.25x | 0.4 |
| balanced | 2.0x | 0.3 |
| premium | 4.0x | 0.3 |

`budget_ratio = total_cost_of_selection / cost_if_everything_were_light`.
**Over budget => that tier scores 0.**  `cost = (in_tok*r_in + out_tok*r_out)/1e6`
with `r_in/r_out` = 1/4 (light), 2.127/8.509 (mid), 6.565/26.26 (k1).
Token counts in the data are already summed over `num_generations`.

## 2. Data facts (measured, `experiments/lab/diag*.py`)

- train 1,760 / dev 880 episodes.  9 source families (regex classifier in
  `similarity.classify_family`): belebele, code(CRUXEval), dmmath, gsm8k_or_other,
  ruletaker, truthfulqa, longdoc(BABILong), aime, hrmcr.
- mean score per model — train `[0.597, 0.679, 0.812]`, dev `[0.619, 0.692, 0.826]`.
- mean cost ratio vs light — train `[1, 2.16, 23.2]`, dev `[1, 2.10, 23.8]`.
- `score` is `k/num_generations`, `num_generations` in {2,4} (4 = all AIME +
  ~60% of gsm8k_or_other, ~0 elsewhere).  So the label is a **2- or 4-sample
  binomial estimate of the latent success probability p** — very noisy.
- Cost anatomy: input tokens are ~50% of light/mid cost but only **12.5% of k1
  cost**; k1 output length is the dominant cost term (median 680/gen,
  p95 5,668, max 32,711 = context limit).
- Input tokens are nearly identical across the three models (corr>0.9998) and
  are **highly predictable from cheap text statistics**: an 11-feature linear
  fit + family one-hot gives dev R2=0.9985, median APE 5.2%, sum ratio 0.995.
- No duplicate prompts anywhere in train+dev.
- Per-family mean scores (dev) `[light, mid, k1]` and cost ratio:
  `aime .657/.721/.884 (1/1.4/29.8)`, `belebele .823/.870/.935 (1/2.8/37)`,
  `code .483/.521/.853 (1/3.1/150)`, `dmmath .413/.570/.848 (1/2.4/24)`,
  `gsm8k .696/.752/.912 (1/2.2/20)`, `hrmcr .000/.075/.075 (1/2.2/93)`,
  `longdoc .506/.525/.544 (1/2.1/10.6)`, `ruletaker .694/.766/.778 (1/2.7/35.5)`,
  `truthfulqa .705/.843/.904 (1/1.8/14)`.

## 3. Where the gap is (measured on dev, `diag1/5/6.py`)

| configuration | final | note |
|---|---:|---|
| deployed E43 (held-out) | **0.7017** | fast .6759 / bal .6994 / prem .7384 |
| pred score, **true cost**, safety 1.0 | 0.7101 | +0.0084 from cost alone |
| pred cost, **best safety** (dev-tuned) | 0.7063 | +0.0046, oracle-tuned |
| per-model cost sums calibrated + best safety | 0.7070 | +0.0053 |
| true score + true cost (realised-score oracle) | 0.8034 | inflated by label noise |
| allocate & evaluate on latent p (noise-free ceiling) | ~0.814 | true prompt-only ceiling |
| family-mean scores + true cost + safety 1.0 | 0.6987 | what family averages alone buy |

Score prediction quality today (dev, per-model corr with realised score):
**0.42 / 0.49 / 0.42**.  Gain corr: mid-light **0.107**, k1-mid 0.357.
Cost: log-error sd 0.57/0.48/**0.72**; predicted sums are 0.82/0.88/**0.66** of
true (Jensen bias from `exp(E[log c])`).

**Exchange rate (the most important number).**  Blending the current score
predictions toward the truth by lam and re-tuning safety:

| lam | corr | final |
|---:|---|---:|
| 0.00 | .42/.49/.42 | 0.7063 |
| 0.10 | .57/.62/.58 | 0.7436 |
| 0.25 | .75/.78/.77 | 0.7702 |

The metric is **steeply** sensitive to score-model quality near our operating
point: reaching 0.72 needs roughly corr .42 -> .48.  (Caveat: blending toward
the *realised* score also injects exploitable label noise, so the real exchange
rate for an honest E[s] improvement is less favourable — quantifying that
honestly is itself a task.)

## 4. Hard lessons already paid for (do not re-derive)

- **E42 (critical):** improving the score head *alone* LOWERS the final score.
  Better score predictions make the allocator upgrade items whose cost is
  under-predicted (selection-induced cost bias), so the bust probability rises
  and the safety factor is pushed down.  Any score-side proposal must come
  with a cost-side answer.
- **E37:** the Lagrangian allocator is provably optimal given the predictions
  (ILP integrality gap ~= 0.0004).  Do not propose a different allocator shape
  unless you change what is being fed to it.
- **E32 / E36 / E37-CP:** per-item cost-uncertainty inflation, conformal cost
  upper bounds and heteroscedastic penalties all lose to the single global
  safety scalar.
- **E36 / E36b:** cost log-RMSE for k1 (0.65 = +-1.9x) could not be reduced by
  quantile heads, smearing, sparse ridge or token-level reconstruction.  The
  unexplained part of think output length looks intrinsic.
- **E39 / E39b:** the deployed safety ratios are the CV optimum; raising them
  (spending more) loses at every level because the extra budget buys only a
  handful of upgrades while the bust cliff is steep.
- **E19 / E22 / E28 / E25:** feature-space expansion (more dense features,
  bigger hash spaces, word 3-grams, supervised vocabulary, sentence-embedding
  distillation) is exhausted on this data.
- **E40:** label-preserving text augmentation adds no label information.
- **E41:** self-labelling with A.X-3.1-Light (6,718 new items, 82% agreement
  with the official labels) did not improve the light head.
- **E34/E35/E38:** IPR, RouteLLM (4 variants) and LLMRouter (6 variants)
  rebuilt on this data all lose to the deployed pipeline, mostly because they
  assume a fixed per-model price and a binary strong/weak choice.

## 5. Current architecture (deployed, E43 constants)

```
prompt text
 [0] SHA-256 public lookup (train+dev memorised; allowed by the rules)
 [1] linear ensemble: official hash-regex 256-bin *0.9  (+) ridge on 16,414
     hashed dims *0.1   -> 6 outputs (3 scores, 3 log-costs)
 [2] + family mean blend 0.15 (9 regex families)
 [3] + kNN over hashed char tf-idf, k=16, weight = top1_similarity*0.25
 [4] meta HistGradientBoosting stack on 58 features
     (dense30 + family9 + legacy6 + linear6 + knn7):
       - 12 ordinal heads P(s>=.25/.5/.75/1) per model -> E[s]
       - 2 gain heads (s_mid-s_light, s_k1-s_mid), alpha 0.5
       - 2 rank-efficiency heads, beta 0.4
       - 6 regression heads (3 score, 3 log-cost)
     tier blend fast .6 / balanced .45 / premium .3
 [5] Lagrangian bisection allocation under budget * safety (.98/.87/.85)
```
Runtime container: `linux/arm64`, **2 CPU, 2 GiB, 90 s per tier**, no network,
read-only rootfs, output <=4 MiB.  Today the runtime is stdlib-only Python
(~7 s/tier under QEMU) but the rules permit installing public dependencies at
build time (numpy, tokenizers, small redistributable language models are all
explicitly allowed) — so there is a lot of unused compute headroom.

## 6. How to run things

```powershell
# full honest held-out evaluation (train-only training, dev scored, ~225 s)
powershell -File tools\run_holdout_local.ps1
# constants are read from ROUTER_* env vars, see the script header
```
`experiments/e43_joint_sweep.py`, `experiments/e39_safety_margin.py` etc. are
the previous experiment harnesses (5-fold nested CV + bootstrap EV).

## 7. Hardware available

i9-13900H (14C/20T), 31.7 GB RAM, **RTX 4090 Laptop 16 GB VRAM**, CUDA 12/13,
580 GB free disk, network available for training-time downloads.  This is far
more than the previous machine (RTX 2050 4 GB) — GPU work that was
infeasible before (running A.X-3.1-Light 7B locally, fine-tuning encoders,
large sweeps) is now cheap.

## 8. Rules that matter for strategy (docs/CHALLENGE_RULES.md)

Allowed: classifiers, regression coefficients, vocab/IDF, tokenizers, **lookup
tables and search indexes built from public data**, exact-prompt or
prompt-hash lookup against public data, general-purpose dictionaries and
redistributable **small language models** running offline inside the resource
limits.  Public Train/Dev prompts + outcomes + the public cost policy may be
used for training and tier-policy optimisation.

Forbidden: calling the three evaluation models or comparing generations,
re-deciding after a choice, using `challenge_id` / `split` / `episode_id` /
input order, non-public evaluation material, any network call at evaluation
time, source/image mismatch.

Note the source benchmarks are all public (AIME24/25, Belebele-ko, CRUXEval,
GSM8K, HRMCR, RuleTaker, TruthfulQA, BABILong, DeepMind-Mathematics) and
`colab-label/build_pool.py --verify` already proved the official prompt
templates are **exactly reproducible** from those public sources — so a
prompt-hash lookup built over the full public datasets would hit unseen
evaluation items drawn from the same benchmarks.
