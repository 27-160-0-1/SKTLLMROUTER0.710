<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# SKT LLM Router — held-out dev 0.7100

A prompt-only router for the SKT Efficient LLM Routing Challenge: for each episode it picks one
of `ax31-light` / `ax31` / `axk1-think` under a per-tier cost budget, using only the prompt text.
The runtime is pure Python standard library — hashed linear heads → family/kNN → meta GBM →
Lagrangian allocation — and performs no model inference at evaluation time.

This repository is the **0.7100 build**. Read [the limits](#limits-read-this-before-quoting-the-number)
before quoting that figure: it is a conditional number, and the configuration that produces it
carries a real risk of scoring **zero** on the premium tier.

## Scores

Held-out protocol: rebuild on Train (1,760 episodes) only, score Dev (880) once, public lookup
stripped so every prompt goes through the full path. `run_repo_chain.sh` does this end to end.

| build | held-out dev | expected score¹ | premium bust risk |
|---|---:|---:|---:|
| previous release (`LLM-ROUTE-0.7000` line) | 0.702727 | 0.6760 | 11.4 % |
| **this build** — 34B prior column + seed-averaged meta heads | **0.709972** | 0.6719 | **16.6 %** |
| the same build at a no-bust safety triple ([below](#the-safe-variant)) | 0.701903 | **0.7011** | **0 %** |

¹ Expected score counts the zeros: a tier that exceeds its budget multiplier scores 0 outright,
so the expected value is `mean(score of resamples that stayed inside budget, 0 for the rest)`.
Measured by `tools/bust_probability.py` over 3,000 bootstrap resamples with the allocator re-run
inside each one.

All three rows were measured on the same machine with the same chain. Cross-machine comparison
is not valid — see [reproducibility](#reproducibility).

## What produced the +0.0073

Two changes over the previous release, both measured on the repository's own GPU chain:

**1. A real `skt/A.X-3.1` (34B) prior column.** The offline difficulty prior previously used
`Qwen2.5-14B-Instruct` as a stand-in for `ax31`, agreeing with it at `corr 0.612`. Running the
organiser's actual 34B model offline over the public benchmark sources and keying the result by
prompt hash raises that to **`corr 0.699`**, and to **0.709** once coverage was completed.
CHALLENGE_RULES permits lookup tables built from public data and offline use of publicly-weighted
models; nothing is inferred at evaluation time.

Coverage of the challenge's own items went 0.753 → **0.975**, and the column now contains every
digest of both previously shipped columns. All 36 organiser AIME episodes are covered, up from 0
— `colab-label/build_pool_aime.py` renders AIME from its upstream sources and proves the
rendering by hashing against those 36 (35/36 reproduce exactly).

**2. Seed-averaged meta heads, actually implemented.** The previous round recorded seed averaging
as adopted, but `tools/build_meta_gbm.py` fitted each head once. Since a HistGradientBoosting
prediction is `baseline + Σ(leaf values hit)`, the mean of N models is exactly
`mean(baseline) + Σ over all N models' trees of (leaf value / N)`; the trees are concatenated with
leaves scaled by 1/N and the runtime evaluator is untouched. Worth **+0.0021** on its own.
`ROUTER_META_SEEDS` controls it (default 5; `=1` reproduces the old export byte for byte).

## Limits — read this before quoting the number

**The premium tier busts about one run in six.** At the shipped safety triple `.94/.80/.73`,
premium uses 3.53 of an allowed 4.0 and exceeds it in 16.6 % of bootstrap resamples (18.2 % with
an injected runaway episode, 51.2 % under a cost-inflation stress). A busted tier scores 0, so
the honest expectation is **0.6719**, not 0.7100. The gap between the headline and the
expectation is **0.038**.

This risk is inherited, not introduced: the previous release busts premium 11.4 % of the time at
the same triple. But this build spends slightly more of the premium budget and so busts slightly
more often.

**The prior's coverage does not transfer in full.** 0.975 of Dev is covered partly because the
build feeds `bundle/public_all.jsonl` — the public prompts themselves — as the deployed chain
always has. An unseen private prompt is only covered through the source-rendered pool
(`bundle/ext.jsonl`, 30,921 items). Expect the private-set contribution to be smaller.

**Constants are not virgin with respect to Dev.** The blend weights, gain α, rank β and the
safety triple were fixed in earlier rounds that used Dev and CV. Only the model fit in this
build is Train-only.

**Cost prediction is untouched.** All of the gain is score prediction. Log-cost RMSE on dev:
light 0.556, mid 0.458, think **0.677** — identical to the previous release (0.676). Since
allocating on *true* costs never busts at any safety ratio, the entire safety margin is insurance
against that error, and it is worth roughly **+0.022** to remove. Nothing here removes it.

**AIME coverage bought nothing.** The 36 organiser AIME episodes now have a prior, and it changed
no routing decision. Upgrading Dev's 12 AIME items to `axk1-think` would cost **4.13x the entire
budget** for +0.0097 of score — `axk1-think` emits ~9,765 output tokens on an AIME item. The
allocator refuses them, and is right to.

**Runtime margin is thinner.** Seed averaging multiplies the exported meta trees by the seed
count: 1,296 → 7,599, artifact 15 → 25 MB, +26 % per-episode on the lookup-miss path
(`tools/time_heavy_path.py`). Tree evaluation is not the bottleneck — the kNN posting scan is —
but against the previous round's estimate of 40-50 s per tier on the official Apple Silicon
hardware this implies roughly 50-63 s against the 90 s limit. **Not verified on arm64.**

## The safe variant

Safety ratios enter allocation only, never the fitted model, so switching costs nothing but a
field:

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path("src/ossp_router/resources/learned-router.v1.json")
a = json.loads(p.read_text(encoding="utf-8"))
a["tier_safety_ratios"] = {"fast": 0.92, "balanced": 0.70, "premium": 0.56}
p.write_text(json.dumps(a), encoding="utf-8")
PY
```

That triple is the largest per tier that busts in **no** resample of any of four scenarios
(plain / injected runaway / cost inflation / half-size batches), confirmed at 3,000 resamples.
It reports 0.701903 and expects 0.7011 — a headline 0.0096 lower, an expectation 0.0277 higher,
and the reported-versus-expected gap closes from 0.038 to 0.0008.

Which to ship depends on whether a one-in-six chance of a zero is acceptable. If it is not, use
the safe triple. `tools/price_safety.py` re-derives it for any artifact.

One caveat on that triple: its strict no-bust threshold moves with the resample count once
half-size batches are included (0.56 at 500 resamples, 0.52 at 1,200). At 0.56 it is clean for
880-item batches — 0 busts in 3,000 resamples of the other three scenarios — but it assumes the
evaluation batch is not much smaller than Dev.

## Reproducibility

```bash
# 4 arms: baseline / [A,C] / [A,B,C] / [C] alone
ROUTER_META_SEEDS=5 bash run_repo_chain.sh append
```

Requires a GPU for the linear head (`cupy`); `tools/cpu_shim_train.py` substitutes scipy's LSMR
and was verified to give a bit-identical held-out score.

**The published 0.705568 of the previous round is not reproducible from that repository.** Its
own tools, constants and data give 0.702727 — the missing 0.0028 is the seed averaging that was
recorded as adopted but never implemented, which this build restores.

**The chain is not deterministic across hardware.** The identical baseline build gives 0.702727
on an RTX 2050 and 0.704148 on a Colab GPU — a 0.0014 spread from GPU and library differences
alone, as large as the noise limit. Only compare numbers produced on one machine.

## Layout

| path | contents |
|---|---|
| `src/ossp_router/` | the runtime (standard library only) and the shipped artifact |
| `run_repo_chain.sh` | the full build chain, Train-only, Dev scored once |
| `tools/price_safety.py` | re-derive a no-bust safety triple |
| `tools/bust_probability.py` | per-tier pass probability and expected score |
| `tools/diag_safety_headroom.py` | what the safety margin is insuring against, and its price |
| `tools/holdout_by_family.py` | per-family, per-tier breakdown of two artifacts side by side |
| `tools/time_heavy_path.py` | per-episode cost of the lookup-miss path |
| `tools/splice_prior_column.py` | add or replace a prior column without a refit |
| `colab-label/` | the offline labelling pipeline and its Colab notebooks |
| `EXPERIMENT_LOG.md` | every experiment, including the rejected ones and why |

The label pools (`colab-label/bundle/`, `colab-label/out/`, several hundred MB) are not in the
repository; `colab-label/build_pool*.py` regenerates them from the pinned public sources, and
`colab-label/prior_column_c.json` carries the compiled 34B column so the prior can be rebuilt
without them.

## What would move it further

The one axis with real headroom, measured: **predicting `axk1-think`'s cost**. Its output length
is 87 % of its cost and nothing in the artifact predicts it — the 34B column's output length
correlates 0.182 with think's, and even the real `ax31`'s measured length reaches only 0.319.

A reasoning model is the natural proxy, and a pilot over the public 2,640 with
`DeepSeek-R1-Distill-Qwen-14B` reaches **corr 0.628**. That is the input a narrower safety margin
would need. `colab-label/e66_think_cost_colab.ipynb` runs it; the ceiling on that axis is about
**+0.022**, three times what this release gained.

Closed directions, with evidence in `EXPERIMENT_LOG.md`: MLP and embedding heads, fine-tuned
encoder distillation, nine external routers, isotonic gain calibration, text augmentation,
self-labelling, side-information features, cost-uncertainty inflation, tail-quantile costs,
per-item cost caps, and per-model conservative cost offsets.
