<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# LAB BRIEF 2 — corrected facts after the 14-agent analysis round

Read `BRIEF.md` first for the task definition, then this file, which **corrects
several load-bearing claims in BRIEF.md**.  Everything here was measured by the
analysis fleet (reports in `reports/lab/a01..a14*.md`) or by the orchestrator.

## 1. Corrections to BRIEF.md

| BRIEF.md said | corrected |
|---|---|
| held-out dev 0.7017 | 0.701648 at safety **.98/.89/.88**; the *deployed* triple .98/.87/.85 gives **0.700483**.  Always state the safety triple with the score. |
| "noise-free ceiling ~0.814" | wrong.  a14: honest ceiling **0.790** (noise inflates the realised-score oracle by only +0.013); a02 computes **0.7445** for perfect *item-level* knowledge under the deployed allocator.  Treat 0.74–0.79 as the range, not 0.81. |
| exchange rate: corr .42→.48 buys 0.72 | wrong axis.  Blending toward the realised label overstates by ~1.8x, and the axis itself is invalid (see §2). a07's honest figure: **+0.0025 final per +0.01 score corr**. |
| "cost calibration is worth +0.0053" | that was oracle-tuned on dev.  Estimated out-of-fold it is **negative** (E44: cvEV 0.7022 vs 0.7027).  Only the *variance-form* and *k1-relative-price* variants survive (§3). |

## 2. The axis that matters: gains, not levels

The allocator picks `argmax_m (s_m − λ·c_m/L)`, so **adding a constant to all
three of an item's predicted scores changes nothing**.  Only the two gains
`d1 = s_mid − s_light` and `d2 = s_k1 − s_mid` carry decision information.

- 69–81 % of the score variance — and essentially all of the reported
  "score correlation" — lives in the invariant level channel (a03).
- In a noise-regenerated simulation, a perfect level buys **+0.009** final; a
  perfect set of gains buys **+0.078** (a03).
- `d1` is **at chance within family** (AUC 0.545) yet worth **+0.052** final if
  solved — six times the entire cost side (a05).
- 86 % of the 0.1018 dev gap to the oracle is real, not label noise, and 98 % of
  it is score-side (a05).
- **`corr(prediction, realised score)` is banned as a proxy objective.** a06
  built a head that reaches the BRIEF's corr-0.48 target and it *lost* 0.016
  point / 0.037 bootstrap EV.

## 3. The risk side: the deployed safety ratios are mispriced

Five agents independently re-priced them against honest (train-only) predictions
with the project's own 880-item bootstrap:

| safety triple | bust % fast/bal/prem | E[final] | dev point |
|---|---|---:|---:|
| .98/.87/.85 (deployed) | ~6 / 2 / **14** | **0.647–0.651** | 0.7005 |
| .98/.89/.88 (the "0.7017" run) | ~7 / 2 / 16 | 0.617 | 0.7017 |
| ~.93–.96 / .80–.84 / .74–.78 | ~0 / 0 / 0–1 | **0.69** | ~0.695 |

E39/E43b reported 0.0/0.2/0.2 % because those bust curves were computed on
predictions that had already seen the evaluation episodes.  The premium tier is
a coin-flip-with-14 %-tails, not a safe bet.

**Consequence for every experiment from now on: report BOTH**
`EV` (bootstrap expected final score, bust priced in) **and** `dev` (the single
held-out sample), each with the safety triple that produced it.

## 4. The honest protocol (use it, do not invent another)

`experiments/lab/bench2.py`:

1. 10-fold OOF predictions over **Train only**; the legacy 256-bin head is
   **refitted inside every fold** — see §5, this matters a lot.
2. per-tier safety = argmax of the 3-seed x 400-sample bootstrap EV on those OOF
   rows over a wide grid.  Dev is never read at this step.
3. Train-only refit -> Dev scored once at the chosen safety.

```python
from harness import Lab
import bench2 as B
lab = Lab(); cv, arr = B.stage(lab, exp, tag="mytag")
B.run(lab, cv, arr, cfg, transform=my_transform, label="...")
```
`experiments/lab/protocol.py` also provides `exact_allocate` and `safety_curve`
— a concave-envelope allocator that reproduces the deployed bisection exactly
(verified, 100 % pick agreement) and evaluates a whole safety grid from one sort,
~100x faster.

## 5. A real defect found in the deployed pipeline (already fixed in the lab)

`build_meta_gbm.py` feeds the meta-GBM the legacy 256-bin predictions of the
*shipped* artifact.  That artifact was fitted on all 1,760 Train episodes, so
those features are **in-sample for every training row** (score corr **0.60**) and
out-of-sample at prediction time (corr **0.39** on train-OOF, **0.38–0.45** on
dev).  The tree ensemble therefore learns to over-trust them.

Refitting the legacy head out-of-fold for the meta-GBM's training rows
(`exp["legacy_oof_meta"]=True`) gives, under §4:

| config | EV | safety | dev |
|---|---:|---|---:|
| baseline | 0.661181 | .960/.840/.735 | 0.697642 |
| **legacy-OOF meta** | **0.670729** | .960/.825/.840 | **0.699915** |

+0.0095 EV, +0.0023 dev, and it lets the premium safety rise from .735 to .840
at 0 % bust because the cost predictions are better calibrated.

## 6. Confirmed candidate improvements from round 1 (to be composed and re-tested)

| id | idea | reported effect | source |
|---|---|---|---|
| C1 | legacy head refit out-of-fold for the meta features | EV +0.0095 / dev +0.0023 | orchestrator (§5) |
| C2 | repair 2 family regexes (RuleTaker, DM-Math rescued from `gsm8k_or_other`); 7.0 % of items are mislabelled | CV EV +0.0018, 3/3 seeds, and pushes optimal safety **up** .93→.98 | a08 |
| C3 | sub-family split (`aime` is 2/3 GSM8K money problems; `gsm8k_or_other` hides DM-Math), correcting score AND cost | +0.0038 weighted bootstrap EV at zero bust | a06 |
| C4 | variance-form cost re-transformation `c *= exp(kappa*sigma^2)` used in BOTH the decision and the cap accounting | +0.0031 ± 0.0014, 5/5 cross-fit splits | a09 |
| C5 | per-model relative-price correction on the k1 cost | +0.0019 weighted EV, 39/40 split-halves | a11 |
| C6 | per-family affine reconstruction of the k1 score from the predicted mid score (18 train-only constants) | premium 0.7366→0.7418 at constant spend, final 0.7005→0.7020, premium bust .130→.068 | a12 |
| C7 | re-price the safety triple on honest predictions | E[final] +0.04, dev −0.006 | a01, a10, a13 |
| C8 | hard ceiling on the cost head | +0.0034 | a07 |

C4/C5/C8/C7 all act on the same mechanism (budget-ratio dispersion) and will not
simply add.  C2/C3 are both family-partition repairs.  Composition must be
re-measured, not assumed.

## 7. What did NOT work (round 1, do not repeat)

- Per-model / per-family cost sum calibration estimated out-of-fold (E44).
- Gain pair-balance scaling and gain shrinkage toward the family mean (E45).
- Coordinate descent on the 8 post-hoc constants using CV EV — it walks straight
  into configurations that bust on dev (E46), because CV with in-sample legacy
  features is optimistic.
- Per-item cost-uncertainty inflation, conformal cost bounds, post-hoc repair
  passes (a11 proves a repair pass is the deployed allocator with a different
  stopping test), abstention on kNN similarity (a04: kNN top-1 similarity is
  *anti*-correlated with accuracy).
