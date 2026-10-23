# V5 — user/task-grouped LambdaRank, prepared but not submitted

## Decision and preservation

Selected **standalone LightGBM LambdaRank with Top-5 truncation** improves exact
local MAP@5 on both frozen holdouts. V5 merits a future explicitly authorized
attempt, not a claim about an unobserved Kaggle score. V3 remains the best actually
submitted solution; V4 was not submitted. No Kaggle request, PDF or publication
was made in this stage.

`artifacts/stage6/frozen_prior.json` protects **227 prior files**, including V1–V4
CSVs, manifests, model/configuration records, experiments and monitoring evidence.
Checks confirm they remain unchanged. The original ALS, CBF and popularity methods
and separate evaluations remain available. No raw preprocessing or candidate
experiments were repeated.

## 1. V4 audit: labels already correct, objective is classification

Inspection of the actual V4 model confirms `HistGradientBoostingClassifier` with
**binary log loss**, per-query-size weighting and positive-class weight ×10.
Its binary labels already represent membership of the user's **five most frequent
≥80%-completed programmes**, not every programme completed ≥80%. The hypothesis
therefore concerns ranking loss, not another label repair.

The same 59 cutoff-correct features and cached candidate pools are used. There
are no user/item identity columns, future user views or target-derived predictors.
All preprocessing/model values in rolling folds were built at those folds' own
cutoffs; later-trained ALS/CBF values are not substituted.

| Training task | History cutoff | Label window | Candidate rows | Positives | Zero-positive groups |
|---|---|---|---:|---:|---:|
| A | 23 March | [23 March,6 April) | 252,933 | 4,408 | 135 |
| B | 6 April | [6 April,13 April) | 260,359 | 3,152 | 256 |

**2,520 groups**, 513,292 rows and 7,560 positives. Each group is **user × forecast
task**; the same user in two tasks is two queries. Cached rows are verified
contiguous, programme IDs unique within query, group sizes positive integers
summing exactly to row count, and labels binary with at most five positives.
Query sizes range 200–247. All training labels end **13 April**, before both the
earlier evaluation `[13 April,18 May)` and main evaluation `[18 May,27 July)`.

Zero-positive training groups provide no informative LambdaRank pairs; they are
retained, and all **1,260 evaluation users**, including empty targets, remain in
MAP. Cold fallback and repeat eligibility are unchanged.

### Ranking headroom, without changing retrieval

V4 candidate-ranked micro recall:

| Window | @20 | @50 | @200 | Missed relevant inside pool below5 | Absent pool |
|---|---:|---:|---:|---:|---:|
| Main | .68332 | .78809 | .87067 | 2,393 | 762 |
| Earlier | .72634 | .82590 | .88881 | 2,244 | 627 |

The fixed union pool has recall **.87185/.88952**, mean size 222.05/219.29.
Ranking remains the larger bottleneck. Selected LambdaRank's post-ranking
Recall@200 is **.87117/.88934**; that small positional change is not new candidate
generation. Item universes remain 6,361/4,412 and the pool rows are identical.

## 2. A genuine group-aware objective

LightGBM was initially absent. Its **4.6.0 binary wheel** installed/imported
successfully into ignored `cache/ltr_dependencies` without changing original
requirements or global packages. `requirements-ltr.txt` adds a separate pinned
optional profile. No source compilation/system-library modification was needed.

Source attribution: [LightGBM objective documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective),
[LambdaRank truncation documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html#lambdarank_truncation_level),
and [Python API](https://lightgbm.readthedocs.io/en/stable/Python-API.html).
The new helpers use the existing library implementation; no external solution
code was copied.

Configuration shared by both ranking fits:

* `objective=lambdarank`, **binary `label_gain=[0,1]`**, query normalization on.
* 80 rounds, learning rate .06, 15 leaves, min leaf samples 50, max bin 63.
* L1=0, L2=1; two CPU threads, deterministic, forced column-wise histograms.
* Main/data/feature/bagging seeds all42; no subsampling, eval-set early stopping
  or round selection on either holdout.
* Contiguous user/task `group` sizes explicitly passed to the native Dataset.
* No classification positive-times10 weighting; ranking uses normalized queries
  and binary relevance gains.

**LambdaRank is an NDCG-based pairwise surrogate, not exact differentiable MAP@5.**
No NDCG value is used for model selection. Every comparison computes the existing
exact AP/MAP@5 outside LightGBM. Also, library, regularization and weighting differ
from the V4 classifier, so this is not a perfect objective-only causal experiment.

### Bounded experiment history, including failures

1. Initial predeclared model: truncation **8** (Top-5 plus a small tail).
2. It produced useful complementary corrections: ≥50 improved users and ≥20 new
   relevant IDs in each window, with no large MAP drop. Only then was the one
   declared **equal RRF with frozen V4** evaluated.
3. Neither passed the main uncertainty rule. Their original records/initial
   `selection.json` retain that outcome. One transparently documented adaptive
   follow-up changed **only truncation8→5** to focus the actual Top-5 region.
   No other parameter or fusion weight was searched, and fusion was not repeated
   for the follow-up. Total: **two real-data ranking fits, one fusion**.

| Strategy | Main MAP@5 | Earlier MAP@5 |
|---|---:|---:|
| Frozen V4 | .391427 | .402687 |
| LambdaRank truncation8 | .395373 | .408963 |
| One equal RRF of truncation8 + V4 | .392427 | .405961 |
| **LambdaRank truncation5 (selected V5)** | **.395679** | **.408858** |

The first ranking model's main gain CI was **[-.000131,.007930]**; the fusion's
was **[-.001300,.003402]**. They are preserved as insufficiently robust results,
not hidden. The follow-up plan and its unaltered decision rule are separate files.

## 3. Exact MAP evidence for selected V5

| V5 versus V4 | Main | Earlier |
|---|---:|---:|
| Absolute gain | **+.004252** | **+.006171** |
| Paired 95% bootstrap interval | [.000601,.007983] | [.001846,.010358] |
| Improved / regressed users | 289 / 239 | 321 / 240 |
| Warm MAP | .393534 → .397821 | .406885 → .413120 |
| Cold MAP | .128 → .128 | 0 → 0 |
| Light-history gain | +.007715 | +.015690 |
| Heavy-history gain | +.004107 | +.004619 |

History groups are the same training-derived <10 / ≥10 programme groups used in
earlier stages. Cold groups contain 10/13 users; final prediction has two cold
users. Their actual ranked lists remain identical, not merely equal in score.

Bootstrap uses 1,000 paired user samples, seed42. It does **not** correct adaptive
selection across two configurations/one fusion or earlier work on the holdouts.
The main lower bound is small. Earlier repeat-heavy users lose **.004122** on
average (131 users), despite positive heavy/light-history group means. No claim
of universal improvement or guaranteed Kaggle gain is made.

A cheap reporting-only recency-tie sensitivity remains positive **+.003773 /
+.002269**. Primary targets were not changed or refitted. Session/device, target
tie and timezone ambiguities remain inherited limitations.

### Concrete ordered examples

* **Correction .32→.60:** “Говорит Украина” moves rank4→1 and “Сегодня. Итоги…”
  rank5→2; “Губка Боб” moves rank2→3. The same three relevant items move from
  hits 2,4,5, `(1/2+2/4+3/5)/5=.32`, to hits1,2,3, `3/5=.6`. This is a genuine
  ordering gain, not increased candidate coverage.
* **Failure .76→.42:** “Маша и Медведь” and “ТСН” are over-promoted in a list
  containing “Моланг”, “Октонавты” and “Вспыш”; relevant “Пчёлка Майя” is lost.
  The learner's query loss can still make user-specific mistakes.

Exact identities, histories/counts, programme IDs/titles/categories, all Top-5s
and rational AP calculations are in ignored `private/user_examples.json`.
The same original two users retain popularity, repeat, ALS, CBF, V4 and selected
model comparisons. `private/user_ap_deltas.csv` records every user's old/new AP,
Top-5 and hit-rank movements. All checks were AI-executed, not student verification.

## 4. Final refit and independently checked candidate

The selected final ranker is refitted on the original two early tasks plus the
allowed historical earlier/main tasks through **27 July**. Features for each
task still use only its own pre-cutoff history. **5,040 queries, 1,069,381 pairs,
17,792 positives**; all target ends precede the hidden prediction window.

This final model is separate from the early-only model used for the reported
holdout numbers. It is not evaluated on its own historical training labels and
presented as independent validation. Unchanged final ALS/CBF and feature caches
are reused; no additional base-model fits or dataset joins were needed.

**`artifacts/final/submission_hybrid_v5.csv`**

SHA-256: **`249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f`**

V5 uses standalone LambdaRank scores inside the fixed pool (not the rejected
fusion). Negative raw ranking scores remain valid for relative ordering; only
programme IDs must be positive. Ties retain the known V2 pool order. Repeats are
allowed and cold users retain the original common popularity fallback.

The independent checker reloads the native text model, reproduces all scores and
1,260 ranked lists, validates original string IDs/order, five unique positive IDs
from 6,632 test candidates, group boundaries and training-role dates. It rechecks
six heldout prediction files against frozen targets, user groups, availability,
AP/hit positions and cold fallback. All V1–V4 artifacts remain hash-unchanged.

Measured resources:

* Initial ranking fit **5.19s**, complete A/B experiment **6.64s**, ~1.03GB peak.
* Follow-up fit **4.98s**, follow-up total **6.18s**, ~1.05GB peak.
* Final refit/generation **12.56s**, final fit **11.51s**, peak **1.48GB**.
* Heldout prediction ~.08–.09s per pool; final scoring/ranking/CSV ~.129s.
* **52 tests passed** in .362s, including actual LambdaRank ordering/determinism,
  group boundaries, duplicate items, binary Top-5 labels, separate user/fold
  queries, cold fallback and negative-score handling, plus all previous tests.

## Files and reproduction

* `tvrec/group_rank.py`: native objective, group validation, prediction policies.
* `tvrec/ltr_experiment.py`: audited initial comparison and conditional fusion.
* `tvrec/ltr_top5.py`: the one bounded truncation follow-up.
* `tvrec/ltr_final.py`, `tvrec/verify_v5.py`, `tvrec/ltr_evidence.py`.
* Plans, target/group audit, failure/gain results, decision, resource records,
  segments, source/environment/native-library hashes: `artifacts/stage6/`.
* Full configuration/model hashes and input hashes:
  `artifacts/final/submission_hybrid_v5.manifest.json`.
* Native text models/scores: ignored `cache/group_ltr/`; package: ignored
  `cache/ltr_dependencies/`. Root `requirements.txt` remains unchanged.

Completed commands, not necessary to rerun for the existing file:

```sh
python3 -m pip install --only-binary=:all: --no-deps --target cache/ltr_dependencies lightgbm==4.6.0
python3 -m tvrec.ltr_experiment audit
python3 -m tvrec.ltr_experiment experiment
python3 -m tvrec.ltr_top5
python3 -m tvrec.ltr_final
python3 -m tvrec.ltr_evidence
```

Fast final checks:

```sh
python3 run_tests.py --out artifacts/stage6/test_results.json
python3 -m tvrec.verify_v5
```

Alternatively install the pinned optional `requirements-ltr.txt` in a fresh
environment. Exact reproducibility requires the recorded base caches, version/
platform and native library; cross-platform bitwise identity is not promised.

Recommendation: V5 is justified for a future authorized attempt, with frozen V4
as fallback. No submission was made and no screenshot/session-export substitute,
PDF or GitHub publication was created.
