# Stage four — temporal supervised reranking, V3 prepared but not submitted

## Decision and frozen evidence

The selected V3 is **equal reciprocal-rank fusion of V2 and one small supervised
reranker**, inside a fixed retrieved pool. It improves both heldouts under the
pre-specified rule. This is a modest measured gain, not a promised Kaggle gain.

The user reports V2 Kaggle public/private MAP=.43381/.41721. These are recorded
separately in `artifacts/monitoring/kaggle_v2_user_report.json` with user-message
provenance. The submission ID/screenshot were not supplied or independently
retrieved. No model-selection code reads these scores. **No Kaggle request,
submission, PDF or publication occurred.**

All **161 pre-existing artifact/repository files** are hash-frozen in
`artifacts/stage4/frozen_prior.json`. V1/V2 CSVs, configurations, old validation
results, monitoring addenda and session history remain preserved. V1/V2 hashes:

* V1: `d4026ac73d6f940010d7be3f8241731bd5975054fc3127eebd0cc1cfaebfed72`
* V2: `987f8fde48654ba982c20b36e25bf78017749c58b15d96bccdbc8eca5f7d2671`

The original distinct ALS and metadata CBF implementations and independent
evaluations remain unchanged. The new ranker is additional work, not a relabelled
baseline or a replacement for the course's method comparison.

## 1. Retrieval versus ranking diagnosis

Same official-style reconstructed truth: uninterrupted **≥80%** coverage, distinct
user/airing frequency and user Top-5, numeric-ID frequency ties. All **1,260 users**
remain in MAP, empty truth AP=0. Candidate universes remain **6,361 main** and
**4,412 earlier**; no user/cohort/availability changes explain the gain.

### Candidate recall before learned reranking

Recall below is **micro recall over unique relevant user–programme pairs**.
`candidate_recall.csv` also reports the all-1,260 macro convention (empty=0).
Recall is not substituted for MAP.

| V2 retrieval budget | Main recall | Earlier recall |
|---|---:|---:|
| Top-20 | 65.61% | 68.70% |
| Top-50 | 75.43% | 79.19% |
| Top-200 | 85.47% | 88.12% |
| Expanded union pool | **87.18%** | **88.95%** |

The pool is **V2 Top-200 UNION positive-score Top-50 of repeat, ALS, CBF,
popularity and channel**. There is no ground-truth injection. Mean pool sizes are
222.05/219.29, maximum 269. The first 200 positions retain V2 order; expansion is
ordered by V2, so before reranking its Top-20/50/200 are unchanged.

Of V2's main Top-5 misses, **2,459 relevant pairs are in the pool but below five**,
versus **762 absent**. Earlier: **2,314 versus 627**. Main ranking misses split
into ranks 6–20: 1,176; 21–50: 584; 51–200: 597; 102 further relevant additions
are recovered by the expanded union. Thus reranking is justified, although
remaining retrieval misses impose a ceiling.

Mean Top-50 Jaccard overlaps, including deterministic padding when needed:

| Generator pair | Main | Earlier |
|---|---:|---:|
| Repeat / channel | .364 | .411 |
| ALS / CBF | .263 | .252 |
| Repeat / V2 | .638 | .510 |

The generators overlap substantially but are not identical; their small union
expansion is retained. `generator_overlap.csv` contains all 15 pairwise comparisons
per window. `private/v2_losses.csv` records per-user AP loss, missing candidates,
training-history segments and posthoc target-history share. The latter is
diagnostic only and **never a model feature**.

## 2. Supervised training without heldout-label leakage

LightGBM/XGBoost/CatBoost were not installed. The existing pinned scikit-learn
`HistGradientBoostingClassifier` provides a small efficient binary scoring model.
Its outputs are ranking scores, not calibrated satisfaction probabilities.

For **both reported heldouts**, the one fitted estimator sees only these earlier,
non-overlapping target windows:

| Fold | Feature/history cutoff | Training target interval | Candidate rows | Positives |
|---|---|---|---:|---:|
| A | 23 March 2020 | [23 March, 6 April) | 252,933 | 4,408 |
| B | 6 April 2020 | [6 April, 13 April) | 260,359 | 3,152 |

Histories start 9 March. **Every supervised label ends by 13 April**, exactly when
the earlier evaluation begins. Earlier holdout is `[13 April,18 May)` and main
holdout `[18 May,27 July)`. Neither target was used in fitting this evaluated
model. No random internal validation/early stopping was used.

At each rolling cutoff, historical user/item counts, popularity, channel profiles,
ALS factors and TF-IDF profiles are rebuilt using only that fold's history.
Later-trained main/earlier ALS values are not reused in earlier folds. Known
target schedules may supply availability and metadata, never future views.
The categorical feature vocabulary is frozen using metadata before 23 March.

**513,292 training pairs, 7,560 positives** supported this test. Target membership
is assigned only after feature construction; positives outside the retrieval pool
are not inserted into training. Sample weights normalize each query and multiply
positive pairs by ten, then normalize within fold. This is a weighted binary
classification approximation to ranking, not a claimed LambdaRank implementation.

Only **one** model configuration was fitted:

* 80 iterations, learning rate .06;
* maximum 15 leaves, max bins 63, minimum leaf samples 50;
* L2 regularization 10, early stopping **off**, seed 42;
* two compute threads via threadpool limits.

### Exact feature families (53 columns)

* Normalized scores and reciprocal ranks from repeat, ALS, CBF, popularity,
  channel and V2.
* Qualified historical airing frequency, ≥80% count, recent-14-day count,
  overlapping watch seconds, mean coverage, strong-coverage ratio, first/last
  observation age and history-normalized frequency rate.
* User history count, programme diversity, repeated-airing fraction and recent
  activity fraction; global programme frequency.
* Historical and known-target airing rates, programme durations, channel counts,
  history length and known prediction-horizon length.
* One-hot actual programme categories and twelve historical genre labels.

There are no raw user or item identifiers, targets, target-history shares, or
future user-view features in the matrix. `feature_names` and vocabulary are
saved exactly. Content metadata is canonical permitted schedule data; future-only
categories/genres outside the frozen vocabulary do not expand feature dimensions.

Training tree split gains rank repeat rank, V2 score, global popularity, channel
score, ALS score, history rate and recent observations highly. These are
**training split-gain descriptions**, not causal attribution or an independent
feature-ablation claim. Pure ranker versus blended output is the explicit ablation
isolating the new supervised signal.

## 3. Controlled comparison

Two output policies were declared before training: learned scores alone, or equal
reciprocal-rank fusion with V2. Cold users always retain the exact V2 fallback.

| Method | Main MAP@5 | Earlier MAP@5 |
|---|---:|---:|
| Frozen V2 | .380361 | .393930 |
| Supervised only | .383703 | .404011 |
| **V2 + supervised equal RRF (selected)** | **.384336** | **.400075** |

The standalone ranker's main gain CI crosses zero. The conservative blend passes
the preset rule: main gain ≥.003, earlier gain ≥.001 and positive paired 95%
lower bounds on both. No extra estimator/parameter search followed.

| Selected V3 versus V2 | Main | Earlier |
|---|---:|---:|
| MAP gain | **+.003975** | **+.006146** |
| Paired bootstrap 95% interval | [.001174,.006700] | [.003473,.008806] |
| Improved / regressed users | 211 / 177 | 195 / 113 |
| Warm MAP, V2 → V3 | .382380 → .386387 | .398036 → .404246 |
| Cold MAP, V2 → V3 | .128 → .128 | 0 → 0 |
| Top-5 relevant hits, V2 → V3 | 2,725 → 2,755 | 2,734 → 2,788 |
| Top-5 micro recall, V2 → V3 | .45829 → .46334 | .48176 → .49128 |

Bootstrap uses 1,000 paired user resamples, seed 42. It does **not** correct the
optimism of choosing pure versus blended output on these two heldouts. Both
windows are validation, not a fresh independent test.

### Segments and failure patterns

* Light histories (<10 qualified programmes) gain **+.01530 main / +.01194 earlier**;
  ≥10-programme histories gain +.00342 / +.00523.
* Repeat-heavy histories (≥75% repeated-airing share) gain +.00723 main but only
  +.00015 earlier. More exploratory observed histories gain +.00353 / +.00698.
* Broad-channel users gain +.00395 / +.00661; focused users (≥60% on one channel)
  gain +.00445 / +.00399. Cold users remain unchanged.
* Empty qualified histories are reported separately rather than interpreting
  zero observations as a repeat-heavy profile.

Definitions are operational properties of the supplied histories, not claims
about a user's lifetime preferences. `v2_v3_segments.csv` and private paired
contributions preserve actual counts and all per-user regressions.

Two concrete posthoc diagnostic examples:

1. **Correction .76 → 1.0:** the new list removes “Ничто не случается дважды” and
   introduces the relevant “Чистоnews”, while retaining “Губка Боб”, “ТСН”,
   “Викна-Новости” and “Элвиннн!!!”. V2 AP was `(1+1+1+4/5)/5=.76`; all five
   V3 results are relevant. This is observed local correctness, not enjoyment.
2. **Failure .483333 → .233333:** “Вспыш и чудо-машинки” is incorrectly promoted
   above “Свинка Пеппа”, and relevant “Отчаянные домохозяйки” is displaced. This
   mixed-category user is a counterexample to universally beneficial reranking.

Exact identities, historical continuous runs, programme IDs/titles/categories,
target sets, per-method Top-5 and rational AP calculations are in
`private/user_examples.json`. The original two Stage-2 users are also preserved
with popularity, repeat, ALS, CBF, V2 and selected V3 lists. Their APs remain
.55 → .55 and 0 → 0. All analysis was AI-executed; no student verification is claimed.

## 4. Final refit is separate from reported validation

After freezing the configuration, the **final** estimator was refitted on the
two early folds plus the now-allowed historical target windows
`[13 April,18 May)` and `[18 May,27 July)`. These four targets do not overlap and
all end before the hidden period. Each fold retains features computed at its own
cutoff. Final fitting uses **1,069,381 pairs / 17,792 positives**.

This final model is **not** scored on those training targets and presented as
independent validation. The early-only evaluated model and final model have
separate files/hashes and explicit roles in the V3 manifest.

For test-period features, unchanged V2 base models already trained on allowed
weeks 11–30 are reused. Hidden-period schedule availability and metadata are
permitted; hidden user viewing behavior/labels are never used. Final candidate
universe remains **6,632** and the two cold template users keep V2's fallback.
All repeats remain eligible.

Selected within-pool output score:

`1/(10 + rank_V2_in_pool) + 1/(10 + rank_supervised_in_pool)`.

Pool ties retain V2 order. Positive-gaps in global V2 ranks outside its Top-200 are
compressed to within-pool positions; this exact definition is used in both
validation and final prediction and is tested independently.

**Final candidate:** `artifacts/final/submission_hybrid_v3.csv`

**SHA-256:** `5ebb58dfb26662ea2b10bc3ab0f2e5641f2b846f5cea1a10b604e0d0d9d7c304`

Independent validation checks original 1,260 string IDs/order, five distinct
positive available IDs, all final training-label ends, the evaluated model's
earlier-only label provenance, model parameters/hashes, reloaded-model scores
and every user's independently reconstructed blend. Both V1 and V2 remain
byte-for-byte unchanged. Four heldout prediction files were independently checked
against frozen targets, availability, user groups, AP and hit positions.

## 5. Actual computation and regression checks

* Initial diagnosis/heldout feature-cache construction: **47.91s**, **2.54 GB** peak.
* Early-fold feature construction + early-only fit/evaluation: **52.96s**,
  **1.94 GB** peak. The classifier fit itself: **17.80s**.
* Final refit/features/CSV: **64.44s**, **3.79 GB** peak; final classifier fit
  **30.91s**. Scoring a heldout candidate pool takes approximately 1.65–1.72s.
* Only two new small rolling-cutoff ALS/CBF fits were necessary; main/earlier/final
  existing base models were reused at their correct cutoffs. No raw-data pipeline
  rebuild, old grid rerun, package installation or Kaggle access.
* **41 tests passed** in .167s, including rolling-label cutoff guards,
  non-overlapping supervision, final-refit labels rejected for heldout fitting,
  no random early stopping, target-invariant features, retrieval expansion
  without truth insertion, deterministic ties and common cold fallback.

The early targets span 7–14 days while holdouts span 5–10 weeks. Rate/horizon
features reduce but do not eliminate this distribution shift. Final refitting
adds longer historical targets; its hidden-period improvement remains unmeasured.
Other risks are finite candidate recall, model-selection optimism and original
session/device, frequency-tie and timezone assumptions.

## Files and reproduce

* `tvrec/rerank_data.py`: cutoff-specific candidate retrieval, base fits, 53 features.
* `tvrec/rerank.py`: diagnosis, rolling supervision, one estimator/two output tests.
* `tvrec/rerank_final.py`: conditional final refit, refuses cosmetic V3.
* `tvrec/verify_v3.py`: separate checker, never alters V1/V2 artifacts.
* `tvrec/rerank_evidence.py`: post-selection user/segment evidence only.
* `artifacts/stage4/`: plans, source/resource hashes, recall/overlap/error tables,
  supervision audit, comparison, selection, split gains, tests and verification.
* Private predictions/profiles: `artifacts/stage4/private/`; matrices/models:
  ignored `cache/rerank/`. No raw histories or credentials belong in public Git.
* Complete final provenance: `artifacts/final/submission_hybrid_v3.manifest.json`.

Commands used (completed phases need not be repeated):

```sh
python3 -m tvrec.rerank diagnose
python3 -m tvrec.rerank experiment
python3 -m tvrec.rerank_final
python3 -m tvrec.rerank_evidence
python3 run_tests.py --out artifacts/stage4/test_results.json
python3 -m tvrec.verify_v3
```

Feature matrices are source/cutoff/vocabulary cached. Preserve their recorded
hashes; invalidate the relevant rolling model/feature caches if changing fold
dates or model parameters rather than relabelling an old cache. Reproduction
requires existing audited data caches and pinned dependencies/environment. No
cross-platform bitwise identity or new Kaggle score is claimed.

Recommendation: V3 is justified for a **future explicitly authorized attempt**,
with V2 retained as the best actually submitted fallback. Nothing was submitted
in this step and no PDF/session-export substitute or screenshot was fabricated.
