# Stage 9 — item associations and BPR calibration

## Decision

**Retain frozen V5. No V8 final refit or CSV was created.** Both bounded
integrations scored below V5 on both chronological holdouts. The user's verified
submission history still identifies V5 as best on PRIVATE score (.43385;
public .44580). Local experiments were selected without leaderboard feedback.

Retained file: `artifacts/final/submission_hybrid_v5.csv`.

SHA-256: `249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f`.

## Fixed evaluation and controls

- Main: train [2020-03-09, 2020-05-18), evaluate [2020-05-18, 2020-07-27).
- Earlier: train [2020-03-09, 2020-04-13), evaluate [2020-04-13, 2020-05-18).
- All 1,260 template users; empty truth contributes zero. Binary AP@5 uses
  `min(5, unique truth size)` as denominator.
- Truth remains the frequency Top-5 of distinct qualifying airings with >=80%
  uninterrupted original-duration coverage, descending frequency/item-ID tie
  order. Model input remains >=50%; repeats are eligible; cold fallback is fixed.
- Reranker supervision uses only Mar23–Apr6 and Apr6–Apr13 early tasks, with
  features trained before their respective cutoffs. Neither evaluated holdout's
  labels enter fitting. V5's first 59 features and LambdaRank parameters are fixed.
- `artifacts/stage9/plan.json` records two association scorers and at most two
  integrations. Promotion required gains >=.001 and paired 95% lower bounds >0
  on both windows, subgroup guards and identical cold predictions.
- `frozen_prior.json` protects 334 previous artifact/model/repository files;
  independent checks confirm they remain unchanged.

## BPR diagnosis

The previous Top-200 BPR expansion recovered 220 main / 134 earlier relevant
user–item pairs outside the V5 pool, but none reached Top-5. Inspection found no
verified feature or candidate-exposure bug: old feature arrays matched exactly,
and all recovered pairs had supported embeddings.

Added-pair positive prevalence was only .000837/.000682 in the two early
training tasks, versus .002213/.001565 in main/earlier evaluation. Of recovered
pairs, 156/127 had appeared in early candidate training, but **none was positive
there**. This is training exposure, not evidence that the model learned those
pairs as relevant. Median user-factor norms were .116849/.189024 early versus
.654251/.279786 main/earlier, supporting a scoped scale-calibration experiment.

Actual cases and descriptive TreeSHAP are in `private/bpr_actual_cases.json`.
For example, “Ліга сміху” was BPR rank 46 but final rank 59, with reranker score
-2.649073 versus a .548541 Top-5 cutoff; “Мыслить как преступник” was rank 47
then 70. Their history features were zero. These observations support weak
ranking/generalization, not a causal claim that popularity suppressed them;
popularity contributions in the first case were actually positive.

## Training-only item associations

An independent SciPy implementation uses one binary basket per historical user.
If `s_i` is distinct-user support and `c_ij` shared-user support:

- Smoothed confidence: `c_ij / (s_i + 10)`.
- Shrinkage cosine: `c_ij / sqrt(s_i*s_j) * c_ij / (c_ij + 20)`.

Require >=3 shared users, remove self edges and retain 200 neighbors per
antecedent with deterministic ties. Score candidates from L1-normalized
`log1p(qualified airing frequency)` seed weights. Repeated airings affect seed
weights, not independent-user support. Each graph uses its own training cutoff.

Support/confidence reference:
<https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/>.
No mlxtend dependency or copied solution implementation was used.

| Standalone scorer | Main MAP@5 | Earlier MAP@5 | Main / earlier Recall@200 |
|---|---:|---:|---:|
| Confidence | .268212 | .282135 | .876556 / .901322 |
| Shrinkage cosine | .202816 | .210695 | .872015 / .899559 |

Confidence was selected by average standalone MAP across the two windows. Its
Top-200 retrieves 200/130 relevant pairs outside the old V5 pool. Standalone
selection uses these holdouts, so its scores are not untouched test estimates.

A useful posthoc rule, “Когда мы дома” → “ТСН. Неделя”, has 749 shared users,
792 antecedent users and .933915 smoothed weight. A misleading rule, “Губка Боб”
→ the same news programme, has 1,458/1,702 and .851636 but misses the actual
cartoon-oriented user's future target. News basket prevalence is .835911;
empirical lift is only 1.13135/1.02480. High confidence can mostly reflect
popularity; these weights are neither causal effects nor future probabilities.

## Exactly two downstream experiments

**A — Association expansion:** V5 pool union supported association Top-100,
plus five association features appended to the frozen 59. New rows use
cutoff-correct features; existing rows reuse verified values. The same policy
is applied in early training and evaluation.

**B — Calibrated BPR:** retain the previous expanded BPR Top-200 pool and first
59 features. Replace eight raw CF fields with cutoff-local z/percentile,
cosine/norm-percentile and support fields; unsupported observations are explicit
NaNs with support indicators. Calibration is label-free and pool-invariant.
There are no new latent fits or candidate-size searches.

| Method | Main MAP@5 | Earlier MAP@5 | Main / earlier gain vs V5 |
|---|---:|---:|---:|
| Frozen V5 | .395679 | .408858 | — |
| A: association | .390187 | .406643 | −.005493 / −.002215 |
| B: calibrated BPR | .394045 | .407970 | −.001634 / −.000888 |

Paired user-bootstrap 95% intervals for MAP differences:

- A main [−.009057, −.002110]; earlier [−.005987, +.001792].
- B main [−.004975, +.001951]; earlier [−.004079, +.002274].

B is the better tested integration, but neither window shows an improvement.
Intervals are unadjusted for repeated/adaptive comparisons; two holdouts do not
establish performance on the hidden period. Expanded recall alone did not
deliver a better final ranking. Main A improves light users by .009615 but loses
.006327 for heavy and .009879 for repeat-heavy users. B also loses on main heavy
and repeat-heavy users. All cold lists remain identical.

`segments.csv`, `relevant_top5_hits.json` and private per-user records preserve
the full breakdown. Actual A examples include AP .76→1.0 and .76→.483333;
histories, programme titles, rankings and hit arithmetic are saved, along with
the original two representative users. They are posthoc illustrations.

## Execution and verification

| Executed work | Wall seconds | Peak process RSS, decimal GB |
|---|---:|---:|
| BPR audit | 4.335 | 1.691 |
| Four standalone association evaluations | 12.237 | 2.084 |
| Both integrations and feature construction | 51.696 | 2.909 |

Native ranker fits: A 5.530s, B 7.796s; inference about .11s and .12–.14s per
window. These are measured process peaks, not incremental model allocations.
Existing interaction/latent caches were reused without raw-data preprocessing.

The AI coding assistant executed **70 tests**, all passing (2.102s), including
five new tests for support/count semantics, shrinkage, pruning/repeats,
calibration/missing data and pool invariance. `independent_validation.json`
confirms reconstruction of four standalone and four integration prediction
files, native ranker scores, AP, users, truth, availability, fallback and cutoffs.
It also checks the unchanged BPR pool/59 features, frozen files and V5 hash.

One initial checker glob included `main_recovered_bpr_rows.csv`, a diagnostic
subset rather than full-user predictions. The checker was corrected to name
the four integration prediction files explicitly. Models, data and metrics
were unchanged; the corrected checker passed.

Final verification commands actually executed:

```sh
python3 -m tvrec.association_evidence
python3 run_tests.py --out artifacts/stage9/test_results.json
python3 -m tvrec.verify_association
```

Core code: `tvrec/association_model.py`, `assoc_study.py`, `association_data.py`,
`assoc_integrate.py`, `association_evidence.py`, `verify_association.py`.
Machine-readable plans, metrics, resources, cases and decision are under
`artifacts/stage9/`. Tests are in `tests/test_association.py`.

No Kaggle requests/submissions, publication or PDF creation occurred in this
stage. Verification is AI-executed, not student-performed. Prior submission
history is user-reported verified evidence; remaining submission capacity is
unknown. Historical documentation remains preserved with current addenda.
