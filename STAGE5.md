# Final focused investigation — V4 prepared, no Kaggle submission

## Frozen record and conclusion

Exactly **three** predeclared, single-block feature experiments were run against
otherwise unchanged V3. Only the **channel completion/loyalty** block passed the
selection rule. V4 is supported for one future explicitly authorized attempt;
V3 remains the best actually submitted solution until then.

The user reports V3 public/private MAP **.44042/.42607**. These are stored with
user-message provenance in `artifacts/monitoring/kaggle_v3_user_report.json`.
No identifier/screenshot was supplied, no independent account access is claimed,
and optimization code never uses Kaggle scores. **No Kaggle request was made.**

`artifacts/stage5/frozen_prior.json` protects **194 earlier files**, including all
three submission CSVs, configurations, results, monitoring and existing ranker
models. Repeated checks confirm they are unchanged. ALS, CBF and popularity
implementations and independent evaluation records remain intact.

## 1. Measured V3 errors and final target audit

V3 post-reranking micro recall at Top-20/50/200:

| Window | @20 | @50 | @200 |
|---|---:|---:|---:|
| Main | .68180 | .78019 | .87033 |
| Earlier | .72546 | .82167 | .88863 |

Main relevant misses below rank five: **1,299** at ranks 6–20, **585** at 21–50,
**536** at 51–200, nine further inside the pool, and **762 absent**. Earlier:
1,329 / 546 / 380 / five / 627. Thus ranking remains a larger bottleneck than
candidate absence. Candidate policies and pools were not changed in this study.

Among warm-user V3 Top-5 predictions, zero/one-observation items have precision
**.15576 main / .18603 earlier**, versus **.49102 / .53866** for items with at
least two historical qualified airings. There are **27 main / 23 earlier users**
with a wrong zero/one-observation prediction above a missed programme completed
at least twice. This is a measured error pattern, not a universal explanation,
nor evidence that short/one-off viewing means dislike.

Redundancy was measured on candidate rows with history. Within each window,
`history_log_frequency` and `history_rate` have Spearman **1.0**; repeat and V2
ranks correlate about **.97**, V2 score/rank about **.97–.98**. Duplicating these
signals is not new preference information. No fourth feature-removal search was
run; all controls retain the same 53 baseline columns.

### Target audit, without redesign

Independent SQL exactly reproduced every saved target using longest contiguous
qualifying overlap, original duration and **≥80%**, frequency counted once per
distinct user/airing, then five programmes ordered by frequency descending and
numeric ID ascending. No labelled duplicate physical broadcast keys were found.
Existing single-channel session/duplicate/conflict handling was retained.

| Window | Target pairs | Five-positive users | 1–4 positives | Empty | Candidates |
|---|---:|---:|---:|---:|---:|
| Main | 5,946 | 1,127 | 107 | 26 | 6,361 |
| Earlier | 5,675 | 1,044 | 165 | 51 | 4,412 |

All **1,260 users** remain in MAP, including empty AP=0. Previously watched
programmes stay eligible. VseTV/channel identity is explicitly confirmed by the
official source previously saved; no guessed mapping is introduced.

Official tie/session/device details remain incomplete. A cheap **frequency then
recency** tie sensitivity changes membership for 647/644 users. It was scored
without retraining, never adopted as the primary target; selection also required
that the candidate's gain not reverse under this sensitivity. No alternative
target was chosen to increase a reported score.

## 2. Exactly three hypotheses, unchanged V3 learning protocol

Plans were saved before experimental results. Every experiment adds one coherent
block, uses the same 80-tree/15-leaf histogram classifier, seed 42, sample weights,
V2/learned equal RRF, candidate pools and cold fallback. No combinations, weight
search, new ALS/CBF fits or original pipeline rebuild occurred.

For both reported windows, supervised targets still end **13 April**, using the
two original rolling folds. Histories and aggregates for each fold stop at its
own cutoff. Neither main nor earlier holdout labels fit the evaluated models.

| Single added block | Hypothesis | Main MAP | Earlier MAP |
|---|---|---:|---:|
| Frozen V3 | Control | .384336 | .400075 |
| A: completion consistency | Distinct ≥50%/≥80% days/weeks and completion age distinguish sustained repeats from bursts | .384678 | .399963 |
| **B: channel completion/loyalty** | Personal channel loyalty × population completed-programme popularity improves conditional relevance | **.391427** | **.402687** |
| C: personally gated opportunity | Historical completion yield plus known future timing/opportunity predicts repeat counts | .384736 | .399138 |

A and C had intervals crossing zero and did not improve both windows. They were
rejected; their features are **not** included in final V4. C explicitly gates
opportunity on personal completions, so more future broadcasts alone cannot give
an unseen item a positive personal opportunity score. Even that defensible prior
did not produce a robust validation gain.

### The six selected features (53 → 59 columns)

1. User channel share based on raw qualified-airing frequency.
2. User channel share based on qualified overlapping watch time.
3. Fraction of that user's channel observations completed ≥80%.
4. Log qualified-airing intensity on the channel.
5. Programme popularity conditional on channel, using population ≥80% observations.
6. Personal raw channel loyalty multiplied by that completed-programme popularity.

For a programme on multiple channels, these are averaged using its **known target
schedule's normalized channel mix**. Absolute future broadcast volume is not a
personal relevance bonus. All user/population counts and durations are historical;
only permitted schedule metadata supplies the target channel mix. Duplicate
user/airing observations are rejected; split visits are never summed to meet
coverage. Population counts use all permitted historical users, not future labels.

The **block-level ablation** measures the gain. Training split gains give the
loyalty × completed-popularity interaction the largest new-feature gain, but this
is descriptive and does not separately establish each column's causal effect.

## 3. Gain, uncertainty, subgroups and counterexamples

| V4 − frozen V3 | Main | Earlier |
|---|---:|---:|
| Absolute MAP gain | **+.007091** | **+.002612** |
| Paired bootstrap 95% interval | [.004512,.009772] | [.000456,.004670] |
| Improved / regressed users | 196 / 94 | 141 / 108 |
| Warm MAP | .386387 → .393534 | .404246 → .406885 |
| Cold MAP | .128 → .128 | 0 → 0 |
| Light-history (<10 items) MAP gain | +.005753 | +.001995 |
| Alternative tie-rule gain | +.005868 | +.002504 |
| Top-5 relevant hits | 2,755 → 2,791 | 2,788 → 2,804 |

Bootstrap: 1,000 paired user resamples, seed 42. Intervals do **not** adjust for
choosing among three feature blocks, prior work on the same holdouts, or final
refit distribution shift. Neither window is a fresh untouched test.

Important counterpoint: V4 does **not** universally fix one-offs. Warm false
zero/one-history predictions increase **813→837** main and **1,317→1,323**
earlier; correct ones also increase 150→160 and 301→308. The gain is better
overall Top-5 membership/order from conditional channel completion, not a blanket
rule to remove weak-history items. The two pre-experiment one-off displacement
cases remain AP=0 and .05, respectively.

Concrete actual users, selected posthoc for explanation:

* **Correction .52 → .80:** “Завтрак с 1+1” enters rank three and “Викна-Новости”
  moves from fifth to fourth, replacing “Чистоnews” in the list. V3 had hits at
  1,2,5: `(1+1+3/5)/5=.52`; V4's first four hit: `4/5=.8`.
* **Regression .71 → .40:** relevant “Чистоnews” and “ДжеДАИ” are displaced by
  non-target “Викна-Новости” and “Секретные материалы”. Channel completion priors
  can overgeneralize a channel preference and lose individual programme specificity.

Private exact identities, historical ≥50%/≥80% programme counts, titles/categories,
truth, Top-5 and rational AP arithmetic are saved in `private/user_examples.json`.
The original two representative users and popularity/repeat/ALS/CBF comparisons
are also retained. All checks were AI-executed, not student-performed verification.

## 4. Conditional final refit and independent file checks

Only selected block B was refitted. Final supervised labels may now use the four
non-overlapping historical windows through **27 July**, exactly as V3's final
refit. These labels are never used for the reported independent holdout scores;
the earlier-only and final estimators have separate paths/hashes. No hidden-period
user behavior enters training. Old ALS/CBF models and base feature tables are
reused at their correct cutoffs, preserving mappings and definitions.

**`artifacts/final/submission_hybrid_v4.csv`**

SHA-256: **`08dbcca1aa312f03397447205684143b20c9a632d5fc8868c4c2916210d44ce2`**

The separate checker validated all 1,260 original string IDs/order, exactly five
distinct positive IDs from **6,632** test candidates, model parameters/provenance,
reloaded-model scores and every user's independently reconstructed ranking. The
two final cold users retain the existing fallback. All six ablation files were
rechecked for frozen truth, availability, groups, AP and hit positions.

Actual resources:

* Target/error audit: **1.42s**, peak .50 GB.
* All three early-only experiments plus feature construction: **72.51s**, peak
  **2.10 GB**; selected early-model fit **14.34s**.
* Final refit/generation: **33.26s**, peak **3.07 GB**; selected final fit
  **29.05s**, extra final features 2.59s, scoring/CSV .68s. This reuses base
  feature caches, so it is not an apples-to-apples cold-pipeline speedup claim.
* **46 tests passed** in .171s. New checks cover unchanged V3 learning settings,
  the three-experiment cap, unique-airing/cutoff guards, day/week completion
  arithmetic, hand-checkable channel probabilities and personal gating of future
  opportunity. Earlier join/MAP/leakage/format/model checks remain intact.

One reporting query initially failed on an implicit SQL alias `seconds`; it was
renamed `watch_seconds` and rerun. No model, selection, target or CSV changed due
to that reporting-only correction. The failure/correction are recorded in AUDIT.

## Reproduce / inspect

```sh
# Completed phases; do not repeat expensive work unless reproducing it.
python3 -m tvrec.focused audit
python3 -m tvrec.focused experiment
python3 -m tvrec.focused_final
python3 -m tvrec.focused_evidence
# Fast final regression/independent checks:
python3 run_tests.py --out artifacts/stage5/test_results.json
python3 -m tvrec.verify_v4
```

Plans, results, target audit, tie sensitivity, redundancy, error counts, subgroup
comparisons, models' source hashes and measured resources are under
`artifacts/stage5/`; personal histories/predictions are in ignored `private/`.
New feature matrices/models stay under ignored `cache/focused_v4/`.
Full final provenance is `artifacts/final/submission_hybrid_v4.manifest.json`.
No dependency or old implementation was changed; the pinned environment remains.

V4 merits one **future explicitly authorized** Kaggle attempt based on both local
windows, with V3 retained as the best submitted fallback. No hidden score is
predicted. Selection optimism, session/device and target-tie ambiguity, original
timezone uncertainty, and individual programme regressions remain limitations.
No submission, PDF, GitHub publication, fake screenshot or session-history
replacement was made.
