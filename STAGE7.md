# V6 investigation — RETAIN V5; no new submission file

## Corrected submission/grading record

The latest user message explicitly corrects prior submission information:
**all five versions were submitted**, and the instructor confirmed grading uses
the **BEST PRIVATE MAP@5**. This supersedes earlier public-score guidance and
earlier V4/V5-not-submitted statements. Historical files/session records are not
rewritten; the authoritative addendum is
`artifacts/monitoring/submission_history_grading_correction.json`.

| Submitted version | Private MAP@5 | Public MAP@5 |
|---|---:|---:|
| V1 | .40693 | .42158 |
| V2 | .41721 | .43381 |
| V3 | .42607 | .44042 |
| V4 | .42806 | .44323 |
| **V5 — best submitted** | **.43385** | **.44580** |

Provenance is the user-supplied verified history/instructor confirmation, not a
new assistant account query. Submission IDs/screenshots were not supplied or
fabricated. Five attempts have been used; **remaining capacity is unknown and was
not queried or assumed**. Kaggle scores never enter model fitting/selection.

## Evaluator and frozen protocol

The actual unchanged evaluator passes both explicit instructor examples:

* Truth A,B,C; A,B,D,C,E: `(1 + 1 + .75)/3 = 11/12 = .9166666667`.
* Truth A,B,C; D,A,E,B,C: `(.5 + .5 + .6)/3 = 8/15 = .5333333333`.

These are regression tests and measured JSON evidence, not just prose assertions.
Existing ≥80% uninterrupted original-duration coverage, frequency-selected
Top-5, ID-ascending frequency ties, empty AP=0 and **all 1,260 users** remain fixed.
Original main/earlier pools, candidate universes, dates and cold fallback remain.

Every evaluated ranker uses the same two disjoint historical supervision targets
ending **13 April**; neither evaluation target is used for model fitting. The
same 59 V5 features, 80-round LambdaRank settings, seeds and group sizes are
controls. New aggregates are built separately at each cutoff, from deduplicated
user/airing observations; later-trained aggregates are never reused earlier.

`artifacts/stage7/frozen_prior.json` protects **265 earlier files** including all
five CSVs, parameters/models/results and the correction addendum. All remain
hash-unchanged. The native AI-session export was not touched.

## Existing-feature/error audit and three new hypotheses

V5 already includes ≥50%/≥80% counts, history rates/ages/coverage, ALS/CBF/popularity
scores and ranks, channel completion/loyalty, and historical/future airing rates.
We did not repeat rejected raw day/week counts, global midpoint habits or generic
future-volume boosts. The new, predeclared blocks were:

1. **Active-relative stability:** completed-programme share of the user's active
   completed-viewing days, returns in consecutive *active user weeks*, completion
   burst concentration and distributed completion mass. This differs from merely
   re-adding raw distinct days/weeks. A one-off is not automatically a stable habit.
2. **Channel/calendar compatibility:** exact interval allocation into channel ×
   weekday × two-hour slots, using completed historical watch seconds. User/channel
   profiles shrink toward historical population/channel timing with a fixed
   one-hour prior; unknown user/channel pairs back off rather than imply dislike.
   Future-programme airtime overlap, previously supported share and personal lift
   above population timing add information absent from global time habits.
3. **Personal matched-opportunity drift:** compare historical and known-future
   channel/calendar-compatible opportunity rates, normalized by window length and
   a common .1/day prior, interacting changes with personal completion evidence.
   No future broadcast volume alone boosts an unseen personal programme.

Watch/schedule intervals are clipped at the relevant boundaries before binning,
including midnight/week transitions. Airing units use the visible fraction of the
original duration. Frequency remains distinct user/airing events, never accumulated
fragments. Forecast schedules are metadata only; no forecast user views are read.

V5's remaining relevant misses are largely still in the pool: main ranks6–20:
1,304;21–50:614;51–200:445;four further in-pool;762 absent. Earlier:
1,332/536/339/one/627. Frozen V5 lists were exactly reproduced before comparison.

## Results — no candidate improves both windows convincingly

| Method | Main MAP@5 | Earlier MAP@5 |
|---|---:|---:|
| **Frozen V5** | **.395679** | **.408858** |
| Active-relative stability | .394757 | .409093 |
| Channel/calendar compatibility | .395368 | .409820 |
| Matched-opportunity drift — best average, **rejected** | .395642 | .411555 |

The same selection rule applied to all three: both gains ≥.001, both paired95
lower bounds >0, no heavy-history regression and no >.003 light/repeat-heavy
regression; cold lists identical. **None qualify. Exactly three were run**, with
no combinations, new weights, rounds search or post-result target changes.

Best rejected drift comparison:

| Metric | Main | Earlier |
|---|---:|---:|
| Gain vs V5 | **−.000037** | **+.002698** |
| Paired 95% bootstrap interval | [−.002268,+.001993] | [+.000436,+.004859] |
| Improved / regressed users | 145 / 142 | 161 / 124 |
| Light-history gain | +.003280 | +.004451 |
| Heavy-history gain | −.000210 | +.002431 |
| Repeat-heavy gain | **−.005763** | +.001959 |
| Cold MAP | .128 unchanged | 0 unchanged |

Bootstrap uses 1,000 paired users, seed42; intervals are not multiplicity/adaptive-
selection corrected. The earlier benefit is not sufficient to rescue absent main
gain and main repeat-heavy harm. Channel/calendar and stability intervals also
cross zero on the main window. This does not prove the signals are intrinsically
useless, only that these bounded changes are not supported for replacing V5.

The early training histories span only two/four weeks, limiting personalized
calendar estimates. Timezone, target-tie/session assumptions and repeated use of
the same two validation windows remain limitations.

## Real cases and independent student reproduction

`private/rejected_candidate_examples.json` preserves actual histories, programmes,
new feature values and exact AP for two posthoc cases of the **rejected** drift
model; they were not used as feature inputs or to fit another experiment:

* **Correction .25→.50:** relevant “ТСН. Неделя.” moves rank2→1. The user has two
  actual target programmes, so `(1/2)/2=.25` becomes `(1/1)/2=.5`; it is not
  normalized as if the user had five positives.
* **Failure .80→.60:** relevant “Завтрак с 1+1” at rank4 is displaced by non-target
  “Маша и Медведь”. Four initial relevant hits become three. This remains a real
  counterexample even though the earlier aggregate improves.

The report-evidence bundle retains the same original two justified users'
histories and ordered popularity/repeat/ALS/CBF/V4/V5 lists. It adds explicit V5
rank-by-rank relevance, cumulative hits, precision contributions and rational AP.
The individual-method comparison remains in Stage2; it must precede the final
hybrid explanation in the eventual report.

Byte-exact copies of genuine prior join/source-line checks are in
`artifacts/stage7/private/report_evidence/`: view CSV record3913423 (line3913424),
channel805; schedule11–20 record656405 (line656406), programme10002079, “Спартак”.
The view covers 17 May06:39:08 to18 May02:09:32; the airing is17 May23:55 through
18 May00:50, duration3300s. **Include the validation boundary18 May00:00** when
manually checking: overlap3000s, fraction3000/3300=90.909%. This is a qualifying
airing example; entering a user's Top-5 still requires the separate frequency
selection. Do not treat every qualifying airing as a final relevant item.

These checks were executed by the AI assistant. The student has **not** been
claimed to have independently performed the AP/source checks. Instructions,
source file/row references and actual arithmetic are saved for that future work.

## Costs, checks and retained solution

* All three fits/new-feature construction/evaluation: **45.63s**, peak **2.07GB**.
* Best rejected drift model fit **4.71s**; inference/ranking about .105s main /
  .101s earlier. No final refit was performed because no replacement was selected.
* **59 tests passed** in .415s, including the instructor calculations, completed-
  week transitions, exact channel-conditioned timing, midnight conservation,
  no future-history input and schedule clipping, plus all original tests.
* Independent checker reloaded all three models and reproduced six validation
  files' scores, lists, same users/targets/pools, groups, AP/hits and cold fallback.
  It also proved old evidence unchanged and **no V6 CSV exists**.

The first checker invocation omitted the six frozen loyalty columns when loading
the original 53-column cache. LightGBM's shape guard correctly stopped it
(58 supplied versus64 trained). The **checker only** was fixed to assemble
53+6+new columns, with explicit dimension assertions. The shape guard was not
disabled, experiments were not refitted, and no metric/decision changed. The
corrected independent reconstruction passed every model/window.

**Retain:** `artifacts/final/submission_hybrid_v5.csv`

SHA-256: `249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f`.
V5 remains best submitted/private-grade model (.43385). **No cosmetic V6 file,
new Kaggle attempt, capacity assumption, PDF or publication was created.**

## Files / reproduction

New code: `tvrec/habit_features.py`, `tvrec/habits.py`,
`tvrec/habit_evidence.py`, `tvrec/verify_habits.py`. Source hashes/resources,
feature audit, fixed plan, all three results, decisions, subgroup changes,
instructor AP evidence and independent checks are in `artifacts/stage7/`.
Matrices/models stay under ignored `cache/habit_v6/`; private user data stay
under ignored `artifacts/stage7/private/`. No new dependency was installed.

```sh
# Completed experiment phases need not be repeated:
python3 -m tvrec.habits audit
python3 -m tvrec.habits experiment
python3 -m tvrec.habit_evidence
# Fast verification of the existing decision/evidence:
python3 run_tests.py --out artifacts/stage7/test_results.json
python3 -m tvrec.verify_habits
```

Use the corrected submission/grading addendum when writing the future report.
Older "not submitted" and public-grading statements are historical and superseded,
not an instruction to rewrite the native session export. Current submission
capacity must be checked separately if any future upload is ever requested.
