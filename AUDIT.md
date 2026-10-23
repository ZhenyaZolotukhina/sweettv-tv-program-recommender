# AI implementation and execution audit — baseline stage

This record describes work actually executed by the AI coding assistant in this
session. It does not attribute verification or authorship to the student.
The harness's original, unmodified session export is the authoritative full
conversation/tool record and must be preserved separately.

## Inspected originals

Read the actual `exam_quiz_GSB.md`, all eight slide text parts from
`LLM4Rec. Midterm Quiz.pptx`, CSV headers/sample rows and the supplied directory.
No archives or channel bridge files are present. Original filenames, file hashes,
schema, row counts and ranges are saved in `artifacts/dataset_accounting.json`.
The original raw files were not edited. No Git repository was initialized and no
commit, publication or Kaggle submission was made.

## Source verification, including failures

* Kaggle overview/evaluation and rules URLs yielded only the competition title,
  not inspectable rule text. Public competition-list/data API attempts returned
  401; one attempted internal endpoint returned 404. Exact hidden-label relevance,
  normalization/empty-user details and daily cap remain unverified.
* The local quiz verifies MAP@K and the five-ID template; it also documents late
  submission acceptance and requests the public score. Slides request private
  score and say the cap is “usually 5”. This conflict is saved for clarification,
  not silently resolved by inventing current competition rules.
* Direct HTTPS VseTV failed TLS transport; direct Python HTTP requests redirected
  in a loop. `webfetch` retrieved HTTP content, but Cyrillic windows-1251 text was
  corrupted. Jina's actual response to
  `https://r.jina.ai/http://www.vsetv.com/channels.html` decoded Cyrillic correctly.
  Direct Python retrieval then failed local certificate trust; curl timed out.
  The actual tool response was imported via `--channel-source` and saved at
  `artifacts/sources/vsetv_channels.txt`, hash recorded in dataset accounting.
* Parsed 1,128 external channel entries and found **211** normalized ID/title
  matches against supplied schedules. This supports shared namespace identity;
  remaining historical titles are not represented as individually verified.

## Implementation failures and corrections

1. Initial full-dataset attempts exposed DuckDB reserved-word parse failures:
   implicit aliases `rows`, `views`, `source`, the `window` column and an implicit
   `seconds` alias after a cast. Quoted aliases, explicit `AS`, and column name
   `phase` corrected these. Incomplete builds had no valid cache manifest and
   were rebuilt; no failed run was reported as successful.
2. Initial successful full run measured 54.34s preparation / 8.94s join /
   173.80s total / 2,572,156,928-byte peak RSS. That first baseline grid used
   each model's own popularity fallback, producing different cold-user lists.
3. Corrected fallback consistency: A is selected first; **all personalized
   configurations use that same A ranking for cold users and repeat padding**.
   Reran the full 240-configuration evaluation on cached interactions. Current
   reported MAP values and submission come from this corrected run. Current
   selection time is 118.57s. Its post-selection diagnostic initially failed on
   reserved alias `source`; corrected it and executed diagnostics separately,
   without unnecessarily rerunning the completed evaluation.
4. Binary-search overlap logic was factored into the tested production primitive.
   Added 50 deterministic randomized comparisons against brute force, plus
   leakage/strength tests invoking the actual feature aggregation on a tiny
   database. Source-semantic behavior is unchanged by that extraction.
5. Original source data has 13 schedule conflicts; resolved with a deterministic
   disjoint sweep and full-data per-view second conservation assertion. There is
   one zero-duration airing. Zero IDs never become item zero in interactions.
6. Original `duraton` differs from timestamp subtraction in 117 rows, always
   -3,600s near 29 March. Retain timestamp-based joining and record timezone
   uncertainty rather than inventing a correction. Seventeen exact duplicate
   viewing records are retained and documented as a small count-overstatement
   limitation. Original source counts are preserved.
7. Schedule spillover into the following Monday changes available item counts.
   Enforced calendar interval clipping rather than assuming filename contents
   exactly equal their calendar weeks. Validation/test candidates are 6,361/6,632.
8. Final verification: all **12 tests passed** in 0.419s. One chained diagnostic
   refresh subsequently exceeded the shell's 200-second timeout; its cause was
   not established. An isolated diagnostic retry with a 15-second traceback
   timer succeeded without triggering the timer. The independent saved-CSV check
   was then rerun successfully; its SHA256 is
   `5326f14eb0b82bb4927b56b299e053b871d68524f4353040bbc909b770471ea2`.
   Existing validated metrics/submission were not changed by these diagnostic runs.

## Measured evidence and interpretation

* `validation_results.json` and `validation_grid.csv`: actual model settings,
  identical eligible users, cold/warm MAP and timing, plus truth sensitivity.
* `diagnostics.json`: actual truth-size distribution and popularity prevalence
  explain the high local MAP. They are not a claim about Kaggle scoring or value
  to users. Single-holdout selection optimism is explicitly acknowledged.
* Private JSONs contain real source rows, intervals, histories and two-user
  comparisons. They are excluded from Git rather than exposed in routine logs.
* Unit test executions are captured in `test_results.json`; actual independent
  full-file readiness validation is in `submission_independent_check.json`.
* Current code is AI-authored, using installed DuckDB/NumPy/pandas and Python's
  standard library. No external model, pretrained weights, hidden labels or
  leaderboard optimization was used. External VseTV information verifies channel
  semantics only; recommendation scores use supplied view/schedule data.

No advanced method, final academic report, official score, screenshot or personal
student verification has been fabricated. The two required advanced approaches
and all final course submission requirements remain next-stage work.

## Stage two continuation — target repair and advanced methods

The preceding section describes the completed historical baseline stage. Its
numbers are retained; they are superseded for selection by `STAGE2.md`, not
silently edited into competition-aligned results.

1. Read the user-supplied Melentev article. Used Jina's text proxy to access the
   **actual official Kaggle Data page**, which specifies **>=0.8** and frequency
   **Top-5**, and explicitly confirms channel-ID identity. The article says
   **over** 80% and no channel changing; official inclusive wording takes
   precedence, with strict >80% fully computed as a sensitivity. Source excerpts,
   dates, URLs, authority and ideas used are in `relevance_sources.json`.
2. Inspected the existing ground-truth SQL. It used >=60-second overlaps and all
   qualifying IDs. Reproduced all three old selected baseline MAP values exactly,
   then held their predictions and 1,246-user cohort fixed to isolate target
   repair. Repeat drops .797207 -> .335243. The original numbers/artifacts remain.
3. Reused original cached views, original disjoint schedule and original matched
   intervals; no raw CSV re-ingestion or old 240-setting sweep was performed.
   Constructed only the necessary new continuous-session tables in a separate
   DB. Removed 17 duplicated intervals; conflicting concurrent channels were
   conservatively excluded. Session construction took 33.73s. All assumptions,
   target versions, frequency ties, long-session and cumulative-visit sensitivity
   results are saved. The official target has 5,946 pairs; all 1,260 users remain
   in the new primary denominator, including 26 zeros.
4. Implemented independent implicit ALS using SciPy sparse/positive-definite
   solves, and genuine metadata TF-IDF CBF using scikit-learn. Small deterministic
   tests verify objective decrease/latent groups/seeding and content similarity,
   unseen metadata IDs and exclusion of future-only vocabulary. This stage now
   implements both required advanced methods; the preceding historical statement
   that they remained future work no longer describes current project status.
5. The first partial experiment fit three ALS configurations in 5.06/7.69/7.12s
   and produced predictions. CBF runs then spent ~54s each on redundant accounting:
   `set(f.item)` was rebuilt for every candidate. The 200s shell timeout stopped
   that partial run. Moved set construction outside the loop, retained the model
   code/data, reused already fitted ALS arrays, and saved progress incrementally.
   Successful CBF generation is ~1s. No failed experiment was reported as a
   completed run; this failure did not alter label policy or candidate/user sets.
6. Completed 17 individual and 26 fusion/switch comparisons. Best official-target
   MAP=.373631 (strict-target MAP=.373229), using weighted RRF with k=10,
   component Top-200 and repeat/CF/CBF/popularity weights 1/.1/.1/.1. All component
   input coverage thresholds are >=50%; ALS factors32/alpha10/reg1/12 iterations,
   seed42; rich CBF. Same 1,260 users and 6,361 candidates throughout.
7. Froze main choices and ran one earlier retrospective chronology check without
   tuning: repeat .381888, same RRF .387936. Main paired gain over repeat is
   +.007624, 223 improved / 90 worsened users. Saved bootstrap uncertainty with an
   explicit warning about same-holdout selection optimism.
8. Successful experiment took 89.57s with cached main ALS arrays and peaked at
   2,953,428,992-byte RSS. Since the partial initial run had not persisted original
   fit memory, ran **one winning-ALS fresh-fit resource benchmark**: 5.006s,
   peak 1,709,686,784 bytes including cached training read/feature/matrix creation.
   No further tuning was done during this benchmark. Cached loading is not
   represented as training time.
9. Independently recomputed the real 3000/3300 >80% witness using an endpoint
   enumerator that does not call the production interval/session code. Rechecked
   all 43 main prediction CSVs against fixed original user order, candidates and
   target; every AP and hit position recomputed. Preserved the same two stage-one
   representative users and their real titles, full temporal run histories,
   predictions, actual content-profile terms and actual AP arithmetic. Private
   identifiers stay in gitignored paths. Public summaries use U-heavy/U-light.
10. Final **23 tests passed** in .089s; machine-readable result/code hashes are
    `artifacts/stage2/test_results.json`. The original test code remains intact.
    The shared old test-summary path was refreshed by the existing test runner
    during the initial continuation; historical 12-test success remains in this
    audit/session. Added `--out` so current-stage test records are now separate.
11. Directly read the original CSV view line 3,913,424 and schedule line 656,406
    with the dedicated Read tool, confirming the cached witness against both
    originals. Saved exact private source text and midnight clipping arithmetic
    in `artifacts/stage2/private/source_line_checks.json`. This was a targeted
    evidence check, not a repeated whole-data ingestion.

No hidden Kaggle labels, leaderboard optimization, final refit, submission,
publication or PDF was used/created. External ideas are credited. Local target
ties, concurrent-device semantics, timezone and model-selection uncertainty
remain explicit limitations; there is no claim of student-performed verification.

## Final-refit continuation — first candidate, manual Kaggle upload required

1. Verified the exact `selected_strategy.json` against the Stage-2 winner and
   fusion grid: RRF Top-200, k=10, repeat/CF/CBF/popularity weights 1/.1/.1/.1,
   local MAP .37363117283950614. Verified original algorithm-source hashes against
   the actual Stage-2 execution. The two advanced implementations/evaluations
   were preserved. A pre-refit snapshot hashes **98 earlier artifact files**;
   independent post-refit checks prove all are unchanged.
2. Added a separate final-refit runner. Opened existing databases read-only;
   no original pipeline rebuild or experiment search. Refit on weeks 11–30 only,
   preserving corrected continuous sessions and ≥50% model-input coverage.
   All checks found **zero hidden-period views/broadcasts/interaction targets**.
   Final qualified user–airing occurrences: 744,122 (initial 405,446); user–item
   pairs: 293,911 (156,741); users: 4,867 (4,784); programmes: 4,029 (2,661).
3. Rebuilt user/item mappings, ALS factors and metadata vocabulary. Matrix shape
   4,897×7,881; vocabulary 25,210 terms from 8,233 historical metadata items,
   versus initial 20,258 terms. Test schedule used only for candidate availability
   and transformed permitted metadata. No hidden test viewing behavior exists
   in the supplied views or was inferred as a training label.
4. Generated **one final hybrid candidate** and a byte-exact repository archival
   copy. Both SHA-256:
   `d4026ac73d6f940010d7be3f8241731bd5975054fc3127eebd0cc1cfaebfed72`.
   Paths: `artifacts/final/submission_hybrid_v1.csv` and
   `repository_artifacts/submission_hybrid_v1.csv`. Original string IDs/order,
   five distinct positive available IDs, repeats and shared cold fallback are
   preserved. This is not the preliminary baseline.
5. Actual final refit/generation took 14.997s, peak sampled RSS 2,684,878,848 bytes.
   Separately executed **27 tests**, all passed in .112s after refitting. The new
   independent validator checks formatting, original template/candidates,
   factor mappings and all-row RRF reconstruction directly from component
   scores. Compressed score arrays are materialized once before per-user checks
   to avoid repeated decompression; no failed expensive verification was run.
6. Checked only the presence of standard/custom Kaggle credential locations and
   credential environment variables, plus CLI/package availability; all absent.
   No secrets were printed or extracted. Rule acceptance/submission permission
   are unknown because authenticated access is unavailable. **Zero Kaggle network
   requests or submission attempts were made.** No browser-cookie workaround,
   invented score, submission ID or screenshot. The manifest records nulls and
   `MANUAL_UPLOAD_REQUIRED`; exact one-upload instructions were saved.
7. Copied the two actual Stage-2 representative users and source join example
   into an ignored final report-evidence bundle, with byte-identical original
   files and additional exact per-user/per-method AP fractions and arithmetic.
   All personal verification remains for the student; current checks are AI-run.

Final artifacts/source/README/tests are reproducible. No second candidate,
hyperparameter search, PDF, repository publication or claimed Kaggle score was
created. The one authorized real attempt remains a manual user upload because
account authentication is missing.

## Stage-three offline improvement — v1 monitored, v2 not submitted

1. User reported successful v1 submission with public MAP=.42158 and private
   MAP=.40693, filename `submission_hybrid_v1.csv`, description
   `midterm-hybrid-v1-rrf`. Recorded this in a separate monitoring addendum with
   user-message provenance. Submission ID/screenshot were not supplied or
   fabricated. No account access or Kaggle request was attempted. Kaggle scores
   are not read by the experiment-selection code.
2. Hash-froze **115 prior artifact/repository files**, including the exact v1 CSV,
   manifest, configuration and Stage-1/2 outputs. V1 remains SHA
   `d4026ac73d6f940010d7be3f8241731bd5975054fc3127eebd0cc1cfaebfed72`.
   Original model/matching code and separate ALS/CBF evaluation results remain.
3. Saved a fixed initial plan and evaluated eight new interpretable configurations
   against v1 on BOTH unchanged windows/cohorts/candidate pools. Exact v1 outputs
   and MAP were reproduced. Reused cached joins and ALS scores; reconstructed
   only small CBF/feature scores. No new ALS fits or original CSV re-ingestion.
4. Mild recency, confidence-only, opportunity-only and time habits did not improve
   both windows. Confidence/opportunity combination gained .001506 main but only
   .000008 earlier. Channel addition at .1 gained .001124/.001946 but the main
   bootstrap interval crossed zero. Saved all results, including failures.
5. Based only on these local results, tested **two** extra channel weights .3/.5,
   with a documented stricter rule requiring both gains >=.001 and both paired
   95% bootstrap lower bounds >0. Both qualify; .5 improves both windows more.
   Selected main .3803611111 vs v1 .3736311728, earlier .3939296737 vs .3879356261.
   Main gain .006730 CI [.002926,.010436]; earlier .005994 CI [.002096,.009988].
   Adaptive selection optimism remains explicit; no further weight expansion.
6. Main 306 users improve / 196 worsen; earlier 303 / 206. Saved all per-user AP
   contributions/hit positions, plus training-derived segments. Main richer-history
   users benefit; earlier sparse users regress .007607. Saved a real largest-gain
   example (.146667 -> .55) and real largest regression (1 -> .6), with programme
   titles/categories, personal historical counts/channel profiles, exact Top-5
   and AP arithmetic. Retained the original Stage-2 two-user comparisons as well.
7. Refitted the channel component on all allowed weeks 11–30. Reused frozen v1's
   already-correct final ALS/CBF models/mappings because their settings and training
   data did not change. New rank contribution: .5/(10+channel rank), positive-score
   Top-200, deterministic v1 ties/fallback. Only history-derived channel counts
   enter this new component. Final candidate set remains 6,632, two cold users
   retain exactly the same v1 fallback.
8. Created `artifacts/final/submission_hybrid_v2.csv`, SHA-256
   `987f8fde48654ba982c20b36e25bf78017749c58b15d96bccdbc8eca5f7d2671`.
   Separate checker independently reconstructed input qualification/channel counts
   in SQL, every channel score and every final Top-5. It also checked all 22 offline
   files against frozen targets/users/candidates/warm-cold groups and recomputed
   AP/hit positions. All prior frozen files remain unchanged. **33 tests pass**.
9. Actual initial experiment 31.83s, follow-up 10.13s, final new-component refit/
   generation 5.345s, ~2.32GB peak for that final process. Final measurement reuses
   v1 fitted models and is not compared as an equivalent cold full-model refit.
   Resource, dependency/source hashes, comparisons, code and reproducibility
   instructions are saved. No execution failures were concealed.

V2 is a measured candidate for a future explicitly authorized attempt, not a
claim that its Kaggle score will improve. No Kaggle submission, PDF or public
repository was created in this stage. All calculations/evidence preparation
were performed by the AI assistant; no student verification is asserted.

## Stage-four supervised reranking — temporal fit, V3 prepared, no submission

1. User reported V2 public/private MAP=.43381/.41721. Saved a separate monitoring
   addendum with user-message provenance, not an independently retrieved ID or
   screenshot. No optimization code reads Kaggle scores. Hash-froze 161 prior
   artifact/repository files; preserved V1/V2 and their old source/validation.
2. Diagnosed fixed V2 retrieval: main recall@20/50/200=.65607/.75429/.85469;
   earlier .68705/.79189/.88123. A truth-independent union of V2 Top200 and five
   components' supported Top50 gives .87185/.88952 recall. More missed positives
   are already inside that pool below rank5 (2459/2314) than absent (762/627).
   Saved overlaps, per-user losses and history/repeat/channel segments.
3. LightGBM/XGBoost/CatBoost absent; used existing sklearn histogram gradient
   boosting. Declared one configuration and two outputs (pure / equal RRF blend).
   Built two independent rolling-cutoff feature sets: histories before Mar23
   with targets Mar23–Apr6, and histories before Apr6 with targets Apr6–Apr13.
   Refit each early fold's ALS/CBF/popularity/channel values at its OWN cutoff.
   All 7560 positive training pairs and all negatives end before the earlier
   evaluation starts Apr13. No main/earlier labels in this evaluated model fit.
4. Saved 53 numerical/category/genre features, train-only coverage/frequency/
   recency/duration/user summaries, score/rank signals and permitted schedules.
   No user/item identity, target or target-history share is a predictor. No
   random validation/early stopping. Fit 513292 candidate pairs in 17.80s.
5. Pure model MAP=.383703/.404011; its main gain interval crossed zero. Preset
   conservative blend passes both gain/CI requirements: .384336/.400075 versus
   V2 .380361/.393930. Gains .003975/.006146, paired CIs [.001174,.006700] and
   [.003473,.008806]. Main 211 improve/177 regress; earlier 195/113. Selection
   optimism and short-target horizon shift are explicit limitations.
6. Only after freezing output style, refitted a SEPARATE final model on four
   non-overlapping historical target windows through Jul27: 1069381 candidate
   pairs, 17792 positives. This final model includes historical heldout labels
   legitimately for final refit, but it is NEVER evaluated on them and presented
   as independent validation. Both model paths/hashes and label dates are saved.
7. Produced `artifacts/final/submission_hybrid_v3.csv`, SHA
   `5ebb58dfb26662ea2b10bc3ab0f2e5641f2b846f5cea1a10b604e0d0d9d7c304`.
   Independent checker verifies formatting, original IDs/order, 6632 available
   candidates, temporal provenance, model reload/scores and every final blend.
   It also rechecks four heldout files' targets, groups, availability and AP/hits.
8. Actual diagnosis 47.91s/2.54GB; early-fold experiment 52.96s/1.94GB; final
   refit/features/CSV 64.44s/3.79GB, with final fit 30.91s. No raw re-ingestion,
   old hyperparameter grid, external package install or Kaggle request.
9. Saved exact correction (.76 -> 1) and failure (.483333 -> .233333) with real
   programmes/histories/AP arithmetic, plus original two users' popularity,
   repeat, ALS, CBF, V2 and V3 lists. Empty qualified histories are separated
   from repeat-heavy histories in the final segment comparison. Training split
   gains are labelled descriptive, not causal/holdout feature ablation.
10. Final regression: 41 tests passed in .167s, covering cutoff and overlap
    guards, invalid use of final-refit labels for reported holdouts, target-
    invariant features, retrieval without truth insertion, cold fallback and
    previous join/MAP/submission tests. All prior frozen files remain unchanged.

V2 remains the best actually submitted result. V3 is a modest validated candidate
for a future separately authorized attempt. No Kaggle submission, report PDF,
GitHub publication, artificial screenshot or replacement session history was
created. New calculations were executed by the AI assistant, not claimed to have
been personally verified by the student.

## Stage-five final focused investigation — three experiments, V4 not submitted

1. Recorded the user's V3 public/private .44042/.42607 report separately with
   provenance, no invented submission ID/screenshot and no Kaggle access. Frozen
   V1/V2/V3 and 194 prior artifact/model/repository files remained unchanged.
2. Independently reproduced both established targets by SQL: 5946/5675 positives,
   26/51 empty users, all1260 in MAP, same availability and deterministic ID ties.
   Found zero labelled physical broadcast duplicate keys. Retained original
   session/device/conflict rules. A recency-tie sensitivity was scored without
   fitting alternative labels; target membership ambiguity remains explicit.
3. Measured V3 ranking errors: main warm one-off precision .15576 versus .49102
   for sustained repeats. Only 27 main/23 earlier users had a wrong one-off above
   a missed repeatedly completed programme. Measured frequency/rate Spearman1,
   high rank/score redundancy. Saved real per-user failures before experiments.
4. Predeclared and ran exactly THREE single-block additions, with all V3 model
   settings, early-only supervision, candidate pools and fusion fixed. Consistency
   and personal opportunity blocks did not improve both windows robustly. Only
   six channel completion/loyalty features passed: main .391427 vs .384336,
   earlier .402687 vs .400075. No combinations, additional weights or fourth
   experiment. No new ALS/CBF fits or original pipeline rebuilding.
5. Gains .007091/.002612; paired CIs [.004512,.009772]/[.000456,.004670]. Main
   196 improve/94 regress; earlier141/108. Cold fallback unchanged; light-history
   gains .005753/.001995. Recency-tie gains remain positive .005868/.002504.
   Bootstrap/model-selection optimism and reused-window limitations retained.
6. Refitted only the chosen block with allowed historical labels through Jul27,
   using separate final/early model files. Final holdout scores remain from the
   earlier-only model (labels end Apr13), not the refit. Generated V4 at
   `artifacts/final/submission_hybrid_v4.csv`, SHA
   `08dbcca1aa312f03397447205684143b20c9a632d5fc8868c4c2916210d44ce2`.
   Independent checker verified temporal roles, original IDs/order, five distinct
   valid IDs, all-row learned ranking reconstruction and six offline ablations.
7. Target/error audit1.42s; three experiments72.51s/2.10GB; final refit33.26s/
   3.07GB, final fit29.05s. Only selected final features were computed. The final
   regression suite passed 46 tests in .171s; prior source/results remain intact.
8. Evidence includes correction .52->.8, failure .71->.4, original two users and
   required popularity/ALS/CBF lists. Important contrary evidence is preserved:
   false one-off predictions increased813->837 main and1317->1323 earlier, while
   total relevant hits rose2755->2791 and2788->2804. Thus the winner is not claimed
   to universally solve the one-off hypothesis. Split gains are descriptive;
   causal experimental evidence is for the six-feature block as a whole.
9. A reporting-only SQL query failed on an implicit reserved alias `seconds`.
   Renamed it `watch_seconds` and reran successfully. No target/model/selection/
   CSV changed. This was not hidden as a successful first execution.

V3 remains the best actually submitted solution; V4 is prepared for one future
separately authorized attempt. No Kaggle submission, PDF, GitHub publication or
fabricated personal verification occurred. All new checks were AI-executed.

## Stage-six group-aware learning-to-rank — V5 prepared, no submission

1. Preserved 227 prior artifact/repository/model files, including V1–V4. Audited
   actual V4 classifier: binary log_loss, correct frequency-selected Top5 labels,
   not every >=80%-completed show. Training query audit: 2520 user/task groups,
   513292 rows/7560 positives, all target ends <=Apr13. Feature cutoffs, contiguous
   group boundaries, unique programme IDs and binary relevance were checked.
2. V4 recall@200=.87067/.88881, with2393/2244 relevant misses inside the fixed pool
   below5 versus762/627 absent. Existing pools and all1260-user MAP targets retained.
   No original preprocessing, candidate experiment or base ALS/CBF fit repeated.
3. LightGBM absent initially. Successfully installed/imported 4.6.0 binary wheel
   into ignored cache/ltr_dependencies without altering original requirements or
   system packages. Added optional requirements-ltr.txt and source attribution to
   actual LightGBM docs. No system-library workaround or compile was needed.
4. Initial native LambdaRank: 80 rounds, 15 leaves, lr.06, L2=1, maxbin63,
   deterministic col-wise CPU2 threads, all seeds42, binary gains[0,1], normalized
   user/task groups, truncation8. No heldout eval_set/early stopping or classifier
   class weighting. Objective is NDCG-based surrogate; selection uses exact MAP5.
5. Initial A improved point MAP to .395373/.408963, but main CI crossed zero.
   It passed the predeclared complementarity gate, so ran exactly one equal
   V4/Lambda RRF: .392427/.405961, also main CI crossing zero. Preserved the
   initial retain-V4 result; did not silently replace those failed conclusions.
6. Disclosed one bounded adaptive follow-up changing ONLY truncation8->5 to
   focus Top5; two real-data parameter configurations total, no more fusions or
   weight search. Selected .395679/.408858 vs V4 .391427/.402687, gains
   .004252/.006171 with CIs [.000601,.007983]/[.001846,.010358]. Main289 improve/
   239 regress; earlier321/240. Heavy/light history groups gain both; cold lists
   unchanged. Earlier repeat-heavy subgroup loses .004122, explicitly retained
   as a limitation. No claim of universally beneficial ranking or unbiased CIs.
7. After configuration freeze, separately refit all allowed historical tasks
   through Jul27: 5040 groups,1069381 rows,17792 positives. Reported heldout scores
   remain from the model whose labels end Apr13; final model not retested on its
   training labels as independent validation. Used identical 59 V4 feature columns.
8. Generated artifacts/final/submission_hybrid_v5.csv, SHA
   `249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f`.
   Independent checker reloaded the model and reconstructed every final ranking,
   original1260 IDs/order, five distinct positive available items, temporal roles,
   group boundaries, and all six offline files' truth/groups/AP/hits/cold fallback.
   V1–V4 remain byte-identical. No Kaggle request or attempt.
9. Actual fits5.19s/4.98s, experiments6.64s/6.18s ~1.03/1.05GB; final12.56s,
   fit11.51s, peak1.48GB. Heldout inference ~.08s. Saved optional package/native-
   library/environment hashes and all configurations. No execution failure hidden.
10. Saved a true order-only correction (.32->.60: relevant hits2,4,5 become1,2,3),
    failure (.76->.42), exact programme names/categories/histories/AP arithmetic,
    original two representative users and required ALS/CBF/popularity lists.
    Per-user old/new hit-rank movements, subgroup deltas and target-tie reporting
    sensitivity remain available. Final52 tests passed in .362s, including genuine
    native ranking learning, grouping/label errors, reproducible seed, negative
    raw-score handling and previous project checks.

The comparison changes ranking engine/regularization/weighting as well as loss,
so no perfect objective-only causal claim is made. Adaptive selection optimism,
short training horizons and original session/tie/timezone assumptions qualify
all results. V5 is a candidate for a future separately authorized attempt; no
Kaggle score, PDF, screenshot, publication or student verification was fabricated.

## Stage-seven V6 investigation — corrected private grading, RETAIN V5

1. Latest user correction: ALL FIVE V1–V5 were submitted, including V4, and the
   instructor explicitly confirmed BEST PRIVATE score is used for grading.
   Saved all five public/private scores and CSV hashes in a new monitoring/
   grading addendum, with user-supplied verification provenance. Best is V5
   private.43385/public.44580. Prior creation-time submission/public-grading
   statements remain historical, now explicitly superseded. Five attempts used,
   remaining quota unknown; no account/capacity query or submission.
2. Froze265 prior artifact/model/repository files, including correction evidence
   and all five CSVs/configurations. Verified the unchanged evaluator against
   both instructor calculations: 11/12=.9166666667 and8/15=.5333333333. Added
   explicit regression tests and JSON actual values. No target/cohort changes.
3. Inspected actual V5's59 features and avoided rerunning rejected raw day/week
   counts, generic global habits or generic future-volume priors. Predeclared
   exactly THREE new blocks: active-user-relative completion persistence,
   exact channel/weekday/two-hour overlap compatibility, and personal matched-
   opportunity change. All cutoffs, groups, labels, seeds,80-round LambdaRank
   settings and pools otherwise fixed. Evaluated models' labels end Apr13.
4. Main/earlier MAP: V5 .395679/.408858; active stability .394757/.409093;
   channel/calendar .395368/.409820; matched drift .395642/.411555. No main-window
   improvement. No fourth configuration, combinations or weight search.
5. Best diagnostic drift gains -.000037/+.002698, CIs[-.002268,.001993] and
   [.000436,.004859]. Main145 improve/142 regress; earlier161/124. Cold unchanged;
   main heavy -.000210 and repeat-heavy -.005763, though light gains .003280.
   Earlier-only benefit cannot justify V6. Retained V5; no final refit or V6 CSV.
6. New experiment/feature construction45.63s, peak2.07GB; drift fit4.71s and
   inference~.1s. No dataset reprocessing, base-model fits or new dependencies.
7. Saved real rejected-candidate correction (.25->.50, target size2) and failure
   (.80->.60), exact feature values/programme titles/histories and AP arithmetic.
   Copied genuine join/source-line evidence byte-exact, plus the same two original
   users' all-method histories/predictions and rank-by-rank V5 AP calculations.
   These are AI checks, not assertions that the student performed verification.
8. First independent checker omitted the six V5 loyalty columns from its feature
   assembly. LightGBM shape checking rejected58 supplied versus64 expected.
   Fixed ONLY checker assembly to53+6+new, added explicit dimension checks and
   did not disable the shape guard, retrain experiments or change decisions.
   Corrected checker reproduced all scores/lists/AP/hits/groups for six files,
   preserved every prior frozen file, and confirmed absence of V6 CSV.
9. Final59 regression tests passed in .415s, including both instructor AP cases,
   active-week persistence, channel-conditioning beyond global habits, exact
   midnight-second conservation, future-history rejection and boundary clipping.
   Independent validation passed. Existing ALS/CBF/popularity records remain.

V5 is the retained best submitted/private-grading solution. No Kaggle request,
attempt-capacity assumption, PDF, GitHub publication or native session-export
edit occurred. Corrections are append-only historical addenda. Bootstrap
intervals remain unadjusted for repeated/adaptive comparisons and original
session/tie/timezone ambiguities remain explicit.

## Stage-eight new collaborative model — genuine novelty, no V7 replacement

1. Reviewed the actual Melentev Medium prize-winning second-place account and
   LightFM documentation, plus original BPR reference(arXiv1205.2618/UAI2009).
   Saved exactly adapted ideas: ranking CF scores/ranks, available biases and
   two-stage candidates; no external solution code copied or historical Kaggle
   score used as a target. Existing V1–V5 artifacts/models were frozen/preserved.
2. One LightFM1.17 attempt, isolated targetcache/no-deps, failed obtaining build
   requirements with dict.__LIGHTFM_SETUP__ AttributeError on Python3.14.
   No source patch, system/compiler/dependency fight or repeat install. Switched
   to explicitly named NumPy/SciPy BPR fallback; did not fabricate LightFM/WARP
   experiments. Exact error/command/provenance saved in source methodology JSON.
3. Implemented tested implicit BPR MF, sparse per-entity minibatch AdaGrad,32
   factors/25epochs/lr.05/reg.002/seed42. Two positive samplers only: binary and
   log(1+qualified-frequency), uniform users, unseen historical-item comparisons
   excluding all known positives. Unobserved pairs not claimed dislikes. Item
   bias learned; user bias cancels and is omitted as unidentifiable. Preserved
   >=50% input and >=80%/frequency-Top5 evaluation, all1260, same fallback.
4. Log-frequency standalone MAP=.311584/.308569 vs frozen ALS=.165965/.225822.
   Recall200=.87639/.89692; new relevant pairs370/340 outside ALS200 and220/134
   outside V5's whole pool. This passed declared information gate. No conclusion
   that BPR is generally superior to tuned ALS, or that working WARP would fail.
5. Fitted two own-cutoff early BPR models beforeMar23/Apr6 for reranker labels
   endingApr13. Added8 actual latent score/rank/bias/cosine/norm/support columns
   to unchanged59 V5 features. Same-pool MAP=.392906/.408882, main -.002774.
   Original Top50 expansion had only3/7 novel positives and was rejected; gate
   and original plan preserved, not silently altered.
6. Disclosed ONE bounded adaptive Top200 expansion, justified by measured tail
   novelty220/134. Total downstream variants two; no new CF configuration or
   weight search. Existing feature rows reused; only new pairs rebuilt, old
   sample rows matched original features. Pool recall rises to.90885/.91313,
   but MAP=.394048/.408347. Gains -.001631/-.000511, paired CIs both crossing0.
   No newly added outside-pool relevant items entered Top5 in this model.
7. Main expanded219 improve/224 regress, earlier214/206; cold unchanged, heavy
   main -.002265. Light main+.010242/earlier-.001780; no robust replacement.
   Saved real correction.52->.8/failure1->.76 and actual programme metadata/
   histories/exact AP, original representatives, latent scale-shift diagnostics
   and all independent method records. Observations not causal proof.
8. Actual standalone44.15s/1.75GB; same-pool17.67s/1.43GB including early CF;
   expansion42.35s/2.46GB with no new CF fits, native reranker fits5.37/6.69s.
   All independent source SQL positive matrices, factors/biases/scores, four CF
   and four reranker file rankings/AP/groups/fallback/cutoffs passed checker.
   65 deterministic tests passed in .485s. No original data preprocessing rerun.

Retained V5; no final V7 refit/CSV, Kaggle request, extra submission, PDF,
publication, native-session edit or claimed student verification. Methodological
deviations/failures and bounded adaptive expansion remain explicit. WARP itself
was not successfully run; findings apply to this tested fallback and integration.

## Stage-nine associations and BPR calibration — retain V5, no V8

1. Froze 334 prior artifact/model/repository files. Audited actual BPR candidate
   exposure, raw scores, feature arrays and recovered-positive cases. No verified
   implementation bug was found. Recovered pairs had supported embeddings, but
   none of those exposed in early candidate training was positive there. Sparse
   positive exposure and measured latent norm shift motivated calibration.
2. Independently implemented binary distinct-user item baskets, smoothed
   confidence and shrinkage cosine, minimum support3 and top200 non-self edges.
   Repeats affect log-frequency seed mass, not distinct-user support. Credited
   mlxtend's support/confidence documentation; no dependency/code was copied.
   Standalone main/earlier MAP: confidence .268212/.282135; cosine
   .202816/.210695. Selected confidence by mean standalone holdout MAP.
3. Ran exactly two integrations with fixed LambdaRank parameters and earlier-only
   supervision: A adds association Top100 and five fields to frozen59; B retains
   previous BPR Top200 expansion and calibrates eight CF fields with explicit
   missing/support handling. No new latent fits or raw-data preprocessing.
4. A main/earlier MAP .390187/.406643; B .394045/.407970; frozen V5
   .395679/.408858. Both lose on both windows. A main paired CI is wholly
   negative; the other three cross zero. No robust promotion; retain V5.
   Full intervals, subgroup guards, actual corrections/regressions, useful and
   misleading rules and original representative-user records are saved.
5. Rule evidence includes consequent basket prevalence and empirical lift:
   high confidence for popular news does not imply specific or causal affinity.
   Actual BPR cases include score/rank gaps and descriptive TreeSHAP; popularity
   contributions do not justify a blanket suppression claim.
6. Audit4.335s/1.691GB; standalone12.237s/2.084GB; both integrations including
   features51.696s/2.909GB peak process RSS. Ranker fits5.530/7.796s.
   70 tests passed in2.102s. Independent checker reconstructed four standalone
   and four integration files and checked users/AP/truth/cutoffs/fallback,
   frozen59/BPR pools and all protected files. These checks were AI-executed.
7. Initial checker wildcard accidentally included a subset recovered-pair CSV
   and failed the full1260-user assertion. Restricted verification to four
   declared integration prediction filenames; reran successfully. This was a
   checker correction, with no model, data or metric change.

`STAGE9.md` and `artifacts/stage9/` contain full results and provenance. No V8
CSV/final refit, Kaggle request, submission, PDF or publication occurred.
Retained V5 SHA-256:
`249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f`.
V5 remains best at .43385 PRIVATE per the user's corrected verified history.
Bootstrap intervals are unadjusted for repeated/adaptive comparisons and do not
establish hidden-period gains. Historical records remain preserved.

## Submission finalization — frozen V5, no further optimization

At the user's explicit finalization request, selected V5 LambdaRank and verified
PRIVATE .43385 were fixed. Prepared `github_ready/` by whitelisting source/tests,
pinned requirements, aggregate historical plans/results, V1–V5 exact CSVs and
manifests. Raw Kaggle files, caches, credentials and full private records remain
outside the package. Prior research artifacts/models were not overwritten.

Created consolidated `REPORT_EVIDENCE.md`, `STUDENT_CHECKS.md`, `REPRODUCE.md`
and `FINAL_CHECKS.md`. Included only the two original representative users'
qualified programme-frequency profiles, a few timestamped intervals, ordered
same-method Top5s/metadata and exact AP fractions, plus the required real join
witness. Numerical evidence is mechanically derived from saved original records,
not invented. Source references and AI assistance remain explicit.

Added a standalone standard-library release checker, a whitelist package builder
and a fixed V5 replay entry point. The replay is supplied for future use and was
not trained/executed end-to-end at finalization. Its imports/syntax and all five
frozen feature roles were checked. Added `--read-only`/separate-output support to
the existing V5 validator so final verification cannot rewrite old manifests.

Executed 70 regression tests from the release source: all passed in .458s.
Reloaded the existing V5 native model and exactly reproduced every score/final
ranking, checked six old holdout files, original1260-user order, five distinct
positive available IDs and6632-candidate universe. Verified the original two
physical source lines for the join and exact AP11/20. All334 protected old files
remain hash-identical; all V1–V5 CSV hashes agree with verified history.

All13 runtime pins match installed versions; `pip check` reported no broken
requirements. Package hash/inventory, syntax and exclusion/secret-pattern checks
passed with no secret findings. This is a local publication scan, not an online
vulnerability audit. Results are AI-executed, not personal student verification.

No new real-data model experiments, Kaggle requests/submissions, GitHub
publication, PDF creation or native-session-export modification occurred.
Manual repository publication/link, actual score screenshot, student exercises,
report PDF and original unmodified native export remain before24October2026,
19:00Moscow. The fixed replay/new-environment installation was not rerun; this
scope limit is disclosed in the release instructions and final checks.
