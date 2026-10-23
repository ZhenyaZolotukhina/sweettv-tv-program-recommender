# Final refit and first-upload readiness

The Stage-2 winner is frozen: weighted RRF, repeat/ALS/CBF/popularity weights
**1/.1/.1/.1**, component **Top-200**, **k=10**. Its measured local validation
MAP@5 is **.37363117283950614**. This is **not** a Kaggle score. All original
individual method evaluations, label repairs and earlier results remain unchanged.

## Result

* Final CSV: **`artifacts/final/submission_hybrid_v1.csv`**.
* Exact repository archival copy:
  **`repository_artifacts/submission_hybrid_v1.csv`**.
* SHA-256 for both:
  **`d4026ac73d6f940010d7be3f8241731bd5975054fc3127eebd0cc1cfaebfed72`**.
* 1,260 original template users, unchanged string IDs/order and exactly two
  columns. Each row has five distinct, positive, space-separated available IDs.
* **6,632** calendar-filtered test candidates; **2,939** have qualifying refit
  history. Repeats are allowed. Two template users have no ≥60-second unambiguous
  history and use the same popularity fallback as in validation.

## Historical refit and leakage checks

Only views within **`[2020-03-09,2020-07-27)`** are training observations. The
view log actually ends **26 July 23:59:58**. Hidden evaluation begins **27 July
00:00:00** and ends **19 October 00:00:00**, using original naive timestamps.
The refit asserts **zero hidden-period source views**, **zero hidden-period
training broadcasts/interactions**, positive programme IDs/durations and feature
ends no later than the prediction cutoff. Test schedule use is restricted to
availability and permitted metadata transformation.

No original CSV re-ingestion, join rebuilding or hyperparameter selection was
performed. The two cached databases were opened **read-only**. Original input
SHA-256s were reused after exact size/mtime checks; the original template was
independently re-read. The same corrected session logic handles duplicate logs,
concurrent-channel ambiguity, disjoint conflicting airings and uninterrupted
coverage. Model-input qualification remains **≥50% of original airing duration**,
independent of the official **≥80% evaluation target**.

| Usable historical quantity | Initial weeks 11–20 | Final weeks 11–30 | Increase |
|---|---:|---:|---:|
| Qualified distinct user–airing occurrences | 405,446 | 744,122 | 338,676 |
| Distinct user–programme pairs | 156,741 | 293,911 | 137,170 |
| Users with qualified interactions | 4,784 | 4,867 | 83 |
| Programmes with qualified interactions | 2,661 | 4,029 | 1,368 |

The rebuilt sparse matrix is **4,897 × 7,881**, including known cold identifiers
and zero-history available items. Its nonzero count is 293,911. Mappings are
rebuilt and saved in ignored `cache/final/hybrid_v1/mappings.json`; Stage-2 factor
indices were not reused.

ALS is freshly refitted: **32 factors, alpha=10, regularization=1, 12 iterations,
seed=42**, with confidence `1+10*log(1+airing frequency)`. CBF fits a **new** TF-IDF
vocabulary on **8,233 historical metadata items**: **25,210 terms**, versus the
initial 20,258, with the unchanged 40,000-feature cap. Known test metadata is
transformed with this refitted, frozen vocabulary. User profiles use normalized
log-frequency-weighted item vectors. Independent ALS/CBF source and evaluations
from Stage 2 are preserved.

Final tie-breaking is unchanged: training-popularity order, then numeric item ID.
Only positive component scores enter each Top-200 list; absent ranks contribute
zero. Cold users receive the common A fallback. No previously watched item is
filtered merely for being a repeat.

Actual refit/generation time: **14.997s**, sampled whole-process peak RSS
**2,684,878,848 bytes**. Breakdown: event/features 3.337s, repeat/popularity .053s,
ALS fit/score/save 4.935s, CBF fit/score/save 1.151s, RRF/CSV/cache save 5.465s.
No Kaggle feedback was available or used to change the selected strategy.

## Independent validation and tests

```sh
# Refit exactly the frozen winner; an existing frozen v1 is reused without refitting.
python3 -m tvrec.final_refit
# Save new test output separately, preserving old result artifacts.
python3 run_tests.py --out artifacts/final/test_results.json
# Separate validator; does not import the generation-format validator.
python3 -m tvrec.verify_final
```

The independent script re-reads the original template and queries test candidates
directly from the original schedule cache. It checks every token, exact user order,
archive identity, regenerated factor mapping/shapes and positive integer sparse
counts. It independently reconstructs **all 1,260 RRF lists** from saved component
scores using lexicographic ranking and verifies the cold fallback. It also checks
the hashes of **all 98 frozen earlier artifacts**.

**27 tests passed in .112s** after final refit. New tests prevent accidental winner
parameter changes, reject hidden-window views/broadcasts, and independently test
formatting and exact RRF arithmetic/ties. The original tests remain intact.

`artifacts/final/manifest.json` records input hashes, cutoffs, frozen strategy,
counts, parameters, availability, mappings/model artifact hashes, resources,
submission checksum, independent checks, regression results and Kaggle status.
Saved model factors, vocabulary and scoring matrices remain ignored local caches.

## Kaggle: manual action required, zero assistant attempts

There is no API token, legacy credential pair, configured `kaggle.json` or access
token; neither Kaggle CLI nor package is installed. A custom config location was
also checked. Credentials/cookies were not exposed or extracted. Consequently,
authenticated account access, existing rule acceptance and submission permission
cannot be established. **No Kaggle request/submission was made.** Submission ID,
returned public/private scores and screenshot path are null, not invented.

Use **[the exact manual upload instructions](artifacts/final/MANUAL_UPLOAD.md)**
to send the validated file once with description `midterm-hybrid-v1-rrf`.
Wait for **COMPLETE**, preserve the actual ID/score(s) and take a genuine
screenshot before reporting a Kaggle result. Do not submit the old preliminary
baseline or create another attempt because of an unexpectedly low score.

## Evidence for report preparation, no PDF

`artifacts/final/private/report_evidence/` contains byte-exact copies of the two
Stage-2 users' actual profiles/histories, method Top-5 lists with IDs/titles/
categories, weighted CBF terms, the original viewing/schedule witness and source
line checks. `two_users_exact_ap5.json` adds exact fractions, hit ranks and
arithmetic for **every method for both users**. U-heavy's hybrid example remains
`(1/1 + 2/2)/5 = 2/5 = .4`; U-light has zero hits. The genuine 3,000/3,300-second
join example and all old-versus-corrected metrics remain preserved.

All work/checks here were executed by the AI assistant; no personal student
verification is claimed. Raw data, credentials, model caches and private history
bundles stay outside public version control. The specifically requested final
submission archival CSV is prepared separately for the future repository; no
Git repository was initialized or published in this stage.

Remaining score uncertainty: official continuity/concurrent-device and frequency-
tie details are incomplete, timezone is unspecified, and same-holdout selection
can be optimistic. A valid local protocol/CSV does not establish a Kaggle score.
