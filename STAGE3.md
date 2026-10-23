# Stage three — targeted channel-ranking improvement, no Kaggle submission

## Frozen v1 and monitoring provenance

The user reports a successful v1 Kaggle upload, description
`midterm-hybrid-v1-rrf`, public MAP@5 **.42158**, private **.40693**. These are
recorded in `artifacts/monitoring/kaggle_v1_user_report.json` as **user-reported
results**, not independently retrieved evidence. No submission ID or screenshot
was supplied. The scores were never read by experiment/model-selection code and
are not optimization targets. No Kaggle requests were made in this stage.

V1 CSV remains byte-identical, SHA-256:
`d4026ac73d6f940010d7be3f8241731bd5975054fc3127eebd0cc1cfaebfed72`.
Its archive, configuration, manifests, parameters and Stage-1/2 results are all
preserved. `artifacts/stage3/frozen_prior.json` hashes **115 prior files**;
generation and independent validation confirmed none changed. Earlier manifests
describe their creation-time state; the monitoring addendum records the later
user upload without rewriting history. Required ALS and CBF implementations and
their separate evaluations remain untouched.

## Fixed offline protocol

Same reconstructed official-style target: **≥80% uninterrupted coverage**, once
per user/airing, then the five most frequent programme IDs (ID-ascending ties).
All **1,260 users**, including empty targets, remain in every denominator.

* Main: train `[2020-03-09,2020-05-18)`, validate `[2020-05-18,2020-07-27)`;
  **6,361 candidates**, 1,250 warm / 10 cold users.
* Earlier: train `[2020-03-09,2020-04-13)`, validate `[2020-04-13,2020-05-18)`;
  **4,412 candidates**, 1,247 warm / 13 cold users.
* Input coverage stays **≥50%**, distinct-airing frequency, no decay for the
  retained v1 components. The fixed ≥60-second history grouping and common
  popularity fallback are unchanged. Repeats remain eligible.

V1 predictions were reproduced **exactly**, not just approximately matching MAP.
The independent checker also verified every experiment's user order, candidate
universe, target membership, warm/cold groups, AP contributions and hit ranks
against the frozen Stage-2 files. All 22 window/configuration CSVs passed.

Cached joins and ALS scores were reused; only the two small TF-IDF scorings needed
to reconstruct validation v1 were rerun. **No new ALS fit, raw CSV re-ingestion or
schedule/view join rebuild occurred.** Final v1 ALS/CBF models already use weeks
11–30, so they are also reused for v2's unchanged components.

## Small, documented experiment budget

The initial plan was saved before running experiments. It contains eight new
ablation/standalone configurations plus v1. Each feature is computed from
pre-cutoff history only; the opportunity and habit experiments use the permitted
target schedule, never future user behavior.

| Configuration | Main MAP@5 | Earlier MAP@5 |
|---|---:|---:|
| **Frozen v1** | **.373631** | **.387936** |
| Mild recency replaces repeat ranking | .366756 | .378124 |
| Confidence-weighted repeat ranking | .371068 | .384797 |
| Schedule-opportunity-adjusted repeat ranking | .370931 | .384595 |
| Channel component alone | .358992 | .366082 |
| Time-of-week component alone | .002368 | .001825 |
| V1 + channel, weight .1 | .374755 | .389882 |
| V1 + time-of-week, weight .1 | .373398 | .387970 |
| Confidence + opportunity replacement | .375137 | .387944 |
| V1 + channel, weight .3 (follow-up) | .378537 | .392103 |
| **V1 + channel, weight .5 (selected v2)** | **.380361** | **.393930** |

Exact feature definitions:

* **Recency:** `frequency * (.75 + .25 * exp(-ln(2)*days_since_last/35))`.
  This mild floor avoids discarding old repeats, but still hurt both windows.
* **Confidence:** number of ≥50% training airings plus number of ≥80% training
  airings. This is an input confidence feature, not a target-definition change.
* **Opportunity:** repeat score multiplied by
  `sqrt(clip((future_airings+4)/(past_airings*future_days/past_days+4),.25,4))`.
  Future quantities are known schedule airings only. The combined confidence/
  opportunity variant improved the main window slightly but not the earlier
  window meaningfully; it was rejected.
* **Channel:** normalize the square root of each user's qualifying channel
  frequencies. Multiply these channel affinities by programme popularity
  conditional on channel, learned from all training users' qualified airings.
  Denominators include all historical labelled programmes, not just target
  candidates. Unseen items have zero channel popularity. These are heuristic
  affinities/conditional frequencies, not calibrated preference probabilities.
* **Time-of-week:** 42 weekday × four-hour slots using the original naive clock.
  User viewing-midpoint counts have a fixed Laplace-1 prior; known future programme
  airing-midpoint distributions provide compatibility. It was ineffective alone
  and did not improve both windows in fusion, so it is not in v2.

The channel addition was the only initial feature with a consistent improvement
of at least .001 in both windows. Its main paired interval still crossed zero,
so **only two stronger weights, .3 and .5**, were tested next. The follow-up plan
required positive gains ≥.001 and positive paired-bootstrap lower bounds in
**both** windows; otherwise v1 would be retained. No further weight expansion was
made after this bounded follow-up. This is ten new configurations total, not a
large hyperparameter sweep.

## Selected exact v2 strategy

Keep frozen v1 unchanged, then add:

`score_v2(u,i) = score_v1(u,i) + 0.5 / (10 + rank_channel(u,i))`.

Only the channel component's positive-score **Top-200** contribute; absent ranks
contribute zero. V1 retains repeat/ALS/CBF/popularity weights **1/.1/.1/.1**,
component Top-200, k=10. Tie-breaking remains training popularity then item ID.
Cold users get exactly the v1 popularity fallback. Scores from different methods
are fused by rank rather than by incompatible raw score scale.

| Paired comparison with v1 | Main | Earlier |
|---|---:|---:|
| Absolute MAP gain | **+.006730** | **+.005994** |
| Warm-user v1 → v2 MAP | .375596 → **.382380** | .391980 → **.398036** |
| Cold-user v1 → v2 MAP | .128 → .128 | 0 → 0 |
| Improved / worsened users | 306 / 196 | 303 / 206 |
| Paired 95% bootstrap interval | [.002926, .010436] | [.002096, .009988] |

Bootstrap: 1,000 resamples of users, seed 42. Both windows were used during
selection, including the adaptive two-weight follow-up. These intervals **do not
correct selection optimism**, and neither window is now an untouched test.
They support a measured candidate for a future attempt, not a promised Kaggle gain.

## Where the gain comes from, and an observed failure

The channel component mostly improves familiar-programme ranking rather than
discovery. Main-window lists introduce 147 relevant IDs and remove 100; only five
of the introduced relevant IDs lack ≥50% history for that user. Of 1,016 changed
lists, 405 change only order. Earlier: 135 relevant IDs introduced, 106 removed,
seven without qualifying history; 427 lists change only order.

Training-derived segment analysis:

* Users with ≥10 history programmes gain **+.006958** main / **+.008391** earlier.
* Sparse warm users (<10 programmes) gain **+.003441** main but lose **−.007607**
  earlier. This is an actual counterexample to universal improvement.
* Broad-channel users (no channel ≥60% of qualified history) gain **+.007102** /
  **+.008193**. Channel-focused users gain +.004177 main but lose −.005755 earlier.
* Cold users are unchanged by design. Their groups are too small for strong
  population conclusions.

Two deliberately contrasting actual users were selected **after evaluation** for
diagnosis, not for feature fitting or representative population claims:

1. **Largest AP improvement:** .146667 → .55. V1's relevant hits were at ranks 3
   and 5: `(1/3 + 2/5)/5 = 11/75`. Channel ranking promotes **“Факты. День.”** and
   **“ТСН. Неделя.”** to ranks 1 and 2 and introduces **“Сегодня. Итоги с Олегом
   Панютой.”** at rank 4: `(1 + 1 + 3/4)/5 = 11/20`. Historical series such as
   “Мамочки” and “Кухня” had larger repeat counts but much smaller pooled channel
   scores than the news programmes.
2. **Largest regression:** 1 → .6. The channel signal displaced relevant
   **“Гражданская оборона.”** and **“Утро с Интером”** with non-target news shows.
   The former had five historical qualified airings but channel score .00948,
   below the promoted “Сегодня…”/.02776 and “Викна-Новости”/.02767. This measured
   failure illustrates pooled channel popularity overriding individual specificity.

`artifacts/stage3/private/contrasting_users.json` contains exact identities,
historical programmes, channel profiles, programme IDs/titles/categories, both
Top-5 lists, target sets and exact AP fractions. A public summary uses aliases.
The same original two Stage-2 representative users are also retained: their AP
changes .4 → .55 and 0 → 0. These checks were AI-executed, not student verification.

## Final v2 file, resources and independent checks

**`artifacts/final/submission_hybrid_v2.csv`**

SHA-256: **`987f8fde48654ba982c20b36e25bf78017749c58b15d96bccdbc8eca5f7d2671`**.

The new channel component was refitted on all allowed weeks 11–30 using **744,122
qualified user–airing occurrences**, 4,867 users and 4,029 programmes. The existing
v1 final ALS/CBF scores were hash-verified and reused because their settings,
historical inputs, mappings and 6,632-item test universe are identical. There is
no reason to refit those unchanged models again. Two final cold users keep v1's
fallback. No hidden-window source views or broadcasts enter channel training.

Actual runtime/resource measurements:

* Initial two-window experiment: **31.83s**, peak **2.52 GB** RSS.
* Two-weight follow-up, both windows: **10.13s**, peak **2.30 GB**.
* Incremental main channel feature/rank construction: **.651s**; earlier **.302s**.
  Final list ranking is ~.949s main, versus v1's measured ~.904s; these are
  separate runs and not a controlled microbenchmark.
* Final channel refit + v2 generation: **5.345s**, peak **2.32 GB**, reusing fitted
  v1 models. Channel feature generation/count saving takes 2.830s, final ranking/
  CSV writing 1.015s. V1's 14.997s full refit includes ALS/CBF fitting, so it is
  not an apples-to-apples speedup claim.

**33 tests pass.** New deterministic tests cover channel conditional probabilities,
sqrt profile weighting, non-renormalization when candidates change, zero-history
users, uninterrupted qualification/deduplication, cutoff rejection, confidence,
recency, weekday bins, schedule-only habits and opportunity smoothing.

The independent checker uses SQL `DISTINCT user,airing` qualification, separate
channel-count reconstruction, and independent final rank calculation. It verifies
all 1,260 output IDs/order, five distinct positive available IDs per row, all
channel scores and final rankings, and every one of the 22 offline configuration
files' fixed truth/denominator/group/AP/hit positions. V1 and all prior evidence
remain hash-unchanged. No network or Kaggle submission occurs in any new runner.

## Files and reproduction

* Feature code: `tvrec/rank_features.py`; experiments: `tvrec/improve.py`.
* New-component final refit: `tvrec/channel_v2.py`; separate checker:
  `tvrec/verify_v2.py`; diagnostics: `tvrec/improvement_evidence.py`.
* Plans, comparisons, selection, uncertainty, segment tables, source hashes and
  timings: `artifacts/stage3/`. Per-user predictions/history evidence are in its
  gitignored `private/` directory. New cached aggregates/scores are under ignored
  `cache/improvements/`.
* Full v2 manifest:
  `artifacts/final/submission_hybrid_v2.manifest.json`.

To inspect/check the existing candidate without fitting models:

```sh
python3 run_tests.py --out artifacts/stage3/test_results.json
python3 -m tvrec.verify_v2
```

To reproduce the controlled experiments into a new evidence directory instead of
overwriting the original measured records:

```sh
python3 -m tvrec.improve experiment --out artifacts/stage3_reproduction
python3 -m tvrec.improve refine --out artifacts/stage3_reproduction
```

`python3 -m tvrec.channel_v2 prepare` generates v2 from the original selected
record if absent; an existing v2 is reused unchanged. It refuses a cosmetic v2
when selection retains v1. Exact preprocessing/model prerequisites are the
preserved Stage-1/2/final-v1 caches and pinned dependencies; no new dependency was
introduced. Floating-point reproducibility is tied to the recorded environment.

No GitHub publication, final PDF or Kaggle attempt was made. Original session
history remains in the harness; no replacement export or screenshot is fabricated.
