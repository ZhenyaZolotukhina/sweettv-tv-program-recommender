# Stage two — repaired target, independent ALS/CBF and rank fusion

All code/checks described here were executed by the AI assistant. No calculation
is attributed to the student. No Kaggle submission, publication, PDF or final
weeks-11–30 refit was performed. The earlier README, counts, validation grid,
private examples and preliminary CSV remain available as historical evidence.

## 1. The authoritative target and the old-label error

Newly accessible **official competition documentation**:

* [Kaggle Data](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data),
  retrieved through [this text-proxy URL](https://r.jina.ai/https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data).
* [Official overview/evaluation](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/overview/description),
  retrieved through [this proxy URL](https://r.jina.ai/https://www.kaggle.com/competitions/sweettv-tv-program-recommender/overview/description).

The Data page specifies **screen-time fraction ≥0.8**, followed by frequency
counting and **strictly the five most frequent programmes per user**. It also
explicitly confirms `vsetv_id == channel_id`, resolving the previous mapping
ambiguity. The overview confirms MAP@5 and five ranked space-delimited IDs.
Its historical quota says 10 submissions/day; this does not establish the
student account's current late-submission quota. No attempt was consumed.

[Nikita Melentev's participant account](https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156)
(20 March 2021) describes **over 80% without changing channels** and a model-input
threshold above 50%. Its historical 0.41296/0.42905 scores are **not our scores**.
The linked [participant utility code](https://raw.githubusercontent.com/ndmel/2nd_place_recsys_cinema_challenge_2020/main/utils.py)
uses inclusive ≥0.8, frequency and head(5), corroborating the official page.
It was inspected as evidence; no participant code was copied.

The official **inclusive** inequality takes precedence over the article's strict
wording. A fully computed **strict >80% parallel target** is preserved. The old
target was every available programme with ≥60 seconds assigned to an airing:
it was **both too permissive in coverage and missing the Top-5 restriction**.

`artifacts/sources/relevance_sources.json` records URLs, actual excerpts,
source authority and external ideas used. The official page does not completely
define uninterrupted visits, multiple devices, or ties in frequency; those are
assumptions below, not claims of official confirmation. Its old descriptive
filenames/field names differ from the supplied CSVs; actual local schemas are
still used for parsing.

### Continuous-coverage reconstruction

`tvrec/labels.py` reads the **cached** original views; it does not reparse CSVs or
rebuild the original schedule join. A new, separately cached session audit:

1. Removes the 17 duplicate channel/start/stop viewing intervals.
2. Unions touching or overlapping same-channel intervals only when there is no
   competing channel. A positive gap, however small, ends continuity.
3. Treats simultaneous different-channel records as ambiguous: removes their
   concurrent time rather than guessing which device/channel was actually used.
   This excludes **524,325,074 user-seconds** across the full cached view log.
4. Reuses the original disjoint schedule and exact airing durations. An airing
   interrupted by a conflicting schedule row has separate runs; those runs are
   **not summed** to manufacture 80% coverage. Zero/missing IDs are never items.
5. Clips runs at the train/validation boundaries **before** thresholding. One
   qualifying distinct airing counts once per user; repeated visits separated
   by switching do not combine into a positive. Long continuous sessions may
   qualify multiple consecutive airings. A >24h-session exclusion is a sensitivity
   check, not an invented main-label rule; it changes no template user's Top-5.
6. Counts qualifying airings by programme and selects at most five IDs, frequency
   descending then numeric ID ascending. A recency-tie alternative is saved.

The sweep yields **5,283,479 unambiguous sessions** and **6,912,291 schedule runs**
in **33.73s**. This is new label-repair work, not a repeated original pipeline.
The conservative concurrent-device policy changes 430 users' target membership
relative to treating each original log row as an independent continuous viewing.
The source has no device ID, so this uncertainty cannot be resolved from the data.

### Cohort, candidate and target accounting

Dates remain unchanged: train `[2020-03-09,2020-05-18)`, validation
`[2020-05-18,2020-07-27)`. Candidates remain **6,361** positive IDs in intersecting
weeks-21–30 schedule airings. Repeats are allowed everywhere.

The primary MAP denominator is **all 1,260 original template users**, including
empty truth as AP=0. Every method uses this exact cohort, availability set and
target. There are **1,250 warm / 10 cold** users under the fixed ≥60-second
unambiguous-training-history grouping. Cold users always get the same popularity
fallback. Nonempty-only and exactly-five-truth metrics are reported as alternatives,
not silently substituted into the primary score.

| Target | Qualified distinct user–programme pairs before Top-5 | Final target pairs | Empty users | Exactly-five users |
|---|---:|---:|---:|---:|
| Old ≥60 seconds, all items | 229,811 | 229,811 | 14 | not a Top-5 target |
| Official ≥80%, continuous, Top-5 | **30,593** | **5,946** | **26** | **1,127** |
| Strict >80%, continuous, Top-5 | 30,580 | 5,946 | 26 | 1,127 |

Only four users' final target sets differ between ≥80% and >80%. **All final
target pairs are available candidates**. Two of 2,388 distinct pre-Top-5 qualified
items occur in schedule spillover outside the selected candidate file; neither
enters a user's final Top-5. Thus pre-Top-5 item coverage is 99.916%, final-target
coverage 100%. Per-user counts, including zeros, and exact target memberships
are saved privately. These are reconstructed local labels, not hidden labels.

### Isolating the score change

These rows reuse the **old trained baseline settings/predictions**, with no new
model selection. Both score columns use the **same original 1,246 eligible users**:

| Old selected baseline | Old ≥60s/all-items MAP | Corrected ≥80%/Top-5 MAP |
|---|---:|---:|
| Popularity | .777788 | .272959 |
| Repeat | .797207 | .335243 |
| Repeat + popularity | .796182 | .286231 |

On all 1,260 users the unchanged repeat predictions go from **.788349** (old)
to **.489312** (≥80% but still all qualifying items) to **.331519** (≥80% and
Top-5). This separates the two principal target errors from denominator changes.
Do not compare .797207 directly with a retuned model's corrected MAP and
attribute every difference to the target alone. Full comparisons are in
`artifacts/stage2/old_vs_corrected.csv`; old scores are reproduced with assertions
and preserved in the original `artifacts/validation_results.json`.

## 2. Independent methods and small selection grid

**Model-input thresholds are separate from target relevance.** They use training
continuous runs only, before the fixed prediction cutoff. A training airing counts
at most once, with the longest qualifying run and its end time. The modest grid
has **17 individual configurations**, rather than repeating 240 old settings:

* Popularity: qualifying-airing frequency at ≥50%, with no/35-day decay;
  or the previous ≥60-second duration-weighted/14-day-decay setting.
* Repeat: frequency at ≥5%, ≥50%, ≥80%; no/35-day decay; fraction-weighted
  alternatives at ≥50%. Common A fallback fills missing slots.
* Simple repeat/popularity baseline: per-user-max-normalized scores with
  popularity weights .1/.5. This is not counted as an advanced method.
* **Implicit MF (ALS):** three configurations: 32 factors/alpha=10 and
  64 factors/alpha=20 using ≥50% input, plus 32/alpha=10 using ≥5% input;
  regularization=1, 12 alternating iterations, seed=42 throughout.
* **CBF:** rich metadata/≥50% input, category+genre-only/≥50%, and rich
  metadata/≥5% with 35-day input decay.

### ALS score construction

`tvrec/models.py` independently implements the implicit-feedback weighted-square
objective using existing SciPy sparse matrices and SPD solvers:

`p_ui = 1[r_ui > 0]`, `c_ui = 1 + alpha * log(1 + r_ui)`;

`min Σ_ui c_ui (p_ui - x_u · y_i)^2 + λ (||X||² + ||Y||²)`.

`r_ui` is qualifying training-airing frequency. Dense latent solves use one BLAS
thread. Dot products provide candidate scores; unsupported/cold items have zero
factors and use the common fallback. Repeats are not filtered out. Unknown/absent
feedback has unit confidence, not an asserted dislike inferred from a glance.
The model is a distinct MF implementation, not popularity relabelled as CF.

Reference: [Implicit ALS documentation](https://benfred.github.io/implicit/api/models/cpu/als.html),
which documents the Hu/Koren/Volinsky implicit-feedback objective. The installed
`implicit` package is **not** used; this small exact-solve implementation uses
SciPy and has objective-decrease, latent-group and deterministic-seed tests.

### CBF score construction

Actual schedule fields: title, category, three genres, production year, director
and actors. Metadata tokens retain field prefixes; genre slots share a namespace.
Canonical metadata is the earliest permitted row per item. TF-IDF vocabulary/IDF
are fitted only on **5,842 training-period metadata items**, capped at 40,000
features (selected rich model actually has **20,258**). Later known schedule
metadata is transformed with this frozen vocabulary to score unseen IDs.

A user profile is the L2-normalized sum of item vectors weighted by
`log(1 + qualifying-airing frequency)`. Cosine similarity gives candidate scores.
No-history/zero-vector cases use A. Tests prove an unseen ID with known metadata
can score positively and a future-only token never enters the fitted vocabulary.
Reference: [scikit-learn TF-IDF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

### Fair individual comparison

| Selected method | Official ≥80% Top-5 MAP | Strict >80% MAP | Warm MAP | Cold MAP |
|---|---:|---:|---:|---:|
| Popularity, ≥50% frequency | .295009 | .294646 | .296345 | .128000 |
| Repeat, ≥50% frequency | .366007 | .365552 | .367911 | .128000 |
| Simple repeat + popularity, weight .1 | .368704 | .368249 | .370630 | .128000 |
| ALS, 32 factors/alpha 10 | .165965 | .165741 | .166269 | .128000 |
| Rich-metadata CBF, ≥50% frequency | .257626 | .257388 | .258663 | .128000 |
| **Selected rank fusion** | **.373631** | **.373229** | **.375596** | **.128000** |

ALS and CBF are weaker individually. They are still separately implemented,
tested and measured, and provide small complementary ranking signals in fusion.
Cold-group MAP is based on ten users and is not a stable population claim.

## 3. Selected fusion and robustness

**26 explicit fusion/switch settings** compare per-user-max-normalized scores,
weighted reciprocal ranks, and sparse-user switching. Raw CF dot products and
CBF cosine scores are never directly added as if interchangeable.

The selected strategy is weighted reciprocal-rank fusion of each component's
positive-score **Top-200**:

`RRF(u,i) = 1/(10 + rank_repeat) + 0.1/(10 + rank_CF)`

`           + 0.1/(10 + rank_CBF) + 0.1/(10 + rank_popularity)`.

Missing component ranks contribute zero. Ties use training popularity, then
numeric programme ID. Cold users receive the same A list, and all outputs are
distinct and availability filtered. Components all use **≥50% model-input
coverage, qualifying-airing frequency, no input decay**. ALS is **32 factors,
alpha=10, regularization=1, iterations=12, seed=42**. CBF uses rich metadata as above.

Against repeat alone, main-holdout gain is **+.007624 MAP**, with 223 users
improving and 90 worsening. A seed-42 paired bootstrap gives [.005139, .010218],
but does **not** correct same-holdout model-selection optimism. Against the simple
repeat/popularity baseline, gain is +.004927.

The exact main-selected settings were frozen and applied to a retrospective
earlier split: train `[2020-03-09,2020-04-13)`, validate
`[2020-04-13,2020-05-18)`. Same 1,260 user IDs; 4,412 then-available candidates;
51 zero-positive users. No settings were tuned on this earlier window:

| Method | Earlier MAP@5 |
|---|---:|
| Popularity | .290427 |
| Repeat | .381888 |
| Repeat + popularity | .383161 |
| ALS | .225822 |
| CBF | .290187 |
| Frozen main-selected RRF | **.387936** |

This supports temporal stability of the modest fusion gain, but is not a fresh
independent test: main settings were selected on the later holdout. Neither
window used Kaggle labels, historical leaderboard values or future views to fit.

**Next-stage refit configuration:** use `artifacts/stage2/selected_strategy.json`.
Refit exactly these components on allowed weeks 11–30, preserving session,
interaction, TF-IDF and score definitions; use weeks-31–42 schedule availability
and permitted metadata. Perform final formatting checks before any explicitly
requested Kaggle attempt. The old readiness CSV is not this strategy's output.

## 4. Runtime, memory, tests and evidence

* Continuous-session/cache construction: **33.73s** once.
* Successful complete experiment, including audit, 43 main configurations and
  frozen earlier check: **89.57s**, reusing the three already fitted main ALS
  arrays; peak **2,953,428,992 bytes RSS**. Do not call cached ALS loading fit time.
* A separate winning-ALS fresh-fit benchmark took **5.006s**, peak
  **1,709,686,784 bytes RSS**, including cached-event read, feature aggregation
  and the 4,851 × 7,137 sparse matrix with 156,741 nonzeros. It was run once to
  persist missing fit-memory evidence, not for further tuning.
* Selected CBF feature/profile/score generation took **1.147s**, with whole-process
  sampled RSS **2,667,331,584 bytes** in the multi-model run. Repeat/popularity
  feature/score generation took about .49s each; selected RRF including ranking
  took .929s. Per-model process RSS includes resident shared data and other models;
  it is not an isolated allocation estimate. Baseline/model times exclude shared
  preprocessing and their separate ranking/evaluation unless explicitly stated.
* **23 deterministic tests passed** in .089s. Tests cover exact strict/inclusive
  boundaries, duplicates, gaps, switching, concurrency, disjoint schedule
  conflicts, independent endpoint coverage, actual production cutoff clipping,
  frequency Top-5, MF optimization, CBF vocabulary leakage, score-scale
  invariance, common fallback, the original AP cases and submission formatting.

One initial partial experiment timed out because an accounting expression rebuilt
`set(f.item)` once **per candidate**. This was diagnosed in the code and moved
outside the loop. Already fitted ALS arrays and predictions were reused. That
failure and its correction are recorded in `AUDIT.md`; successful CBF timings
exclude the corrected redundant bookkeeping, not an omitted modeling step.

### Reproduce using existing caches

```sh
python3 -m pip install -r requirements.txt
python3 run_tests.py --out artifacts/stage2/test_results.json
python3 -m tvrec.advanced audit
python3 -m tvrec.advanced run
python3 -m tvrec.verify_stage2
# Optional isolated resource benchmark, only when needed:
python3 -m tvrec.verify_stage2 --benchmark-cf
```

The new runner requires `cache/data.duckdb` and attaches it **read-only**. It
never calls the old pipeline. `cache/stage2.duckdb` contains audited sessions/runs;
a manifest hashes session code and original cached data accounting. ALS arrays
are cached by model code, configuration and window. Current implementation
semantics must be kept with its recorded hashes; invalidate caches if changing
feature rules. All seeds/dependencies and execution hashes are recorded.
`requirements.txt` pins SciPy/scikit-learn/threadpoolctl/psutil in addition to the
existing DuckDB/NumPy/pandas dependencies; Python 3.14.0/macOS arm64 were used.

### Saved files

Public/non-user-identifying summaries in `artifacts/stage2/`:

* `label_audit.json`, `session_accounting.json`, `old_vs_corrected.csv`.
* `main_individual_results.json`, `main_best_individual_configs.json`,
  `fusion_results.json`, `leaderboard.json/.csv`, `selected_strategy.json`.
* `method_label_sensitivity.csv`, `paired_gain.json`, `earlier_robustness.json`.
* `execution_audit.json`, `execution_run.json`, `cf_fresh_fit_benchmark.json`,
  `test_results.json`, `independent_verification.json`, `verification_environment.json`.
* `representative_user_summary.json`: **same two users as stage one**, identified
  only by U-heavy/U-light here. U-heavy has five corrected truth items; U-light
  has one. Their exact identities remain in ignored private files.

Ignored private evidence:

* `truth_versions.json`, `truth_counts_per_user.csv`, target-airing frequencies.
* **43 main method/configuration CSVs**, each with all 1,260 string user IDs,
  exact Top-5, truth, hit positions, AP contribution and warm/cold group. Earlier
  method predictions are also retained. Independent checking recomputed every
  main AP and hit list and checked identical IDs, order, candidates and truth.
* `coverage_witness.json`: source view row **3,913,423**, schedule source row
  **656,405** of weeks 11–20, programme **10002079**,
  **“Спартак: Кровь и песок”** (Сериалы). On 18 May, clipping at midnight gives
  **3,000/3,300 = 90.909%** contiguous coverage; `5×3000 > 4×3300` exactly.
  An independent endpoint-enumeration checker reads the cached original source
  rows and recomputes this result without the production join/session functions.
* `representative_users.json`: complete corrected train/validation run histories,
  actual IDs, each method's ordered recommendations, real titles/categories and
  AP for the same heavy/light users. `cbf_profile_terms.json` preserves their
  actual weighted TF-IDF terms and source programme metadata. U-heavy has hybrid
  AP=.4 versus popularity AP=.71; U-light is missed by every method. The hybrid
  is not universally superior or a guarantee of satisfaction.
* `actual_manual_ap5.json`: U-heavy's hybrid list hits ranks **1 and 2** against
  five relevant IDs, so **(1/1 + 2/2)/5 = .4**. Exact programme IDs are stored for
  independent reproduction, along with the original synthetic AP example.

### Remaining ambiguity

Frequency-tie handling changes 647 users' target membership under the saved
recency alternative; concurrency interpretation changes 430. These uncertainties
are larger than the inclusive/strict difference (four users). Cumulative visits
allowed as a sensitivity-only counterexample change 300 users; that policy is
never the main uninterrupted target. Original timezone uncertainty remains.
No official scorer source or hidden labels were obtained. These limitations and
single-holdout selection optimism qualify every local score above.
