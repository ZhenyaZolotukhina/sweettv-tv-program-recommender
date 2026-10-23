# New collaborative signal investigation — RETAIN V5, no V7 CSV

## Source review and bounded installation failure

Reviewed the actual [Nikita Melentev prize-winning second-place account](https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156),
published 20 March 2021. It describes repeat + LightFM/WARP candidates followed
by a boosting layer, with latent score/rank, user/item biases and schedule-change
features. Its historical Kaggle results are not this project's measurements or
tuning targets. No participant code was copied.

One bounded `lightfm==1.17` installation attempt failed while obtaining build
requirements on Python3.14/macOS: `AttributeError: dict object has no attribute
__LIGHTFM_SETUP__`. It did not reach wheel compilation. No setup patch, system
dependency fight, Python downgrade or repeat install was attempted.

Used the requested fallback: independently implemented **implicit BPR matrix
factorization** with existing NumPy/SciPy. This is **not LightFM/WARP**, and these
results do not establish whether working LightFM/WARP would help.

Attribution and exact adaptations/deviations are saved in
`artifacts/sources/new_cf_methodology.json`. References actually inspected:
[LightFM model/loss documentation](https://making.lyst.com/lightfm/docs/lightfm.html)
and [Rendle et al., BPR, UAI2009 / arXiv2012](https://arxiv.org/abs/1205.2618).

## BPR model and temporal correctness

`score(u,i) = p_u · q_i + item_bias_i`.

Training maximizes pairwise preference differences using logistic BPR loss,
sampled L2 and sparse per-entity minibatch AdaGrad. It is an independently written
optimizer, not claimed identical to original SGD or LightFM. Sampled unobserved
pairs are weak relative comparisons, **not confirmed dislikes**. Known positive
pairs are never sampled as negatives; only historical interacted items are in
the training/negative-sampling universe, not future-only programme IDs.

**User bias cancels from pairwise differences and is unidentifiable here**; it
was omitted rather than fabricated. Item biases, latent scores/ranks, cosine
similarity and embedding norms are genuine available signals.

Two configurations only: binary versus log-frequency positive sampling. Both use
32 factors, 25 epochs, batch2048, learning rate.05, regularization.002, seed42.
Users are sampled uniformly; the second configuration samples their positive
programmes proportional to `log(1+qualified airing frequency)`.

Inputs retain **≥50% uninterrupted airing overlap** and one count per distinct
user/airing. Evaluation retains **≥80% completion then frequency-selected Top-5**,
ID ties, all1260 users/empty AP0 and the same date/availability policies. Repeats
are allowed; every method uses the original global fallback for cold users.

Every historical model/mapping/popularity statistic is fitted at its own cutoff.
Two additional early BPR models were fitted before Mar23 and Apr6 for supervised
examples whose labels end Apr13. Main/earlier factors were **not** reused in those
early folds. Later-user/item identities or vocabularies were not learned as early
features. BPR has no metadata vocabulary. Sparse fallback popularity for early
tasks is recovered from that task's original count matrix, not later values.

## Standalone CF and genuinely new retrieval information

| Method | Main MAP@5 | Earlier MAP@5 |
|---|---:|---:|
| Frozen implicit ALS | .165965 | .225822 |
| New BPR, binary positives | .270489 | .290652 |
| **New BPR, log-frequency positives** | **.311584** | **.308569** |
| Frozen V5 final stack (reference) | .395679 | .408858 |

These compare the tested settings, not an assertion that BPR is universally
superior to tuned ALS. Full original popularity/CBF/ALS individual comparisons
remain immutable in Stage2. New CF's Top-5 files and examples are saved separately.

| Micro candidate recall/complementarity | Main | Earlier |
|---|---:|---:|
| ALS Top-200 recall | .83535 | .84881 |
| V5 post-ranking Top-200 recall | .87117 | .88934 |
| **BPR log-frequency Top-200 recall** | **.87639** | **.89692** |
| Relevant pairs BPR retrieves outside ALS Top-200 | **370** | **340** |
| Relevant pairs BPR retrieves outside V5's entire old pool | **220** | **134** |

BPR/ALS mean Top-200 Jaccard is .348/.399. This passes the predeclared information
gate, so the study did not stop merely because standalone CF was below V5.

## Controlled integration — new information did not become a robust Top-5 gain

Selected log-frequency BPR, then appended exactly eight features to V5's 59:
raw score, eligible-item reciprocal rank, learned item bias, latent cosine,
user/item embedding norm and explicit user/item-supported indicators. Missing
embeddings are flagged; no fake user-bias column is supplied. Raw scores may be
negative and are not probabilities.

The original same-pool variant keeps every candidate, seed, ranker parameter,
temporal training label and evaluation convention otherwise unchanged. Ranker
supervision uses the same two disjoint pre-Apr13 target windows, never either
reported main/earlier holdout. Both weight strategies' source models were fitted
before each respective CF prediction cutoff.

The original proposed Top-50 expansion added only **3/7** relevant pairs and was
rejected. That gate/plan remains preserved. Because the independently measured
novelty was at Top-200, one explicitly disclosed adaptive **Top-200 tail expansion**
was then tested. This was the second and last downstream variant, not another
CF configuration/weight search. Existing 59-feature rows were reused byte-wise;
only new pairs were rebuilt. Three old candidates/query were independently
reconstructed as identity checks and matched their cached controls.

| Strategy | Main MAP@5 | Earlier MAP@5 |
|---|---:|---:|
| **Frozen V5** | **.395679** | **.408858** |
| Same pool + BPR features | .392906 | .408882 |
| Expanded Top-200 union + same features | .394048 | .408347 |

| Expanded variant − V5 | Main | Earlier |
|---|---:|---:|
| Gain | −.001631 | −.000511 |
| Paired 95% bootstrap interval | [−.004628,+.001528] | [−.003998,+.003047] |
| Improved / regressed users | 219 / 224 | 214 / 206 |
| Heavy-history gain | −.002265 | −.000301 |
| Light-history gain | +.010242 | −.001780 |
| Repeat-heavy gain | −.000025 | +.007557 |
| Cold MAP | .128 unchanged | 0 unchanged |

The same-pool variant loses .002774 main, with CI[−.005661,+.000085], and changes
earlier MAP only +.000024, CI[−.002540,+.002687]. It also harms main repeat-heavy
users by .005010. No downstream variant qualifies; **retain V5**.

Bootstrap:1000 paired users, seed42, not corrected for choosing CF weighting or
the adaptive tail follow-up. No selected interval is treated as unbiased proof.

### Why recall alone was insufficient

Expanded pool recall increases **.87185→.90885** main and **.88952→.91313** earlier;
mean sizes222.05→300.95 and219.29→287.26. Nevertheless, **none of the relevant
items newly added outside the original pool reach the expanded model's Top-5**.
Top-5 relevant hits fall2817→2803 main and2840→2839 earlier. This is actual evidence
that retrieval novelty was not exploited by the tested reranker, not evidence
that no collaborative information exists.

Posthoc raw latent scales shift with history length: median user-factor norm is
.117/.189 in the early supervised tasks versus .280/.654 earlier/main. Rank and
cosine features are also present, but this is a possible transfer/calibration
limitation, not a proven causal diagnosis. BPR's pairwise observed/unobserved
criterion is also not identical to WARP precision focus or future frequency
Top-5 relevance. No third feature/calibration/epoch search was run.

Real diagnostic cases are preserved, with history, actual programme IDs/titles/
categories and rational AP:

* Correction **.52→.80**: “Следствие вели…” enters and “Утро с Интером” moves up
  while “Говорит Украина” leaves the list. This local success does not rescue
  the aggregate negative result.
* Failure **1→.76**: non-target “Мир наизнанку” displaces relevant “Викна-Новости”
  and moves “Сегодня…” down. A previously perfect list becomes imperfect.

The same original two representative users' method/profile records remain in
`private/original_representatives.json`. The required original individual method
comparisons, join example, exact AP examples and session history remain unchanged.
No new calculations are attributed to student-performed verification.

## Costs, independent verification and files

* Four standalone CF fits + evaluations: **44.15s**, peak **1.75GB**. Selected
  main/earlier BPR fits10.86s/5.35s; score generation about.015s/.013s.
* Same-pool integration, including two own-cutoff early CF fits: **17.67s**, peak
  **1.43GB**; ranker fit5.37s.
* Bounded expansion/rebuilt new-pair features + second ranker: **42.35s**, peak
  **2.46GB**; ranker fit6.69s. Expansion reused CF fits rather than fitting again.
* **65 tests pass**, including pairwise-loss reduction, seeded factor
  reproducibility, exclusion of observed pairs and saturated users from negative sampling,
  frequency sampling, bias+dot reconstruction, honest missing/negative features,
  and early-only fallback statistics, alongside all original checks.
* Independent checker rebuilt positive matrices by DISTINCT-airing SQL, matched
  all four standalone models, recomputed scores directly from factors/biases and
  reconstructed all four downstream files' rankings/AP/groups/cold fallback.
  It verified rolling cutoffs, unchanged old evidence and absence of V7 CSV.

New code: `bpr_model.py`, `new_cf.py`, `new_cf_rerank.py`, `new_cf_expand.py`,
`new_cf_evidence.py`, `verify_new_cf.py` under `tvrec/`. Plans, resources, cutoff
records, standalone/recall/complementarity results, failed integration decisions,
segments and checks are in `artifacts/stage8/`. Personal histories/predictions
stay in its ignored `private/`; model mappings/factors stay in ignored
`cache/new_cf/`. No new runtime dependency was required for fallback BPR.

Completed commands (do not repeat fits unnecessarily):

```sh
python3 -m tvrec.new_cf standalone
python3 -m tvrec.new_cf_rerank
python3 -m tvrec.new_cf_expand
python3 -m tvrec.new_cf_evidence
python3 run_tests.py --out artifacts/stage8/test_results.json
python3 -m tvrec.verify_new_cf
```

**No V7 final model refit or CSV was created.** Frozen V5 remains the recommendation
and best submitted/private-grading model (.43385, user-verified history). Five
attempts have already occurred; remaining capacity is unknown. No upload was
attempted, no hidden Kaggle score was tuned, no PDF or GitHub publication was
created, and the native AI-session export was not altered.
