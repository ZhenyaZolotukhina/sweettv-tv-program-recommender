# SWEET.TV TV Programme Recommender

**HSE University · LLM4Rec Midterm · October 2026**  
**Student:** Evgeniya O. Zolotukhina · **Kaggle:** `genie03`

A reproducible study of **Top-5 TV programme recommendation** from implicit viewing histories and broadcast schedules. The final system combines multiple independent candidate generators with a user-grouped **LightGBM LambdaRank** reranker. Models are compared on chronological holdouts, and the submitted file contains exactly five ranked programme IDs for each of 1,260 users.

## Final result

| Selected submission | Private MAP@5 | Public MAP@5 |
|---|---:|---:|
| **V5 · Hybrid + LambdaRank** | **0.43385** | **0.44580** |

The instructor clarified that the **best Private Score** is used for midterm Kaggle grading. These are scores returned by the Kaggle competition, not offline estimates.

**Competition:** [SWEET.TV — TV program recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender)  
**Final submission:** [`artifacts/final/submission_hybrid_v5.csv`](artifacts/final/submission_hybrid_v5.csv)  
**SHA-256:** `249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f`

## Problem and approach

User viewing intervals are joined to the broadcast schedule by **channel ID and overlapping timestamps**. A relevant target airing must have at least **80% uninterrupted, unambiguous coverage** of its original programme duration. For each user, the ground truth contains the five most frequent qualifying programmes. Schedule entries with programme ID `0` are not recommendable. No future user viewing is used when constructing training features.

The project independently evaluates:

- **Popularity**, as the mandatory reference baseline, and **repeat-history** ranking.
- **Implicit ALS matrix factorization**, trained on historical user–programme interactions.
- **Content-based filtering**, using metadata TF-IDF and historical user profiles.
- A **hybrid candidate pool** incorporating personal repeats, CF, content similarity, popularity and channel preferences.
- **V5 group-aware LambdaRank**, which reranks candidates for MAP@5 evaluation.

Additional controlled studies evaluated completion/scheduling features, BPR collaborative filtering and programme co-view association rules. They did not consistently improve both holdouts, so **V5 was retained**. LightFM/WARP was not successfully tested and is not claimed as a result.

## Reproducible evaluation

| Strategy | Main holdout MAP@5 | Earlier holdout MAP@5 |
|---|---:|---:|
| Popularity | 0.295009 | 0.290427 |
| Repeat history | 0.366007 | 0.381888 |
| Implicit ALS | 0.165965 | 0.225822 |
| Metadata CBF | 0.257626 | 0.290187 |
| **Selected V5** | **0.395679** | **0.408858** |

- **Earlier holdout:** history before 2020-04-13; evaluation 2020-04-13 to 2020-05-18.
- **Main holdout:** history before 2020-05-18; evaluation 2020-05-18 to 2020-07-27.
- **Hidden competition target:** future schedule 2020-07-27 to 2020-10-19; viewing outcomes unavailable to training.

For the environment, data layout, verified commands and the distinction between cached-stage reproduction and a complete replay, see **[`REPRODUCE.md`](REPRODUCE.md)** and [`requirements-lock.txt`](requirements-lock.txt). Original competition CSV files are intentionally not redistributed.

### Independent numerical witnesses

The release includes two real numerical exercises covering interval overlap and AP@5. From the project root, run:

```bash
python3 -m tvrec.release_checks --exercise join
python3 -m tvrec.release_checks --exercise ap
```

Expected results: **3,000 / 3,300 = 90.9091%** coverage (qualifies at 80%) and **AP@5 = 11/20 = 0.55** for an actual holdout user. The student separately recalculated both values and ran the scripts; that run did **not** independently read the original raw CSV source lines. See [`STUDENT_CHECKS.md`](STUDENT_CHECKS.md).

## Repository guide

| Location | Description |
|---|---|
| [`tvrec/`](tvrec/) | Source implementation and reproducibility checks |
| [`artifacts/final/submission_hybrid_v5.csv`](artifacts/final/submission_hybrid_v5.csv) | Exact final Kaggle submission |
| [`REPORT_EVIDENCE.md`](REPORT_EVIDENCE.md) | Full evaluation, programme examples, negative results and source attribution |
| [`STUDENT_CHECKS.md`](STUDENT_CHECKS.md) | Two source-backed numerical verification exercises |
| [`FINAL_CHECKS.md`](FINAL_CHECKS.md) | Regression and repository audit; remaining scope limits |
| [`REPRODUCE.md`](REPRODUCE.md) | Setup and execution instructions |

**Integrity:** The coding agent reported 70 passing regression tests and an exact native V5 reload. These are agent-executed checks; a complete fresh-data replay was not rerun during final release packaging. Limitations include sparse programme IDs in broadcasts, historical selection on reused holdouts and non-observation not implying dislike.

## Methodology and attribution

The two-stage idea was informed by [Nikita Melentev's SWEET.TV write-up](https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156); ALS follows the [implicit-feedback framework of Hu, Koren and Volinsky](https://doi.org/10.1109/ICDM.2008.22). The final learning-to-rank implementation uses [LightGBM](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html). Implementation and experiments were AI-assisted using OpenCode and OpenAI Codex; methodology, audit trail and limitations are disclosed in the report and evidence files.

This repository contains no original bulk Kaggle dataset or access credentials. The native AI-session export is submitted privately to the course, not published here.
