# SWEET.TV — final report evidence

**Selected: V5 LambdaRank. Verified Kaggle PRIVATE MAP@5: 0.43385.**
Final CSV: `artifacts/final/submission_hybrid_v5.csv`; SHA-256
`249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f`.
Deadline: 24 October 2026, 19:00 Moscow (16:00 UTC). Optimization is closed.
Kaggle results below are the user's verified history, not assistant API retrieval.
No submission IDs or screenshots were supplied. The later instructor clarification
(best PRIVATE graded) supersedes the original quiz's public-score wording.

## 1. Data, join and chronological protocol

| Original CSV | Data rows | Nonzero programmes | Labelled / unlabelled schedule rows |
|---|---:|---:|---:|
| dataset11-30.csv | 5,821,987 | — | — |
| export_arh_11-20-final.csv | 1,074,235 | 5,847 | 115,613 / 958,622 |
| export_arh_21-30-final.csv | 1,074,394 | 6,363 | 121,505 / 952,889 |
| export_arh_31-42-final.csv | 1,295,378 | 6,636 | 144,774 / 1,150,604 |
| submission.csv | 1,260 users | five placeholders/user | — |

The view log has 4,914 users and 202 channels; schedules have 400/403/404
channels. Source hashes/schema are in `artifacts/dataset_accounting.json`.
Only 10.76% of the first schedule's rows have an item ID. Zero/missing programme
IDs are NULL, never recommendable items; unlabelled airtime participates in the
join/conflict policy but creates no item interaction. Five viewed channels lack
a supplied schedule and remain unmatched. IDs are strings, never floating point.

Join equal channel IDs (`vsetv_id == channel_id`, officially confirmed), using
half-open intervals and `max(0,min(view_end,airing_end)-max(view_start,airing_start))`.
Resolve 13 schedule conflicts by latest start, lowest-source-row ties, retaining
unlabelled slots equally; aggregate fragments to the original airing. Coverage
divides by the **original full programme duration**, not a clipped fragment or
the view log's misspelled `duraton`. Exclude the one zero-duration schedule row.
The initial overlap join matched 5,408,598 views to airtime and 2,964,346 to
nonzero items; 2,857,641 had no valid item match. It produced 3,848,423 positive
interaction rows, 4,895 users and 4,363 items before the later session repair.

Corrected sessions deduplicate 17 identical intervals, union touching same-channel
intervals, and exclude 524,325,074 ambiguous user-seconds with concurrent different
channels. Gaps/switches split visits; separated visits cannot accumulate to 80%.
There are 5,283,479 continuous sessions and 6,912,291 schedule runs. Source data
also have 52 views >24h and 117 one-hour `duraton` discrepancies; no guessed clock
correction is imposed. Original counts remain historical source accounting.

| Role | History | Evaluation/availability | Available programmes |
|---|---|---|---:|
| Main holdout | [2020-03-09,2020-05-18) | [2020-05-18,2020-07-27) | 6,361 |
| Earlier holdout | [2020-03-09,2020-04-13) | [2020-04-13,2020-05-18) | 4,412 |
| Final refit | [2020-03-09,2020-07-27) | [2020-07-27,2020-10-19) | 6,632 |

Clip views at cutoffs before constructing features. Future schedules provide
availability/metadata only. No future user viewing is a predictor. Previously
watched programmes remain eligible for every method. Cold users and padding use
one common historical popularity ordering (10 main, 13 earlier, 2 final cold
users under the fixed >=60-second warm-history policy).

## 2. Corrected labels and metric

An airing qualifies when one uninterrupted unambiguous overlap is **>=80%** of
its original duration (`5*seconds >= 4*duration`). Count a user/airing once, then
select each user's five most frequent distinct programme IDs, with frequency
descending / numeric item ID ascending ties. Model-input qualification is
separate: retained components use >=50% distinct-airing frequency, no decay.

AP@5 is the sum of precision at first relevant hits divided by
`min(5,unique truth size)`; empty truth gives zero. All 1,260 users remain in MAP.
Main truth has 5,946 user-item pairs, 1,127 five-positive users, 107 with 1–4 and
26 empty. Earlier: 5,675 pairs, 1,044 five-positive, 165 with 1–4 and 51 empty.

The old >=60-second/all-watched target was too broad (median 160.5 truth items).
Holding the **same predictions and same original 1,246-user cohort** fixed:

| Historical method | Old MAP@5 | Corrected >=80%/frequency-Top5 MAP@5 |
|---|---:|---:|
| Popularity | .777787587 | .272959470 |
| Repeat | .797207063 | .335243446 |
| Repeat + popularity | .796182451 | .286231050 |

On all 1,260 users those unchanged old predictions score .269926587 /
.331518519 / .283050705 under corrected labels. This isolates target repair;
the newly selected standalone methods below are different fits/configurations.
See `artifacts/stage2/label_audit.json`; historical high MAP is not Kaggle MAP.

## 3. Fair standalone comparison and final versions

| Standalone method | Main MAP@5 | Earlier MAP@5 |
|---|---:|---:|
| Popularity, >=50% frequency | .295008598 | .290426587 |
| Repeat, >=50% frequency | .366007275 | .381887787 |
| Implicit ALS, 32 factors | .165965388 | .225822090 |
| Metadata TF-IDF CBF | .257625882 | .290187169 |

The earlier numbers are checked against the saved standalone artifact when
assembling the release. ALS is confidence-weighted implicit MF, alpha10,
regularization1, 12 iterations, seed42; CBF uses rich schedule attributes
(title/category/genres/year/director/actors), training-only TF-IDF vocabulary
up to40,000 features and frequency-weighted user profiles. They are independent
implemented advanced methods, not names for the popularity baseline.

| Version | Main local MAP@5 | Earlier local MAP@5 | Actual Kaggle PRIVATE | Actual public | Design change and measured rationale |
|---|---:|---:|---:|---:|---|
| V1 | .373631173 | .387935626 | .40693 | .42158 | RRF k10, Top200; repeat/ALS/CBF/pop weights1/.1/.1/.1 adds complementary generators to repeat |
| V2 | .380361111 | .393929674 | .41721 | .43381 | Add weight.5 channel-conditional popularity with sqrt user channel affinity; consistent gains in both windows |
| V3 | .384336420 | .400075176 | .42607 | .44042 | Early-only 53-feature histogram classifier + equal RRF with V2; reorder relevant candidates already retrieved |
| V4 | .391427028 | .402687169 | .42806 | .44323 | Add six channel completion/loyalty fields; stronger personal channel × completed-programme evidence |
| **V5** | **.395679233** | **.408857804** | **.43385** | **.44580** | Group-aware Top5 LambdaRank, same59 features/pool; better within-query ordering |

These are exact implementation differences and observed gains, not proof of a
single causal mechanism or claims about enjoyment. Local choices never read
Kaggle scores. V5 main/earlier gains over V4 are .004252/.006171; paired95% CIs
[.000601,.007983]/[.001846,.010358], unadjusted for adaptive comparisons.

## 4. Preserved negative results

- **V6:** stability .394757/.409093; calendar .395368/.409820; best drift
  .395642/.411555. Drift loses on main and main repeat-heavy users (−.005763).
  Earlier improvement alone does not justify replacing V5. No V6 CSV.
- **BPR:** LightFM1.17 failed its Python3.14 metadata build; no WARP result is
  claimed. Independent BPR log-frequency standalone .311584/.308569 beats this
  frozen ALS configuration, not V5. Same-pool integration .392906/.408882;
  expanded Top200 .394048/.408347. New relevant pairs220/134 outside the V5 pool
  never reach Top5. Audit finds correct feature/exposure code, sparse early
  positive exposure and latent scale shift. No V7 CSV.
- **Association Rules:** binary distinct-user baskets, minimum shared support3,
  non-self Top200 edges; confidence `co/(antecedent_support+10)` scores
  .268212/.282135; shrink cosine .202816/.210695. Confidence-selected Top100
  expansion plus five features scores .390187/.406643. Calibrated BPR on the
  unchanged expanded pool scores .394045/.407970. Both lose both windows;
  calibrated gains' CIs [−.004975,+.001951]/[−.004079,+.002274]. No V8 CSV.
  Confidence can primarily reflect popular news: “Губка Боб”→“ТСН. Неделя” has
  .856639 empirical confidence but only1.02480 lift, and misses the real cartoon
  viewer's future target. More candidate recall is not automatically better MAP.

All plans/results and previous versions are preserved; Stage7–9 explain the
bounded negative studies. The original full private diagnostics remain locally.

## 5. Final model, training and computational evidence

Pool: V2 Top200 union positive-score Top50 from repeat/ALS/CBF/popularity/channel,
mean222.05/219.29 pairs per main/earlier user, recall .87185/.88952. No truth is
injected. The first53 fields cover generator scores/ranks, personal completed/
soft history, recency, global popularity, schedule rates and metadata; six V4
fields add channel loyalty/completion. Exact59 names are in the V5 manifest.

LightGBM4.6.0 LambdaRank: binary gains[0,1], truncation5, query normalization,
80 rounds, lr.06, 15 leaves, min leaf50, maxbin63, L1=0/L2=1, no subsampling or
early stopping, deterministic forced-column histograms, CPU2, all seeds42.
Each contiguous group is one user × forecast task. No identity features or
classification positive-times10 weights. Rank by raw score; stable ties retain
V2 pool order. Cold users retain common popularity. LambdaRank optimizes an
NDCG-based pairwise surrogate; exact MAP@5 is computed independently.

Evaluated models learn only [Mar23,Apr6) and [Apr6,Apr13), with histories ending
Mar23/Apr6. There are2,520 queries,513,292 pairs,7,560 positives. Final refit adds
[Apr13,May18) and [May18,Jul27), each with cutoff-correct features:5,040 queries,
1,069,381 pairs,17,792 positives. Last label end27July precedes hidden target.
Final refit scores are not presented as independent validation on those labels.

Measured on macOS arm64/Python3.14: initial prepare54.34s (join8.94s), session
repair33.73s; fresh winning ALS benchmark5.006s/1.710GB process RSS; CBF fit/score
about1.15s. V5 selected early fit4.98s; final fit11.51s, refit/generation12.56s,
peak1.48GB; final scoring/ranking/CSV .129s. These cached-stage costs exclude
earlier ingestion/base fitting. Stage9 standalone12.24s/2.08GB, two integrations
51.70s/2.91GB. Pinned requirements and reproduction scope: `REPRODUCE.md`.

**Platform strategy:** select V5's repeat-friendly, multi-generator reranking:
best measured local and verified private performance at modest cost. Keep the
same simple popularity fallback. Do not market unobserved items as discoveries
or infer dislike from non-observation. No online test or revenue lift was run.

**Limitations:** sparse item-labelled airtime, missing schedule channels, uncertain
session/concurrent-device/frequency-tie semantics, naive clocks, local users with
fewer than five positives, five/ten-week versus12-week horizon shift, early-task
supervision scarcity, and retrieval misses. Holdouts were repeatedly reused;
bootstrap intervals do not correct selection optimism. V5 harms some users,
including earlier repeat-heavy mean−.004122 versus V4. No universal gain claim.

## 6. Two actual users — same methods and main holdout

<!-- REPRESENTATIVES -->

The two original representatives are retained across **all** listed methods.
They were chosen posthoc for contrasting historical activity, not randomly; these examples do not estimate average performance. IDs are pseudonymous
source identifiers. Only these two histories and the required join witness
are included publicly; full private records remain local.

History below is the complete >=50%-qualified programme-frequency profile
before 18 May, not all short channel visits. f50/f80 count distinct airings.
Last observation uses the source naive clock. Titles/categories are the
first chronological schedule entry for that ID; titles can vary by airing.

### U-heavy — user `13022140237449583302`

Selection: largest distinct validation truth history; 775 training / 1989 validation continuous runs. 62 >=50%-qualified historical programmes.

| Programme ID | Actual historical title | Category | f50 | f80 | Last observation |
|---|---|---|---:|---:|---|
| 2400480 | ТСН. Неделя. | Инфо | 60 | 45 | 2020-05-15 07:07:01 |
| 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | 53 | 31 | 2020-05-08 08:51:57 |
| 200344 | 'Чистоnews'. | Развлечения | 7 | 6 | 2020-05-14 20:45:00 |
| 10002515 | Т/с 'Мамочки', 1 и 2 с. | Сериалы | 7 | 5 | 2020-05-14 21:51:19 |
| 6500479 | 'Мир наизнанку'. | Познавательное | 6 | 5 | 2020-04-16 15:50:00 |
| 10002514 | Т/с 'Кухня'. | Сериалы | 6 | 3 | 2020-05-07 21:45:00 |
| 240081 | Факты. День. | Инфо | 4 | 1 | 2020-04-01 07:49:37 |
| 200337 | 'Утро с 'Интером'. | Развлечения | 3 | 1 | 2020-05-01 09:00:00 |
| 700369 | 'Ліга сміху'. | Развлечения | 3 | 0 | 2020-05-01 22:10:56 |
| 700475 | 'Голос страны', 10 сезон, 8 эп. 'Бои'. | Развлечения | 3 | 1 | 2020-04-05 22:15:00 |
| 1000589 | Т/с 'Отчаянные домохозяйки', 4 сезон, 17 с. | Сериалы | 3 | 1 | 2020-05-15 14:30:00 |
| 10002451 | Т/с 'Великолепный век', 39 и 40 с. | Сериалы | 3 | 1 | 2020-04-15 16:45:00 |
| 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 2 | 0 | 2020-05-14 09:18:48 |
| 5900256 | Утро с Украиной. | Инфо | 2 | 0 | 2020-03-26 07:58:16 |
| 6000483 | 'Свадьба вслепую', 6 сезон, 7 эп. | Развлечения | 2 | 2 | 2020-04-29 15:20:00 |
| 6700482 | Секретные материалы. | Другое | 2 | 1 | 2020-04-27 19:30:00 |
| 10003102 | Т/с 'Опер по вызову', 3 сезон, 9 и 10 с. | Сериалы | 2 | 1 | 2020-05-13 19:52:09 |
| 200477 | 'Завтрак. Выходной'. | Развлечения | 1 | 0 | 2020-05-02 10:00:00 |
| 400357 | 'Холостяк', 10 сезон, 1 эп., 1 ч. | Развлечения | 1 | 0 | 2020-04-10 22:00:00 |
| 700382 | 'Розсміши коміка'. | Развлечения | 1 | 0 | 2020-03-28 19:30:00 |
| 700389 | Кто сверху?, 6 сезон, 7 эп. | Развлечения | 1 | 1 | 2020-05-02 15:15:00 |
| 1000723 | Т/с 'Охотники за древностями', 3 сезон, 14 и 15 с. | Сериалы | 1 | 0 | 2020-04-10 01:04:45 |
| 2400467 | 'Викна-Новости'. | Инфо | 1 | 1 | 2020-04-16 14:47:57 |
| 2500430 | 'Затерянный мир'. | Познавательное | 1 | 0 | 2020-05-12 17:57:39 |
| 90016797 | Х/ф 'Криминальное чтиво'. | Фильмы | 1 | 0 | 2020-04-28 00:51:22 |
| 90028930 | Премьера. Х/ф 'Неспящие в Сиэтле'. | Фильмы | 1 | 0 | 2020-04-26 01:13:24 |
| 90031786 | Премьера. Х/ф 'Чужой билет'. | Фильмы | 1 | 0 | 2020-04-09 00:55:51 |
| 90048220 | Х/ф 'Плохие парни'. | Фильмы | 1 | 1 | 2020-05-02 23:00:00 |
| 90048585 | Х/ф 'Годзилла'. | Фильмы | 1 | 0 | 2020-05-09 20:30:51 |
| 90050385 | Х/ф 'Игры разума'. | Фильмы | 1 | 1 | 2020-04-04 12:06:31 |
| 90051178 | Премьера. Х/ф 'Планета обезьян'. | Фильмы | 1 | 0 | 2020-05-05 01:46:28 |
| 90054203 | Премьера. Х/ф 'Голливудские копы'. | Фильмы | 1 | 0 | 2020-05-03 01:15:07 |
| 90054589 | Х/ф 'Утраченное сокровище'. | Фильмы | 1 | 0 | 2020-03-27 09:15:00 |
| 90056569 | Х/ф 'Ван Хельсинг'. | Фильмы | 1 | 1 | 2020-04-17 22:15:00 |
| 90060136 | Х/ф 'Мистер и миссис Смит'. | Фильмы | 1 | 0 | 2020-03-13 22:58:56 |
| 90068619 | Х/ф 'Особо опасен'. | Фильмы | 1 | 0 | 2020-05-13 23:33:48 |
| 90070754 | Х/ф 'Форсаж 4'. | Фильмы | 1 | 1 | 2020-05-02 18:00:00 |
| 90070896 | Х/ф 'Остров проклятых'. | Фильмы | 1 | 1 | 2020-04-30 00:35:00 |
| 90071286 | Х/ф 'Охотник за головами'. | Фильмы | 1 | 1 | 2020-05-01 17:38:08 |
| 90072474 | Х/ф 'Орёл Девятого легиона'. | Фильмы | 1 | 1 | 2020-05-10 17:06:35 |
| 90073207 | Х/ф 'Три метра над уровнем неба'. | Фильмы | 1 | 0 | 2020-04-04 20:36:15 |
| 90074341 | Х/ф 'Белоснежка: Месть гномов'. | Фильмы | 1 | 0 | 2020-04-29 00:15:00 |
| 90074560 | Премьера. Х/ф 'Люди в чёрном 3'. | Фильмы | 1 | 0 | 2020-05-03 12:52:47 |
| 90074576 | Х/ф 'Призрачный гонщик 2'. | Фильмы | 1 | 0 | 2020-05-12 09:21:09 |
| 90077275 | Премьера. Х/ф 'Миллион способов потерять голову'. | Фильмы | 1 | 0 | 2020-05-11 00:25:00 |
| 90077765 | Х/ф 'Мечты сбываются!'. | Фильмы | 1 | 1 | 2020-05-03 10:58:46 |
| 90077897 | Х/ф 'Планета обезьян: Революция'. | Фильмы | 1 | 1 | 2020-05-07 00:50:00 |
| 90078669 | Х/ф 'Терминатор: Генезис'. | Фильмы | 1 | 1 | 2020-05-08 22:30:00 |
| 90078858 | Х/ф 'Мустанг'. | Фильмы | 1 | 1 | 2020-05-03 23:59:35 |
| 90079323 | Х/ф 'Иллюзия обмана 2'. | Фильмы | 1 | 0 | 2020-04-26 00:00:00 |
| 90079611 | Х/ф 'Сноуден'. | Фильмы | 1 | 0 | 2020-03-28 02:09:38 |
| 90079612 | Х/ф 'Зачинщики'. | Фильмы | 1 | 0 | 2020-05-01 01:21:45 |
| 90080135 | Х/ф 'Скрижали судьбы'. | Фильмы | 1 | 1 | 2020-04-11 19:00:00 |
| 90080685 | Х/ф 'Война за планету обезьян'. | Фильмы | 1 | 1 | 2020-05-09 00:55:00 |
| 90080832 | Х/ф 'Безбашенные'. | Фильмы | 1 | 0 | 2020-05-16 19:06:39 |
| 90080960 | Х/ф 'Очень плохие мамочки 2'. | Фильмы | 1 | 0 | 2020-05-03 00:15:11 |
| 90081097 | Х/ф 'Уцелевшая'. | Фильмы | 1 | 1 | 2020-05-17 10:21:49 |
| 90081440 | Х/ф 'Девочки мои', 1-4 с. | Фильмы | 1 | 0 | 2020-05-12 01:27:00 |
| 90081666 | Х/ф 'Верни мою жизнь', 1-3 с. | Фильмы | 1 | 1 | 2020-05-11 17:00:00 |
| 90082033 | Х/ф 'Никогда не бывает поздно', 1 и 2 с. | Фильмы | 1 | 0 | 2020-05-16 23:00:00 |
| 90082534 | Премьера. Х/ф 'Мой милый найденыш', 1 и 2 с. | Фильмы | 1 | 0 | 2020-03-27 17:00:00 |
| 90082663 | Премьера. Х/ф 'Время уходить, время возвращаться', 1 и 2 с. | Фильмы | 1 | 0 | 2020-05-17 23:00:00 |

Actual uninterrupted history interval examples (same profile):

| Programme ID | Start | End | Seconds / original duration | Coverage |
|---|---|---|---:|---:|
| 700475 | 2020-03-09 09:10:00 | 2020-03-09 10:55:47 | 6347 / 8400 | 75.5595% |
| 20088 | 2020-03-10 06:30:00 | 2020-03-10 06:48:02 | 1082 / 1800 | 60.1111% |
| 20088 | 2020-03-10 07:10:00 | 2020-03-10 07:48:57 | 2337 / 3000 | 77.9000% |

**Actual main-holdout relevant IDs:** `20088` 'Завтрак с 1+1'. Информационно-развлекательная программа. (Развлечения), `240081` Факты. День. (Инфо), `1000589` Т/с 'Отчаянные домохозяйки', 4 сезон, 17 с. (Сериалы), `2400480` ТСН. Неделя. (Инфо), `2400508` Сегодня. Итоги с Олегом Панютой. (Инфо).

Each recommendation cell below is **ID — title — category**; columns are ordered ranks.

| Method | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 | AP@5 |
|---|---|---|---|---|---|---:|
| Popularity | 2400480 — ТСН. Неделя. — Инфо | 240081 — Факты. День. — Инфо | 12001682 — М/с 'Губка Боб Квадратные Штаны'. — Детям | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 2400508 — Сегодня. Итоги с Олегом Панютой. — Инфо | 0.710000 (71/100) |
| Repeat | 2400480 — ТСН. Неделя. — Инфо | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 200344 — 'Чистоnews'. — Развлечения | 10002515 — Т/с 'Мамочки', 1 и 2 с. — Сериалы | 6500479 — 'Мир наизнанку'. — Познавательное | 0.400000 (2/5) |
| ALS | 6000483 — 'Свадьба вслепую', 6 сезон, 7 эп. — Развлечения | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 10002514 — Т/с 'Кухня'. — Сериалы | 6500479 — 'Мир наизнанку'. — Познавательное | 200477 — 'Завтрак. Выходной'. — Развлечения | 0.100000 (1/10) |
| CBF | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 2400480 — ТСН. Неделя. — Инфо | 2400467 — 'Викна-Новости'. — Инфо | 5900319 — Абзац! — Инфо | 200337 — 'Утро с 'Интером'. — Развлечения | 0.400000 (2/5) |
| V1 | 2400480 — ТСН. Неделя. — Инфо | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 200344 — 'Чистоnews'. — Развлечения | 10002515 — Т/с 'Мамочки', 1 и 2 с. — Сериалы | 6500479 — 'Мир наизнанку'. — Познавательное | 0.400000 (2/5) |
| V2 | 2400480 — ТСН. Неделя. — Инфо | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 200344 — 'Чистоnews'. — Развлечения | 240081 — Факты. День. — Инфо | 6500479 — 'Мир наизнанку'. — Познавательное | 0.550000 (11/20) |
| V3 | 2400480 — ТСН. Неделя. — Инфо | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 200344 — 'Чистоnews'. — Развлечения | 240081 — Факты. День. — Инфо | 10002515 — Т/с 'Мамочки', 1 и 2 с. — Сериалы | 0.550000 (11/20) |
| V4 | 2400480 — ТСН. Неделя. — Инфо | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 200344 — 'Чистоnews'. — Развлечения | 240081 — Факты. День. — Инфо | 6500479 — 'Мир наизнанку'. — Познавательное | 0.550000 (11/20) |
| V5 | 2400480 — ТСН. Неделя. — Инфо | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 200344 — 'Чистоnews'. — Развлечения | 240081 — Факты. День. — Инфо | 10002515 — Т/с 'Мамочки', 1 и 2 с. — Сериалы | 0.550000 (11/20) |
| BPR standalone | 2400480 — ТСН. Неделя. — Инфо | 240081 — Факты. День. — Инфо | 2400467 — 'Викна-Новости'. — Инфо | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 2400508 — Сегодня. Итоги с Олегом Панютой. — Инфо | 0.710000 (71/100) |
| Association standalone | 2400480 — ТСН. Неделя. — Инфо | 240081 — Факты. День. — Инфо | 2400467 — 'Викна-Новости'. — Инфо | 2400508 — Сегодня. Итоги с Олегом Панютой. — Инфо | 200344 — 'Чистоnews'. — Развлечения | 0.550000 (11/20) |
| Association integration | 2400480 — ТСН. Неделя. — Инфо | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 200344 — 'Чистоnews'. — Развлечения | 240081 — Факты. День. — Инфо | 10002515 — Т/с 'Мамочки', 1 и 2 с. — Сериалы | 0.550000 (11/20) |
| Calibrated BPR | 2400480 — ТСН. Неделя. — Инфо | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 200344 — 'Чистоnews'. — Развлечения | 6500479 — 'Мир наизнанку'. — Познавательное | 10002515 — Т/с 'Мамочки', 1 и 2 с. — Сериалы | 0.400000 (2/5) |

**Interpretation:** news/breakfast are strong repeated preferences. Popularity
gets four target IDs (.71); repeat/CBF get two (.40), ALS only breakfast at
rank2 (.10). V5 gets news/breakfast/facts at1,2,4 (.55), yet retains two
historical entertainment/series misses. It underperforms popularity here:
global preference can be a strong baseline, and average V5 gains are not universal.

### U-light — user `278032391686083827`

Selection: smallest distinct validation truth history; 64 training / 5 validation continuous runs. 13 >=50%-qualified historical programmes.

| Programme ID | Actual historical title | Category | f50 | f80 | Last observation |
|---|---|---|---:|---:|---|
| 12001682 | М/с 'Губка Боб Квадратные Штаны'. | Детям | 16 | 13 | 2020-05-11 16:20:00 |
| 12002201 | М/с 'Маша и Медведь', 46 с. | Детям | 5 | 4 | 2020-04-27 14:07:00 |
| 12002955 | М/с 'Свинка Пеппа', 1 с. 'Лужи'. | Детям | 4 | 4 | 2020-03-20 12:35:00 |
| 12002856 | М/с 'Элвиннн!!! и бурундуки'. | Детям | 3 | 3 | 2020-05-11 15:15:00 |
| 2400480 | ТСН. Неделя. | Инфо | 2 | 0 | 2020-04-09 20:35:00 |
| 12001734 | М/с 'Волшебные покровители', 161 с. 'Женат на мамочке'. | Детям | 2 | 2 | 2020-03-31 10:55:00 |
| 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | 1 | 1 | 2020-03-26 09:25:00 |
| 200344 | 'Чистоnews'. | Развлечения | 1 | 1 | 2020-04-09 20:45:00 |
| 200469 | Варьяты. Лучшее. | Развлечения | 1 | 0 | 2020-03-28 11:00:00 |
| 2400467 | 'Викна-Новости'. | Инфо | 1 | 0 | 2020-03-10 22:45:00 |
| 12002172 | М/с 'Гуппи и пузырики', 3 сезон, 18 с. 'Забег лягушек'. | Детям | 1 | 0 | 2020-03-20 12:15:00 |
| 12002552 | М/с 'Макс и Руби', 7 сезон, 16 с. 'Довольный малыш Хаффингтон / Кексики Руби'. | Детям | 1 | 0 | 2020-03-20 12:53:53 |
| 90043769 | Х/ф 'Иван Васильевич меняет профессию'. | Фильмы | 1 | 1 | 2020-04-09 22:30:00 |

Actual uninterrupted history interval examples (same profile):

| Programme ID | Start | End | Seconds / original duration | Coverage |
|---|---|---|---:|---:|
| 2400467 | 2020-03-10 22:18:28 | 2020-03-10 22:45:00 | 1592 / 2700 | 58.9630% |
| 12001682 | 2020-03-16 23:30:00 | 2020-03-16 23:40:00 | 600 / 600 | 100.0000% |
| 12001682 | 2020-03-18 19:16:05 | 2020-03-18 19:30:00 | 835 / 1500 | 55.6667% |

**Actual main-holdout relevant IDs:** `12002039` UA: Культура для самых маленьких. М/с 'Монстры и пираты'. (Детям).

Each recommendation cell below is **ID — title — category**; columns are ordered ranks.

| Method | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 | AP@5 |
|---|---|---|---|---|---|---:|
| Popularity | 2400480 — ТСН. Неделя. — Инфо | 240081 — Факты. День. — Инфо | 12001682 — М/с 'Губка Боб Квадратные Штаны'. — Детям | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 2400508 — Сегодня. Итоги с Олегом Панютой. — Инфо | 0.000000 (0) |
| Repeat | 12001682 — М/с 'Губка Боб Квадратные Штаны'. — Детям | 12002201 — М/с 'Маша и Медведь', 46 с. — Детям | 12002955 — М/с 'Свинка Пеппа', 1 с. 'Лужи'. — Детям | 12002856 — М/с 'Элвиннн!!! и бурундуки'. — Детям | 2400480 — ТСН. Неделя. — Инфо | 0.000000 (0) |
| ALS | 12001682 — М/с 'Губка Боб Квадратные Штаны'. — Детям | 12002896 — М/с 'Вспыш и чудо-машинки', 1 сезон, 15 с. 'Авто-Рейнджеры'. — Детям | 12002955 — М/с 'Свинка Пеппа', 1 с. 'Лужи'. — Детям | 2400480 — ТСН. Неделя. — Инфо | 12002856 — М/с 'Элвиннн!!! и бурундуки'. — Детям | 0.000000 (0) |
| CBF | 12001682 — М/с 'Губка Боб Квадратные Штаны'. — Детям | 120079564 — М/ф 'Губка Боб в 3D'. — Детям | 12002201 — М/с 'Маша и Медведь', 46 с. — Детям | 12002955 — М/с 'Свинка Пеппа', 1 с. 'Лужи'. — Детям | 12002856 — М/с 'Элвиннн!!! и бурундуки'. — Детям | 0.000000 (0) |
| V1 | 12001682 — М/с 'Губка Боб Квадратные Штаны'. — Детям | 12002201 — М/с 'Маша и Медведь', 46 с. — Детям | 12002955 — М/с 'Свинка Пеппа', 1 с. 'Лужи'. — Детям | 12002856 — М/с 'Элвиннн!!! и бурундуки'. — Детям | 2400480 — ТСН. Неделя. — Инфо | 0.000000 (0) |
| V2 | 12001682 — М/с 'Губка Боб Квадратные Штаны'. — Детям | 12002955 — М/с 'Свинка Пеппа', 1 с. 'Лужи'. — Детям | 12002201 — М/с 'Маша и Медведь', 46 с. — Детям | 2400480 — ТСН. Неделя. — Инфо | 12002856 — М/с 'Элвиннн!!! и бурундуки'. — Детям | 0.000000 (0) |
| V3 | 12001682 — М/с 'Губка Боб Квадратные Штаны'. — Детям | 12002955 — М/с 'Свинка Пеппа', 1 с. 'Лужи'. — Детям | 2400480 — ТСН. Неделя. — Инфо | 12002856 — М/с 'Элвиннн!!! и бурундуки'. — Детям | 12002201 — М/с 'Маша и Медведь', 46 с. — Детям | 0.000000 (0) |
| V4 | 12001682 — М/с 'Губка Боб Квадратные Штаны'. — Детям | 12002955 — М/с 'Свинка Пеппа', 1 с. 'Лужи'. — Детям | 2400480 — ТСН. Неделя. — Инфо | 12002856 — М/с 'Элвиннн!!! и бурундуки'. — Детям | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 0.000000 (0) |
| V5 | 12001682 — М/с 'Губка Боб Квадратные Штаны'. — Детям | 2400480 — ТСН. Неделя. — Инфо | 12002955 — М/с 'Свинка Пеппа', 1 с. 'Лужи'. — Детям | 12001734 — М/с 'Волшебные покровители', 161 с. 'Женат на мамочке'. — Детям | 20088 — 'Завтрак с 1+1'. Информационно-развлекательная программа. — Развлечения | 0.000000 (0) |
| BPR standalone | 2400480 — ТСН. Неделя. — Инфо | 2400467 — 'Викна-Новости'. — Инфо | 240081 — Факты. День. — Инфо | 12001682 — М/с 'Губка Боб Квадратные Штаны'. — Детям | 2400508 — Сегодня. Итоги с Олегом Панютой. — Инфо | 0.000000 (0) |
| Association standalone | 240081 — Факты. День. — Инфо | 2400480 — ТСН. Неделя. — Инфо | 2400467 — 'Викна-Новости'. — Инфо | 2400508 — Сегодня. Итоги с Олегом Панютой. — Инфо | 200344 — 'Чистоnews'. — Развлечения | 0.000000 (0) |
| Association integration | 12001682 — М/с 'Губка Боб Квадратные Штаны'. — Детям | 12002955 — М/с 'Свинка Пеппа', 1 с. 'Лужи'. — Детям | 2400480 — ТСН. Неделя. — Инфо | 12002856 — М/с 'Элвиннн!!! и бурундуки'. — Детям | 2400467 — 'Викна-Новости'. — Инфо | 0.000000 (0) |
| Calibrated BPR | 12001682 — М/с 'Губка Боб Квадратные Штаны'. — Детям | 12002955 — М/с 'Свинка Пеппа', 1 с. 'Лужи'. — Детям | 12002856 — М/с 'Элвиннн!!! и бурундуки'. — Детям | 2400480 — ТСН. Неделя. — Инфо | 12001734 — М/с 'Волшебные покровители', 161 с. 'Женат на мамочке'. — Детям | 0.000000 (0) |

**Interpretation:** fewer qualifying programmes and only one future Top5 label
make this a sparse-target counterexample. All required methods miss that
actual target (AP0); familiar cartoons/news do not establish relevance to the
future frequency-selected label. This is not a cold user, nor proof that
recommended programmes were disliked. More retrieval diversity does not
by itself solve this case.

Machine-readable two-user evidence and exact fractional arithmetic:
`evidence/representatives.json`. These are **holdout** recommendations from
the early-only model; the refitted hidden-period CSV is a separate artifact.
AP for the hidden-period CSV cannot be reconstructed from unavailable labels.

## 7. References, attribution and required final actions

- Official competition Data and Overview: <https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data>,
  <https://www.kaggle.com/competitions/sweettv-tv-program-recommender/overview/description>.
  Saved quotes/retrieval dates: `artifacts/sources/relevance_sources.json`.
- VseTV channel directory: <http://www.vsetv.com/channels.html>; saved offline
  snapshot and211 normalized ID/title matches. Official Kaggle text confirms IDs.
- Nikita Melentev, “Two-layered Recommender System Methodology: A Prize-Winning
  Solution” (2021-03-20): <https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156>.
  Inspired separate input/target thresholds, repeat/CF candidates and two-stage
  ranking. Participant code was inspected for corroboration, not copied; its
  historical scores are not ours. Official inclusive >=80% takes precedence.
- Hu, Koren, Volinsky, “Collaborative Filtering for Implicit Feedback Datasets”
  (2008), <https://doi.org/10.1109/ICDM.2008.22>; independently implemented ALS.
- LightGBM objective/truncation/API: <https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective>,
  <https://lightgbm.readthedocs.io/en/stable/Parameters.html#lambdarank_truncation_level>.
  Native library supplies LambdaRank; scikit-learn supplies TF-IDF/classifiers.
- Rendle et al., BPR (2009): <https://arxiv.org/abs/1205.2618>;
  <https://making.lyst.com/lightfm/docs/> (failed install, no WARP experiment).
- Association terminology: <https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/>;
  independent SciPy implementation. No pretrained weights/external solution code.

Implementation, numerical verification and evidence assembly were AI-assisted
and locally executed by the OpenCode coding assistant. The student has **not**
yet claimed the two exercises in `STUDENT_CHECKS.md`. The original unmodified
native session export must be supplied separately; this evidence is no substitute.
Finalization passed all70 regression tests, exact native V5 reload/ranking,
original-template/availability checks and preservation of334 prior files. Local
secret-pattern scanning found no matches; dependency consistency passed. Full
commands, scope limits and pending manual work are in `FINAL_CHECKS.md`.
Manually publish the release, insert its URL and the actual Kaggle screenshot
(show both score columns) into the PDF, perform/document student checks, and
submit the PDF plus original session export before the deadline. No PDF,
publication or additional Kaggle submission is created by this finalization.
