# Student checks — perform personally before claiming verification

These are **two complete numerical exercises using actual data**. The assistant
has assembled and checked the evidence; **the student has not yet performed or
signed off these checks**. Run the commands yourself and record your actual
outputs/date. They require only Python's standard library. From `github_ready/`
(or the published repository root), run the commands shown below.

## A. Viewing-to-programme interval join

Actual view source: `dataset11-30.csv`, physical line **3,913,424** (header line1).

| Field | Real value |
|---|---|
| User ID | `7344208324753660769` |
| View channel (`vsetv_id`) | `805` |
| View start | `2020-05-17 06:39:08` |
| View end | `2020-05-18 02:09:32` |
| View duration (`duraton`, not programme duration) | 70,224 seconds |

Actual schedule source: `export_arh_11-20-final.csv`, physical line **656,406**.

| Field | Real value |
|---|---|
| Programme ID | `10002079` |
| Title | Т/с 'Спартак: Кровь и песок', 11 с. 'Старые раны'. |
| Category | Сериалы |
| Schedule channel (`channel_id`) | `805` (Sony Turbo) |
| Programme start | `2020-05-17 23:55:00` |
| Programme end | `2020-05-18 00:50:00` |
| Original programme duration | **3,300 seconds** (55 minutes) |

The main validation interval begins `2020-05-18 00:00:00`. Channel IDs match.
Before split clipping the view spans all 3,300 programme seconds (100%). For the
validation label, clip the overlap at the validation start:

```text
overlap_start = max(view_start, programme_start, validation_start)
              = 2020-05-18 00:00:00
overlap_end   = min(view_end, programme_end)
              = 2020-05-18 00:50:00
overlap       = max(0, overlap_end - overlap_start) = 3,000 seconds
coverage      = 3,000 / 3,300 = 10/11 = 90.909090909...%
>=80% check   = 5 × 3,000 >= 4 × 3,300
              = 15,000 >= 13,200: TRUE
```

Keep the original 3,300-second denominator; using the clipped 3,000 would wrongly
report 100% validation coverage. This qualifies the **airing**; programme-level
Top5 membership still depends on the user's total qualifying-airing frequency.
The saved independent endpoint check found the overlap uninterrupted and
unambiguous; the long original view is not truncated by an invented maximum.

Independent arithmetic from the embedded original CSV rows, without production
join/session code:

```sh
python3 -m tvrec.release_checks --exercise join
```

Optional verification against the locally licensed originals (not in GitHub):

```sh
python3 -m tvrec.release_checks --exercise join --raw-data /absolute/path/to/sweettv-tv-program-recommender
```

This compares both exact physical source lines, reconstructs timestamps and
computes the intersection. `evidence/interval_witness.json` holds the source
rows and prior independent continuity evidence; it is not a raw-data dump.

## B. Actual user's AP@5 for V5 on the main holdout

User **`13022140237449583302`** (U-heavy, the same user in REPORT_EVIDENCE.md).
Use the early-only V5 model's predictions for **18 May–27 July 2020**.
These are historical holdout predictions, not an invented hidden-period truth.

Actual relevant ID set, selected by >=80% coverage and frequency Top5:

`{20088, 240081, 1000589, 2400480, 2400508}`.

| Rank | Programme ID | Title | Category | Relevant? | Hits through rank | Precision if hit |
|---:|---|---|---|---|---:|---:|
| 1 | 2400480 | ТСН. Неделя. | Инфо | Yes | 1 | 1/1 = 1 |
| 2 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | Yes | 2 | 2/2 = 1 |
| 3 | 200344 | 'Чистоnews'. | Развлечения | No | 2 | — |
| 4 | 240081 | Факты. День. | Инфо | Yes | 3 | 3/4 = .75 |
| 5 | 10002515 | Т/с 'Мамочки', 1 и 2 с. | Сериалы | No | 3 | — |

Hits occur at **1, 2, 4**. There are five unique relevant IDs, so:

```text
AP@5 = (1/1 + 2/2 + 3/4) / min(5, 5)
     = (1 + 1 + .75) / 5
     = 11/20 = 0.55
```

Independent exact rational arithmetic, checking the delivered two-user evidence:

```sh
python3 -m tvrec.release_checks --exercise ap
```

Expected fields include `fraction: "11/20"`, `ap5: 0.55`, hit ranks1,2,4.
The arithmetic implementation is separate from the production AP evaluator.
The assistant checked these predictions against the saved main V5 prediction
file and the original source-backed target; the command needs no private cache.

## Personal completion record — currently pending

- Exercise A: **not yet performed by student**. Record date, command/output and
  explain why coverage uses 3,300 rather than 3,000 as denominator.
- Exercise B: **not yet performed by student**. Record date, output and explain
  why a non-hit at rank3 still affects rank4 precision.

Do not substitute an assistant-run test, this document, or a reconstructed
conversation for your personal checks or the original native session export.
