# Reproduction — selected V5 only

Final choice is frozen: **V5 LambdaRank, Kaggle Private MAP@5 0.43385**.
Deadline: **24 October 2026, 19:00 Europe/Moscow (16:00 UTC)**.

## Verify the delivered release (no data, training or network required)

From this repository root, with Python 3.14:

```sh
python3 -m tvrec.release_checks --package .
python3 -m tvrec.release_checks --exercise join
python3 -m tvrec.release_checks --exercise ap
shasum -a 256 artifacts/final/submission_hybrid_v5.csv
```

Expected CSV SHA-256:
`249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f`.
The package check verifies every manifest entry, CSV format, all five version
hashes, the numerical exercises, pinned direct dependencies and the publication
file policy. It does not claim to reproduce model scores without source data.

## Install the recorded environment and run regression tests

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ltr.txt
.venv/bin/python run_tests.py --out artifacts/release/regression.json
```

Recorded environment: macOS arm64, Python 3.14.0, LightGBM 4.6.0. Direct
dependencies are version-pinned; `requirements-lock.txt` also pins the recorded
transitive runtime dependencies. Use it instead of `requirements-ltr.txt` for
that full package set. A compatible native OpenMP runtime may be required by
LightGBM on other systems. Platform-specific wheel/native-library differences
can change bitwise results. Historical measured costs are in REPORT_EVIDENCE.md.

## Replay from licensed original data in a separate, fresh directory

This optional fixed-configuration driver was added at finalization and was
**not executed end-to-end during finalization**, because further model runs were
stopped. Its imports, syntax and frozen-cache compatibility are checked; the
archived original V5 execution/independent reconstruction remain the evidence
for reported scores. Historical exploration entry points remain in source for
audit; the commands below do not run their parameter searches.

Obtain the five original CSVs under your own Kaggle access. Keep them outside
the publication package. Choose a new empty replay directory, copy `tvrec/` into
it, and run there using the installed environment. Supply the original filenames
under `sweettv-tv-program-recommender/` (a symlink to a private data directory is
acceptable). Do not copy archived `artifacts/` or `cache/` into the replay directory.
Copy the packaged `artifacts/sources/vsetv_channels.txt` snapshot to a local path
and pass its absolute filename below:

```sh
python3 -m tvrec.pipeline prepare --channel-source /absolute/path/vsetv_channels.txt
python3 -m tvrec.reproduce_v5
shasum -a 256 artifacts/reproduction/submission_hybrid_v5.csv
```

`pipeline prepare` ingests and joins only; it does not run the old 240-setting
selection. `reproduce_v5` reconstructs continuous sessions, the fixed ALS/CBF/
repeat/popularity/channel generators at each cutoff, the fixed candidate union
and 59 features. It fits only the selected Top-5 LambdaRank early/final models.
Outputs go to `artifacts/reproduction/`; archived final CSVs are never replaced.
Inspect `replay.json` against main/earlier MAP .3956792328042328 /
.40885780423280427. A fresh replay is computational work, not a verification
already performed by the student or a new experiment requested now.

Full raw filenames, counts and SHA-256 values are in
`artifacts/dataset_accounting.json`. The original view source has no hidden-period
views. Preserve naive wall-clock times; do not silently apply timezone conversion.
Final labels end 27 July 2020; target availability is 27 July–19 October.

## Historical records and publication

V1–V5 CSVs, manifests, stage reports and aggregate experiment records are retained.
Historical files say “not submitted” at creation time; the later correction in
`artifacts/monitoring/submission_history_grading_correction.json` takes precedence:
all five were submitted and best PRIVATE is graded, per the user's verified
instructor clarification. Scores were not independently fetched by the assistant.

The only public user evidence is the required two pseudonymous representative
histories plus one interval witness, and the IDs inherently present in the exact
submission CSVs. Full private logs, raw Kaggle data, caches, credentials and the
native session export are excluded. Publish the **contents of `github_ready/`**,
not the research workspace. No repository has been published automatically.
