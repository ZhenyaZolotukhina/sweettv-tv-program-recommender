# Final submission readiness — 23 October 2026

**Selected V5 LambdaRank; user-verified Kaggle PRIVATE MAP@5 .43385.**
No further real-data model fits, searches or optimization were run at finalization.

## Executed by the AI coding assistant

| Check | Genuine result |
|---|---|
| Regression from publication-package source | **70 tests passed**, 0 failures/errors, .458s; includes real native LambdaRank synthetic regression |
| Final V5 model reload | All1,260 rankings and saved score arrays reproduced exactly; six historical holdout prediction files checked |
| Submission format | Header `user_id,tv_show_id`;1,260 original users/order; five distinct positive programme IDs per row |
| Test availability | Every prediction belongs to the6,632 available programmes, checked against original schedule |
| V5 CSV SHA-256 | `249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f` |
| Preservation | All334 Stage9-protected prior files hash-identical; all V1–V5 CSVs retain verified hashes |
| Two numerical exercises | Exact original source-line comparison; join3000/3300=90.90909%; AP11/20=.55 |
| Runtime pins | All13 recorded direct/transitive package versions match `requirements-lock.txt` |
| Dependency consistency | `pip check`: **No broken requirements found** |
| Publication/security | Manifest hashes, Python syntax, path/extension/symlink/size policy and local secret patterns checked; **zero secret-pattern findings** |

The security check is a local publication/credential-pattern scan, **not** an
online vulnerability-database audit or a guarantee about every possible secret.
Raw Kaggle files, credentials, caches, binaries, full private user records,
session exports and PDFs are excluded. Only the required two pseudonymous
profiles, one join witness and IDs in exact prediction CSVs are included.

Records: `evidence/final_regression.json`, `evidence/v5_final_validation.json`,
`evidence/finalization.json`; the package inventory is `RELEASE_MANIFEST.json`.
The research workspace additionally retains current final scan output at
`artifacts/release/package_check.json` and all original detailed evidence.

Commands actually executed from the research workspace unless noted:

```sh
# From github_ready/, using the already-installed isolated LightGBM package:
PYTHONPATH=/Users/zoleo/midterm/cache/ltr_dependencies python3 run_tests.py --out ../artifacts/release/regression.json
# From the research workspace; read-only mode preserves old artifact bytes:
python3 -m tvrec.verify_v5 --read-only --out artifacts/release/v5_validation.json
python3 -m tvrec.finalize_project --verify-workspace
PYTHONPATH=/Users/zoleo/midterm/cache/ltr_dependencies python3 -m pip check
python3 -m tvrec.release_checks --package github_ready --raw-data sweettv-tv-program-recommender --out artifacts/release/package_check.json
```

Clean-clone checks are independently executable from the package root:

```sh
python3 -m tvrec.release_checks --package .
python3 -m tvrec.release_checks --exercise join
python3 -m tvrec.release_checks --exercise ap
```

## Explicit limits and remaining manual actions

- The fixed fresh-data replay entry point is supplied and import/syntax/cache-role
  compatibility checked, but **not executed end-to-end during finalization**.
  Existing original execution and exact independent model reload support the
  reported results. No new training was needed to verify the final CSV.
- No fresh-environment install was attempted; installed package versions and
  dependency consistency were verified. Native-library portability remains in
  `REPRODUCE.md`.
- Repository URL and actual Kaggle screenshot are not yet supplied. No account
  request, submission, automatic publication or PDF creation occurred.
- **Student exercises remain pending personal execution.** Do not claim that
  the assistant's calculations were student-performed.
- Publish the contents of `github_ready/`, record the URL, capture the existing
  V5 result screenshot (both public/private columns), and prepare the PDF with
  the link, screenshot and evidence. Export the original unmodified native
  OpenCode session separately; never edit/reconstruct it.
- Submit the **PDF plus original native session export** by **24 October 2026,
  19:00 Moscow (16:00 UTC)**. No further Kaggle attempt is required by this work.
