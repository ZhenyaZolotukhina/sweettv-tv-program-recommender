"""Assemble publication evidence and a whitelist-based release; never fit models.

Maintainer command: python3 -m tvrec.finalize_project
Requires the original local research evidence; not a clean-clone entry point.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .release_checks import ap_details, digest, V5_SHA256


ROOT = Path('.')


def load(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def stamp(seconds):
    return datetime.fromtimestamp(int(seconds), timezone.utc).replace(tzinfo=None).isoformat(' ')


def clean(text):
    return str(text).replace('|', '\\|').replace('\n', ' ')


METHODS = {
    'Popularity': 'artifacts/stage2/private/main_popularity_0.csv',
    'Repeat': 'artifacts/stage2/private/main_repeat_1.csv',
    'ALS': 'artifacts/stage2/private/main_cf_0.csv',
    'CBF': 'artifacts/stage2/private/main_cbf_0.csv',
    'V1': 'artifacts/stage3/private/main_v1.csv',
    'V2': 'artifacts/stage3/private/main_v1_plus_channel_0.5.csv',
    'V3': 'artifacts/stage4/private/main_v2_supervised_rrf_equal.csv',
    'V4': 'artifacts/stage5/private/main_loyalty.csv',
    'V5': 'artifacts/stage6/private/main_lambda_top5.csv',
    'BPR standalone': 'artifacts/stage8/private/main_bpr_log_frequency.csv',
    'Association standalone': 'artifacts/stage9/private/main_association_confidence.csv',
    'Association integration': 'artifacts/stage9/private/main_association.csv',
    'Calibrated BPR': 'artifacts/stage9/private/main_bpr_calibrated.csv',
}


def representatives():
    import duckdb
    original = load('artifacts/stage2/private/representative_users.json')
    summaries = load('artifacts/stage2/representative_user_summary.json')
    examples = load('artifacts/stage9/private/original_two_method_comparisons.json')
    records = []
    c = duckdb.connect('cache/data.duckdb', read_only=True)
    for n, example in enumerate(examples):
        user = example['user_id']
        old = next(r for r in original if r['user_id'] == user)
        record = {'alias': summaries[n]['alias'], 'user_id': user,
                  'selection_reason': old['original_selection_reason'],
                  'selection_is_posthoc': True,
                  'history_cutoff': '2020-05-18 00:00:00', 'history': example['history'],
                  'training_run_count': len(old['training_continuous_runs']),
                  'validation_run_count': len(old['validation_continuous_runs']),
                  'truth': example['truth'], 'methods': {}}
        record['actual_history_run_examples'] = [
            r for r in old['training_continuous_runs'] if r['fraction'] >= .5][:3]
        for name, path in METHODS.items():
            with Path(path).open(newline='') as f:
                row = next(r for r in csv.DictReader(f) if r['user_id'] == user)
            pred = list(map(int, row['predictions'].split()))
            truth = set(int(i) for i in row['truth'].split())
            assert truth == {p['item'] for p in record['truth']}
            details = ap_details(truth, pred)
            assert abs(details['ap5']-float(row['ap5'])) < 1e-12
            top = []
            for item in pred:
                title, category = c.execute('SELECT title,category FROM airings WHERE item=? ORDER BY ss,sid LIMIT 1', [item]).fetchone()
                top.append({'item': item, 'title': title, 'category': category})
            record['methods'][name] = {'top5': top, **details, 'source_predictions': path}
        records.append(record)
    c.close()
    save('evidence/representatives.json', records)
    witness = {'source': load('artifacts/stage2/private/source_line_checks.json'),
               'witness': load('artifacts/stage2/private/coverage_witness.json')}
    save('evidence/interval_witness.json', witness)
    return records


def report_examples(records):
    lines = ['The two original representatives are retained across **all** listed methods.',
             'They were chosen posthoc for contrasting historical activity, not randomly;'
             ' these examples do not estimate average performance. IDs are pseudonymous',
             'source identifiers. Only these two histories and the required join witness',
             'are included publicly; full private records remain local.', '',
             'History below is the complete >=50%-qualified programme-frequency profile',
             'before 18 May, not all short channel visits. f50/f80 count distinct airings.',
             'Last observation uses the source naive clock. Titles/categories are the',
             'first chronological schedule entry for that ID; titles can vary by airing.', '']
    for record in records:
        lines += [f"### {record['alias']} — user `{record['user_id']}`", '',
                  f"Selection: {record['selection_reason']}; {record['training_run_count']} training / "
                  f"{record['validation_run_count']} validation continuous runs. "
                  f"{len(record['history'])} >=50%-qualified historical programmes.", '',
                  '| Programme ID | Actual historical title | Category | f50 | f80 | Last observation |',
                  '|---|---|---|---:|---:|---|']
        for p in record['history']:
            lines.append(f"| {p['item']} | {clean(p['title'])} | {clean(p['category'])} | "
                         f"{p['frequency50']} | {p['frequency80']} | {stamp(p['last_second'])} |")
        lines += ['', 'Actual uninterrupted history interval examples (same profile):', '',
                  '| Programme ID | Start | End | Seconds / original duration | Coverage |',
                  '|---|---|---|---:|---:|']
        for r in record['actual_history_run_examples']:
            lines.append(f"| {r['item']} | {stamp(r['a'])} | {stamp(r['b'])} | "
                         f"{r['overlap_seconds']} / {r['duration']} | {100*r['fraction']:.4f}% |")
        lines += ['', '**Actual main-holdout relevant IDs:** ' + ', '.join(
            f"`{p['item']}` {clean(p['title'])} ({clean(p['category'])})" for p in record['truth']) + '.', '',
            'Each recommendation cell below is **ID — title — category**; columns are ordered ranks.', '',
            '| Method | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 | AP@5 |',
            '|---|---|---|---|---|---|---:|']
        for name, method in record['methods'].items():
            cells = [f"{p['item']} — {clean(p['title'])} — {clean(p['category'])}" for p in method['top5']]
            lines.append('| ' + name + ' | ' + ' | '.join(cells) + f" | {method['ap5']:.6f} ({method['fraction']}) |")
        lines += ['']
        if record['alias'] == 'U-heavy':
            lines += ['**Interpretation:** news/breakfast are strong repeated preferences. Popularity',
                      'gets four target IDs (.71); repeat/CBF get two (.40), ALS only breakfast at',
                      'rank2 (.10). V5 gets news/breakfast/facts at1,2,4 (.55), yet retains two',
                      'historical entertainment/series misses. It underperforms popularity here:',
                      'global preference can be a strong baseline, and average V5 gains are not universal.', '']
        else:
            lines += ['**Interpretation:** fewer qualifying programmes and only one future Top5 label',
                      'make this a sparse-target counterexample. All required methods miss that',
                      'actual target (AP0); familiar cartoons/news do not establish relevance to the',
                      'future frequency-selected label. This is not a cold user, nor proof that',
                      'recommended programmes were disliked. More retrieval diversity does not',
                      'by itself solve this case.', '']
    lines += ['Machine-readable two-user evidence and exact fractional arithmetic:',
              '`evidence/representatives.json`. These are **holdout** recommendations from',
              'the early-only model; the refitted hidden-period CSV is a separate artifact.',
              'AP for the hidden-period CSV cannot be reconstructed from unavailable labels.']
    path = Path('REPORT_EVIDENCE.md')
    text = path.read_text()
    start = text.index('<!-- REPRESENTATIVES -->')
    end = text.index('\n## 7.', start)
    text = text[:start] + '<!-- REPRESENTATIVES -->\n\n' + '\n'.join(lines) + '\n' + text[end:]
    path.write_text(text)


def has_private_records(obj):
    if isinstance(obj, dict):
        if any(k in obj for k in ('user_id', 'template_users', 'user_ids')):
            return True
        return any(has_private_records(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_private_records(v) for v in obj)
    return isinstance(obj, str) and obj.isdigit() and len(obj) >= 16


def build_package():
    target = Path('github_ready')
    if target.exists() and not (target/'RELEASE_MANIFEST.json').exists():
        raise RuntimeError('Refuse to use a pre-existing unowned publication directory')
    target.mkdir(exist_ok=True)
    # Regression imports create bytecode inside this owned staging directory.
    # Strip generated caches from the delivered filesystem, not just Git's index.
    for generated in target.rglob('__pycache__'):
        shutil.rmtree(generated)
    stale_validation = target/'artifacts/release/v5_validation.json'
    stale_validation.unlink(missing_ok=True)
    generated_logs = target/'artifacts/release'
    if generated_logs.exists() and not any(generated_logs.iterdir()):
        generated_logs.rmdir()
    paths = [Path('run_tests.py'), Path('requirements.txt'), Path('requirements-ltr.txt'),
             Path('requirements-lock.txt'), Path('REPRODUCE.md'), Path('REPORT_EVIDENCE.md'),
             Path('STUDENT_CHECKS.md'), Path('FINAL_CHECKS.md'), Path('FINAL.md'), Path('AUDIT.md')]
    paths += sorted(Path('.').glob('STAGE*.md'))
    paths += sorted(Path('tvrec').glob('*.py')) + sorted(Path('tests').glob('*.py'))
    paths += sorted(Path('evidence').glob('*.json'))
    paths += [Path('artifacts/dataset_accounting.json'), Path('artifacts/sources/relevance_sources.json'),
              Path('artifacts/sources/vsetv_channels.txt'),
              Path('artifacts/monitoring/submission_history_grading_correction.json')]
    # Keep aggregate historical results/plans/configs; never copy full private data.
    for p in sorted(Path('artifacts').glob('stage*/*')):
        if not p.is_file() or p.suffix not in ('.json', '.csv'):
            continue
        if p.suffix == '.json' and has_private_records(load(p)):
            continue
        if p.suffix == '.csv':
            with p.open(newline='') as f:
                header = next(csv.reader(f), [])
            if 'user_id' in header:
                continue
        paths.append(p)
    for n in range(1, 6):
        paths.append(Path(f'artifacts/final/submission_hybrid_v{n}.csv'))
        manifest = Path(f'artifacts/final/submission_hybrid_v{n}.manifest.json')
        if manifest.exists():
            paths.append(manifest)
    paths.append(Path('artifacts/final/manifest.json'))  # Original V1 manifest.
    relative_paths = {p.as_posix() for p in paths}
    # Remove only files listed in our own previous manifest that are no longer selected.
    previous = load(target/'RELEASE_MANIFEST.json')['files'] if (target/'RELEASE_MANIFEST.json').exists() else {}
    for name in previous:
        if name not in relative_paths and name not in ('README.md', '.gitignore'):
            (target/name).unlink(missing_ok=True)
    for p in sorted(set(paths)):
        destination = target/p
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
    (target/'README.md').write_text('''# SWEET.TV midterm — final V5 release

**Selected model: V5 LambdaRank. Verified Kaggle Private MAP@5: 0.43385.**

- [REPORT_EVIDENCE.md](REPORT_EVIDENCE.md): consolidated report facts, comparisons,
  negative results and two actual users.
- [STUDENT_CHECKS.md](STUDENT_CHECKS.md): two numerical exercises to perform personally.
- [REPRODUCE.md](REPRODUCE.md): pinned setup, verification and fixed-model replay.
- [FINAL_CHECKS.md](FINAL_CHECKS.md): executed regression/security checks and manual actions.
- Final CSV: `artifacts/final/submission_hybrid_v5.csv` (exact submitted bytes).
- SHA-256: `249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f`.

Quick offline verification (no third-party packages):

```sh
python3 -m tvrec.release_checks --package .
```

All V1–V5 files and aggregate negative studies are preserved. Historical stage
reports/manifests reflect their creation-time submission status; the later
verified history correction supersedes it. Current grading is best PRIVATE.

Deadline: **24 October 2026, 19:00 Moscow (16:00 UTC)**. Student checks,
manual GitHub publication, score screenshot, PDF and original unmodified native
AI-session export remain personal course-submission actions. No publication,
PDF or new Kaggle submission was made by this release process.

Code and checks are AI-assisted; see AUDIT.md. Required two-user/witness evidence
and exact prediction files contain pseudonymous source IDs. Raw Kaggle data,
caches, credentials, full private histories and session exports are excluded.
No license for redistribution of Kaggle raw data is implied.
''')
    (target/'.gitignore').write_text('''sweettv-tv-program-recommender/
cache/
.venv/
__pycache__/
*.pyc
.env
.env.*
kaggle.json
*.pem
*.key
artifacts/private/
artifacts/**/private/
artifacts/reproduction/
artifacts/release/
.DS_Store
''')
    assert digest(target/'artifacts/final/submission_hybrid_v5.csv') == V5_SHA256
    files = {p.relative_to(target).as_posix(): digest(p) for p in sorted(target.rglob('*'))
             if p.is_file() and p.name != 'RELEASE_MANIFEST.json' and '__pycache__' not in p.parts and
             not p.relative_to(target).as_posix().startswith('artifacts/release/')}
    save(target/'RELEASE_MANIFEST.json', {
        'selected': 'V5 LambdaRank', 'private_map5_user_verified': .43385,
        'final_csv_sha256': V5_SHA256, 'deadline_moscow': '2026-10-24T19:00:00+03:00',
        'files': files, 'excluded': ['raw Kaggle files', 'credentials', 'caches',
                                   'full private user records', 'native session export'],
        'source_workspace_untouched': 'Previous models/artifacts remain in the research workspace',
        'verification_executor': 'AI coding assistant; student exercises not yet performed'})
    print(json.dumps({'package': str(target), 'files': len(files),
                      'bytes': sum(p.stat().st_size for p in target.rglob('*') if p.is_file()),
                      'v5_sha256': V5_SHA256}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-workspace', action='store_true')
    args = parser.parse_args()
    if args.verify_workspace:
        from .final_refit import check_evidence
        from .group_rank import lightgbm_module
        from .reproduce_v5 import base_scores, fit
        from .ltr_experiment import matrix
        from importlib.metadata import version
        frozen = load('artifacts/stage9/frozen_prior.json')['files']
        check_evidence(frozen)
        manifest = load('artifacts/final/submission_hybrid_v5.manifest.json')
        for name in ['roll_mar23', 'roll_apr06', 'earlier', 'main', 'final']:
            X, meta, z, names = matrix(name)
            assert len(names) == 59 and names == manifest['feature_names']
            assert len(X) == len(z['user_index']) and sum(meta['pool_lengths']) == len(X)
            del X
        versions = {}
        for line in Path('requirements-lock.txt').read_text().splitlines():
            if not line or line.startswith('#'):
                continue
            package, pinned = line.split('==')
            actual = lightgbm_module().__version__ if package == 'lightgbm' else version(package)
            assert actual == pinned, f'Version mismatch: {package}'
            versions[package] = actual
        regression = load('artifacts/release/regression.json')
        save('evidence/final_regression.json', regression)
        save('evidence/v5_final_validation.json', load('artifacts/release/v5_validation.json'))
        result = {'valid': True, 'prior_files_hash_verified': len(frozen),
                  'all_five_v1_v5_csvs_unchanged': True,
                  'fixed_replay_imports_and_five_cached_feature_roles_checked': True,
                  'fixed_replay_end_to_end_executed': False,
                  'no_new_real_data_training_or_optimization': True,
                  'regression_tests': regression['tests_run'], 'regression_passed': regression['passed'],
                  'runtime_versions': versions,
                  'student_exercises_status': 'pending personal execution',
                  'remaining_manual_actions': ['Publish github_ready contents and record URL',
                      'Capture actual Kaggle V5 result screenshot', 'Perform and document student checks',
                      'Create report PDF with repository link and screenshot',
                      'Export original unmodified native OpenCode session and submit with PDF'],
                  'blocked_deliverables': ['Repository URL and screenshot not supplied',
                      'PDF and native export intentionally left to manual submission'],
                  'kaggle_requests': 0, 'github_publication': False, 'pdf_created': False,
                  'executor': 'AI coding assistant; student verification not claimed'}
        save('evidence/finalization.json', result)
        print(json.dumps(result, indent=2))
        return
    records = representatives()
    report_examples(records)
    build_package()


if __name__ == '__main__':
    main()
