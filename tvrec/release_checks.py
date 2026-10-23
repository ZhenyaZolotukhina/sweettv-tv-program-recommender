"""Independent standard-library release, security and hand-exercise checks.

No model fits, cache access, network requests or credential discovery.
"""
import argparse
import ast
import csv
from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
from itertools import islice
import json
from pathlib import Path
import re


V5_SHA256 = '249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ap_details(truth, predictions):
    truth = set(truth)
    seen, hits, total = set(), [], Fraction(0)
    for rank, item in enumerate(predictions[:5], 1):
        if item in truth and item not in seen:
            precision = Fraction(len(hits)+1, rank)
            total += precision
            hits.append({'rank': rank, 'item': item, 'precision': str(precision)})
        seen.add(item)
    value = total/min(5, len(truth)) if truth else Fraction(0)
    return {'hits': hits, 'denominator': min(5, len(truth)),
            'fraction': str(value), 'ap5': float(value)}


def exercise(root, which, raw=None):
    if which == 'join':
        witness = json.loads((root/'evidence/interval_witness.json').read_text())
        source = witness['source']
        view = next(csv.reader([source['view_row']]))
        airing = next(csv.reader([source['schedule_row']]))
        assert view[1] == airing[0] == str(witness['witness']['channel'])
        a, b = (datetime.fromisoformat(x) for x in view[2:4])
        start = datetime.strptime(airing[2], '%d.%m.%Y %H:%M:%S')
        duration = int(airing[3])
        end = start + timedelta(seconds=duration)
        cutoff = datetime.fromisoformat(source['validation_start'])
        lo, hi = max(a, start, cutoff), min(b, end)
        seconds = max(0, int((hi-lo).total_seconds()))
        assert seconds == 3000 and duration == 3300
        assert seconds == witness['witness']['overlap_seconds']
        coverage = Fraction(seconds, duration)
        assert coverage == Fraction(10, 11) and 5*seconds >= 4*duration
        if raw:
            for filename, line, expected in [
                ('dataset11-30.csv', source['physical_line'], source['view_row']),
                ('export_arh_11-20-final.csv', source['schedule_physical_line'], source['schedule_row'])]:
                with (raw/filename).open(encoding='utf-8', newline='') as f:
                    actual = next(islice(f, line-1, line), '').rstrip('\r\n')
                assert actual == expected, f'Original source line differs: {filename}:{line}'
        return {'exercise': 'join', 'overlap_start': lo.isoformat(), 'overlap_end': hi.isoformat(),
                'overlap_seconds': seconds, 'original_duration_seconds': duration,
                'coverage_fraction': str(coverage), 'coverage_percent': float(100*coverage),
                'qualifies_ge80': True, 'raw_source_lines_checked': raw is not None}
    records = json.loads((root/'evidence/representatives.json').read_text())
    user = records[0]
    truth = [p['item'] for p in user['truth']]
    pred = [p['item'] for p in user['methods']['V5']['top5']]
    result = ap_details(truth, pred)
    assert result['fraction'] == '11/20' and [h['rank'] for h in result['hits']] == [1, 2, 4]
    assert abs(result['ap5']-user['methods']['V5']['ap5']) < 1e-12
    return {'exercise': 'ap', 'user_id': user['user_id'], 'truth': truth,
            'recommendations': pred, **result}


def check_csv(path, raw=None):
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['user_id', 'tv_show_id']
        rows = list(reader)
    assert len(rows) == 1260
    users = [r['user_id'] for r in rows]
    assert len(set(users)) == 1260 and all(re.fullmatch(r'[0-9]+', u) for u in users)
    for row in rows:
        assert set(row) == {'user_id', 'tv_show_id'}
        ids = row['tv_show_id'].split(' ')
        assert len(ids) == len(set(ids)) == 5
        assert all(re.fullmatch(r'[1-9][0-9]*', i) for i in ids)
    sha = digest(path)
    assert sha == V5_SHA256
    if raw:
        with (raw/'submission.csv').open(encoding='utf-8', newline='') as f:
            assert users == [r['user_id'] for r in csv.DictReader(f)]
        lower, upper = datetime(2020, 7, 27), datetime(2020, 10, 19)
        available = set()
        with (raw/'export_arh_31-42-final.csv').open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                start = datetime.strptime(row['start_time'], '%d.%m.%Y %H:%M:%S')
                duration = int(row['duration'])
                item = row['tv_show_id']
                if item and int(item) > 0 and duration > 0 and start < upper and start+timedelta(seconds=duration) > lower:
                    available.add(item)
        assert len(available) == 6632
        assert all(set(r['tv_show_id'].split()) <= available for r in rows)
    return {'rows': len(rows), 'sha256': sha, 'five_unique_positive_ids': True,
            'original_template_order_and_availability_checked': raw is not None}


def package_check(root, raw=None):
    manifest = json.loads((root/'RELEASE_MANIFEST.json').read_text())
    expected = manifest['files']
    # Runtime-generated regression/security outputs and bytecode are ignored.
    files = [p for p in root.rglob('*') if p.is_file() and
             '__pycache__' not in p.parts and '.git' not in p.parts and '.venv' not in p.parts and
             not p.relative_to(root).as_posix().startswith('artifacts/release/')]
    actual = {p.relative_to(root).as_posix() for p in files}
    assert actual == set(expected) | {'RELEASE_MANIFEST.json'}, 'Unexpected/missing release files'
    for name, sha in expected.items():
        assert digest(root/name) == sha, f'Changed release file: {name}'
    history = json.loads((root/'artifacts/monitoring/submission_history_grading_correction.json').read_text())
    for row in history['history']:
        path = root/'artifacts/final'/('submission_hybrid_'+row['version'].lower()+'.csv')
        assert digest(path) == row['sha256'], f'Historical submission changed: {path.name}'
    forbidden = {'cache', 'private', 'sweettv-tv-program-recommender', '.opencode'}
    secret_names = {'.env', 'kaggle.json', '.DS_Store', 'id_rsa', 'id_ed25519'}
    secret_patterns = [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                       r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{40,}',
                       r'AKIA[0-9A-Z]{16}', r'sk-(?:proj-)?[A-Za-z0-9_-]{40,}',
                       r'(?i)["\x27](?:api_key|access_token|password|secret_key)["\x27]\s*:\s*["\x27][A-Za-z0-9_+/=-]{16,}["\x27]']
    findings = []
    parsed = 0
    for path in files:
        relative = path.relative_to(root)
        assert not (set(relative.parts) & forbidden) and path.name not in secret_names
        assert path.suffix not in {'.pkl', '.npz', '.npy', '.duckdb', '.pdf', '.zip', '.pptx'}
        assert not path.is_symlink() and path.stat().st_size < 10_000_000
        text = path.read_text(encoding='utf-8')
        if path.suffix == '.py':
            ast.parse(text, filename=str(relative))
            parsed += 1
        for pattern in secret_patterns:
            if re.search(pattern, text):
                findings.append({'file': str(relative), 'rule': secret_patterns.index(pattern)})
    assert not findings, f'Potential secret matches (values not printed): {findings}'
    for filename in ['requirements.txt', 'requirements-ltr.txt', 'requirements-lock.txt']:
        for line in (root/filename).read_text().splitlines():
            assert not line or line.startswith(('#', '-r ')) or re.fullmatch(r'[A-Za-z0-9_-]+==[0-9][A-Za-z0-9.]*', line)
    users = json.loads((root/'evidence/representatives.json').read_text())
    assert len(users) == 2 and len({r['user_id'] for r in users}) == 2
    for user in users:
        truth = [p['item'] for p in user['truth']]
        for method in user['methods'].values():
            pred = [p['item'] for p in method['top5']]
            assert len(pred) == len(set(pred)) == 5
            assert abs(ap_details(truth, pred)['ap5']-method['ap5']) < 1e-12
    result = {'valid': True, 'files_hashed': len(expected), 'python_files_parsed': parsed,
              'secret_pattern_findings': findings, 'excluded_file_policy_passed': True,
              'all_five_submission_hashes_match_verified_history': True,
              'csv': check_csv(root/'artifacts/final/submission_hybrid_v5.csv', raw),
              'join_exercise': exercise(root, 'join', raw), 'ap_exercise': exercise(root, 'ap'),
              'security_scope': 'Local path/extension/symlink/size and secret-pattern scan; not an online vulnerability audit',
              'executor': 'AI assistant; student verification not claimed'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--exercise', choices=['join', 'ap'])
    parser.add_argument('--raw-data', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = args.package or Path('.')
    if args.exercise:
        result = exercise(root, args.exercise, args.raw_data)
    else:
        result = package_check(root, args.raw_data)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
