"""Independent full-file and cached-score verification of hybrid v1.

No generation validator is imported. No Kaggle network request is made.
python -m tvrec.verify_final
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import duckdb
import numpy as np
from scipy import sparse


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def validate_file_records(template,header,rows,candidates):
    if header!=['user_id','tv_show_id']:
        raise ValueError('Header/column mismatch')
    if len(rows)!=len(template):
        raise ValueError('Row count mismatch')
    for n,(expected,row) in enumerate(zip(template,rows),1):
        if len(row)!=2 or row[0]!=expected or not isinstance(row[0],str):
            raise ValueError(f'User order/precision or column mismatch at row {n}')
        value=row[1]
        if not re.fullmatch(r'[1-9][0-9]*( [1-9][0-9]*){4}',value,flags=re.ASCII):
            raise ValueError(f'Not exactly five canonical positive IDs at row {n}')
        items=[int(t) for t in value.split(' ')]
        if len(set(items))!=5 or not set(items)<=candidates:
            raise ValueError(f'Duplicate or unavailable IDs at row {n}')


def independently_fuse_row(component_rows,fallback,config):
    """Independent lexicographic ranking of cached scores, same frozen formula."""
    fallback=np.asarray(fallback,dtype=int)
    fallback_rank=np.empty(len(fallback),dtype=int)
    fallback_rank[fallback]=np.arange(len(fallback))
    total=np.zeros(len(fallback),dtype=np.float32)
    for name,weight in config['weights'].items():
        values=component_rows[name]
        order=np.lexsort((fallback_rank,-values))[:config['top']]
        positive=order[values[order]>0]
        contribution=np.zeros(len(fallback),dtype=np.float32)
        contribution[positive]=1/(config['k']+np.arange(1,len(positive)+1))
        total+=weight*contribution
    order=np.lexsort((fallback_rank,-total))
    supported=order[total[order]>0].tolist()
    seen=set(supported)
    return total,(supported+[int(i) for i in fallback if i not in seen])[:5]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('sweettv-tv-program-recommender'))
    parser.add_argument('--cache',type=Path,default=Path('cache'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/final'))
    args=parser.parse_args()
    out=args.out
    manifest=json.loads((out/'manifest.json').read_text())
    with (args.data/'submission.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.reader(f)
        assert next(reader)==['user_id','tv_show_id']
        template=[row[0] for row in reader]
    assert len(template)==len(set(template))==1260
    assert file_sha(args.data/'submission.csv')==manifest['input_files']['submission.csv']['sha256']
    c=duckdb.connect(str(args.cache/'data.duckdb'),read_only=True)
    c.execute("SET threads=4; SET memory_limit='512MB'")
    # Deliberately query the original cached schedule, not generation's candidate list.
    start=1595808000
    end=1603065600
    assert manifest['availability']['window']==[start,end]
    candidates={int(i) for i, in c.execute('''SELECT DISTINCT item FROM airings WHERE period=31
        AND item IS NOT NULL AND item>0 AND duration>0 AND ss<? AND se>?''',[end,start]).fetchall()}
    csv_path=Path(manifest['submission']['path'])
    with csv_path.open(newline='',encoding='utf-8') as f:
        reader=csv.reader(f)
        header=next(reader)
        rows=list(reader)
    validate_file_records(template,header,rows,candidates)
    actual_sha=file_sha(csv_path)
    assert actual_sha==manifest['submission']['sha256']
    archive=Path(manifest['submission']['archive_path'])
    assert actual_sha==file_sha(archive)==manifest['submission']['archive_sha256']
    frozen=json.loads((out/'frozen_evidence_manifest.json').read_text())['files']
    for name,info in frozen.items():
        assert file_sha(name)==info['sha256'],f'Previous evidence changed: {name}'
    assert file_sha('artifacts/stage2/selected_strategy.json')==manifest['strategy_sha256']
    model_dir=args.cache/'final/hybrid_v1'
    mappings=json.loads((model_dir/'mappings.json').read_text())
    assert mappings['template_users']==template
    items=mappings['test_candidates']
    assert set(items)==candidates
    assert all(isinstance(u,str) for u in mappings['model_users'])
    assert len(set(mappings['model_users']))==len(mappings['model_users'])
    assert len(set(mappings['model_items']))==len(mappings['model_items'])
    matrix=sparse.load_npz(model_dir/'training_matrix.npz')
    factors=np.load(model_dir/'als_factors.npz')
    assert matrix.shape==(len(mappings['model_users']),len(mappings['model_items']))
    assert matrix.nnz==manifest['matrix_nonzero']
    assert factors['user_factors'].shape==(matrix.shape[0],32)
    assert factors['item_factors'].shape==(matrix.shape[1],32)
    assert np.all(matrix.data>0) and np.all(matrix.data==np.floor(matrix.data))
    assert np.all(factors['item_factors'][matrix.getnnz(axis=0)==0]==0)
    assert np.all(factors['user_factors'][matrix.getnnz(axis=1)==0]==0)
    # Materialize each compressed component once, rather than decompressing the
    # full matrix again for every user row.
    with np.load(model_dir/'component_scores.npz') as loaded:
        scores={name:loaded[name] for name in loaded.files}
    fallback=scores['fallback']
    expected_shape=(len(template),len(items))
    assert all(scores[k].shape==expected_shape for k in ['repeat','cf','cbf','popularity','hybrid'])
    # Independently re-read the clean history to apply the SAME fixed cold grouping.
    sessions=duckdb.connect(str(args.cache/'stage2.duckdb'),read_only=True)
    # Attach the original schedule instead of relying on stored views/aliases.
    sessions.execute(f"ATTACH '{str((args.cache/'data.duckdb').resolve())}' AS original (READ_ONLY)")
    warm={u for u, in sessions.execute('''SELECT DISTINCT e.user_id FROM continuous_events e
        JOIN original.airings a USING(sid) WHERE a.item>0 AND a.duration>0
        AND least(e.b,1595808000)-greatest(e.a,1583712000)>=60''').fetchall()}
    cold=set(template)-warm
    assert len(cold)==manifest['availability']['cold_template_users']
    for n,user in enumerate(template):
        row_scores={name:scores[name][n] for name in manifest['strategy']['selected_config']['weights']}
        fused,order=independently_fuse_row(row_scores,fallback,manifest['strategy']['selected_config'])
        assert np.array_equal(fused,scores['hybrid'][n]),f'RRF arithmetic mismatch at row {n+1}'
        if user in cold:
            order=fallback[:5]
        assert [int(items[i]) for i in order]==list(map(int,rows[n][1].split(' '))),f'Hybrid ranking mismatch at row {n+1}'
    checks={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'valid':True,'rows':len(rows),
        'columns':header,'original_user_order_and_lossless_ids':True,'five_distinct_positive_available_ids':True,
        'no_missing_zero_nan_extra_columns':True,'test_candidates':len(candidates),'sha256':actual_sha,
        'archive_identical':True,'frozen_earlier_evidence_unchanged':True,
        'new_factor_mapping_and_shapes_verified':True,'sparse_training_counts_positive_integers':True,
        'independent_all_row_rrf_reconstruction':True,'same_cold_fallback_verified':True,
        'independent_of_generation_validator':True,'kaggle_requests_made':0,
        'verifier_source_sha256':file_sha(__file__)}
    (out/'independent_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    manifest['submission']['independent_validation']=checks
    tests=out/'test_results.json'
    if tests.exists():
        manifest['regression_tests']=json.loads(tests.read_text())
        assert manifest['regression_tests']['passed']
    manifest['final_refit_performed']=True
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print({k:v for k,v in checks.items() if k not in ['checked_at_utc']})
    sessions.close()
    c.close()


if __name__=='__main__':
    main()
