"""Independent V5 CSV/group/temporal/model-reload checks; never touches V1--V4."""
import csv
import json
from pathlib import Path

import duckdb
import numpy as np

from .core import ap_at_5
from .final_refit import digest,check_evidence
from .group_rank import lightgbm_module
from .pipeline import save
from .verify_final import validate_file_records


def main(persist=True):
    out=Path('artifacts/stage6');mp=Path('artifacts/final/submission_hybrid_v5.manifest.json')
    m=json.loads(mp.read_text());check_evidence(json.loads((out/'frozen_prior.json').read_text())['files'])
    assert all(r['target_end']<=1595808000 and r['cutoff']<r['target_end'] for r in m['group_records'])
    assert m['reported_holdout_model']['last_label_end']==1586736000
    assert not m['reported_holdout_model']['main_or_earlier_holdout_labels_used']
    assert digest(m['reported_holdout_model']['path'])==m['reported_holdout_model']['sha256']
    assert digest(m['final_model']['path'])==m['final_model']['sha256']
    assert m['training_queries']==5040 and m['last_final_label_end']==1595808000
    with Path('sweettv-tv-program-recommender/submission.csv').open(newline='') as f:
        reader=csv.reader(f);assert next(reader)==['user_id','tv_show_id'];users=[r[0] for r in reader]
    c=duckdb.connect('cache/data.duckdb',read_only=True)
    candidates={i for i, in c.execute('''SELECT DISTINCT item FROM airings WHERE period=31 AND item>0 AND duration>0
        AND ss<1603065600 AND se>1595808000''').fetchall()}
    assert c.execute('SELECT max(ve) FROM views').fetchone()[0]<1595808000
    with Path(m['submission_path']).open(newline='') as f:
        reader=csv.reader(f);header=next(reader);rows=list(reader)
    validate_file_records(users,header,rows,candidates)
    assert len(users)==1260 and digest(m['submission_path'])==m['submission_sha256']
    meta=json.loads(Path('cache/rerank/final/features_manifest.json').read_text())
    extra=json.loads(Path('cache/focused_v4/final/extras_loyalty.json').read_text())
    assert meta['signature']['cutoff']==extra['cutoff']==1595808000
    assert extra['last_used_view_second']<1595808000 and not extra['future_user_views_used']
    assert meta['users']==users and set(meta['items'])==candidates
    assert meta['feature_names']+extra['columns']['loyalty']==m['feature_names']
    assert digest('cache/rerank/final/X.npy')==m['final_base_features_sha256']
    assert digest('cache/focused_v4/final/loyalty.npy')==m['final_loyalty_features_sha256']
    X=np.column_stack([np.load('cache/rerank/final/X.npy'),np.load('cache/focused_v4/final/loyalty.npy')])
    lgb=lightgbm_module();model=lgb.Booster(model_file=m['final_model']['path'])
    assert model.dump_model()['objective']=='lambdarank' and model.current_iteration()==m['rounds']
    score=model.predict(X);np.testing.assert_array_equal(score,np.load(m['final_scores_path']))
    with np.load('cache/rerank/final/indices.npz') as z:
        idx=z['item_index'];cold=z['cold'];fallback=z['fallback']
    old=None
    if m['selected']=='lambda_v4_rrf':old=np.load('cache/focused_v4/final/loyalty_final_probabilities.npy')
    offsets=np.r_[0,np.cumsum(meta['pool_lengths'])]
    for n,u in enumerate(users):
        lo,hi=offsets[n:n+2];position=np.arange(hi-lo)
        order=np.lexsort((position,-score[lo:hi]))
        if old is not None:
            a=np.lexsort((position,-old[lo:hi]));r=np.empty(len(a),int);r[a]=np.arange(1,len(a)+1)
            v4=1/(10+position+1)+1/(10+r);a=np.lexsort((position,-v4))
            r4=np.empty(len(a),int);r4[a]=np.arange(1,len(a)+1)
            rl=np.empty(len(a),int);rl[order]=np.arange(1,len(a)+1)
            order=np.lexsort((position,-(1/(10+r4)+1/(10+rl))))
        chosen=idx[lo:hi][order[:5]] if not cold[n] else fallback[:5]
        assert [meta['items'][i] for i in chosen]==list(map(int,rows[n][1].split())),f'Ranking mismatch row{n+1}'
    checked=0
    for window,lower,upper in [('main',1589760000,1595808000),('earlier',1586736000,1589760000)]:
        with Path(f'artifacts/stage5/private/{window}_loyalty.csv').open(newline='') as f:
            original={r['user_id']:r for r in csv.DictReader(f)}
        sql='SELECT DISTINCT item FROM airings WHERE item>0 AND duration>0 AND ss<? AND se>?'
        if window=='main':sql+=' AND period=21'
        available={i for i, in c.execute(sql,[upper,lower]).fetchall()}
        for file in sorted((out/'private').glob(window+'_*.csv')):
            with file.open(newline='') as f:records=list(csv.DictReader(f))
            validate_file_records(users,['user_id','tv_show_id'],[[r['user_id'],r['predictions']] for r in records],available)
            for r in records:
                before=original[r['user_id']];truth=set(map(int,r['truth'].split()));pred=list(map(int,r['predictions'].split()))
                assert truth==set(map(int,before['truth'].split())) and r['group']==before['group']
                assert abs(ap_at_5(truth,pred)-float(r['ap5']))<1e-12
                assert [n for n,i in enumerate(pred,1) if i in truth]==list(map(int,r['hit_ranks'].split()))
                if r['group']=='cold':assert r['predictions']==before['predictions']
            checked+=1
    # Independent group arithmetic for each training fold, without generation's validator.
    for record in m['group_records']:
        folder=Path('cache/rerank')/record['name'];fm=json.loads((folder/'features_manifest.json').read_text())
        assert fm['signature']['cutoff']==record['cutoff']
        with np.load(folder/'indices.npz') as z:
            user_index=z['user_index'];item_index=z['item_index']
        starts=np.r_[0,np.cumsum(fm['pool_lengths'])]
        assert starts[-1]==record['rows'] and len(fm['pool_lengths'])==record['queries']
        for n,(lo,hi) in enumerate(zip(starts,starts[1:])):
            assert np.all(user_index[lo:hi]==n) and len(np.unique(item_index[lo:hi]))==hi-lo
    check_evidence(json.loads((out/'frozen_prior.json').read_text())['files'])
    result={'valid':True,'rows':len(users),'candidates':len(candidates),'sha256':m['submission_sha256'],
        'original_string_ids_and_order':True,'five_distinct_positive_available_ids':True,
        'user_task_group_boundaries_verified':True,'no_holdout_labels_in_evaluated_model':True,
        'no_hidden_period_labels_in_final_model':True,'independent_model_reload_and_all_rankings':True,
        'offline_files_recomputed':checked,'prior_v1_v4_and_evidence_unchanged':True,'kaggle_requests':0}
    if persist:
        save(out/'v5_independent_validation.json',result)
    m['independent_validation']=result
    if (out/'environment.json').exists():m['execution_environment']=json.loads((out/'environment.json').read_text())
    m['input_file_hashes']=json.loads(Path('artifacts/dataset_accounting.json').read_text())['source_files']
    m['base_requirements_sha256']=digest('requirements.txt')
    if (out/'test_results.json').exists():m['tests']=json.loads((out/'test_results.json').read_text())
    m['verifier_sha256']=digest(__file__)
    if persist:
        save(mp,m)
    c.close();print(result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--read-only',action='store_true',help='Never rewrite historical validation/manifest artifacts')
    parser.add_argument('--out',type=Path,help='Optional separate current verification record')
    args=parser.parse_args()
    result=main(persist=not args.read_only)
    if args.out:
        save(args.out,result)
