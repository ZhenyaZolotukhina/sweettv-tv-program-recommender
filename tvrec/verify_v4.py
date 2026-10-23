"""Independent V4 full CSV, temporal provenance and ranking reconstruction."""
import csv
import json
from pathlib import Path
import pickle

import duckdb
import numpy as np
from threadpoolctl import threadpool_limits

from .core import ap_at_5
from .final_refit import digest,check_evidence
from .pipeline import save
from .verify_final import validate_file_records


def main():
    out=Path('artifacts/stage5');path=Path('artifacts/final/submission_hybrid_v4.manifest.json')
    m=json.loads(path.read_text());block=m['selected_block']
    check_evidence(json.loads((out/'frozen_prior.json').read_text())['files'])
    assert m['offline_results']['experiments_run']<=3
    assert all(f['target_end']<=1586736000 for f in m['offline_results']['main']['training_folds'])
    assert all(f['target_end']<=1595808000 for f in m['final_training_folds'])
    assert m['last_training_label_end']==1595808000
    assert digest(m['reported_validation_model']['path'])==m['reported_validation_model']['sha256']
    assert digest(m['final_model']['path'])==m['final_model']['sha256']
    with Path('sweettv-tv-program-recommender/submission.csv').open(newline='') as f:
        reader=csv.reader(f);assert next(reader)==['user_id','tv_show_id'];users=[r[0] for r in reader]
    c=duckdb.connect('cache/data.duckdb',read_only=True)
    available={i for i, in c.execute('''SELECT DISTINCT item FROM airings WHERE period=31 AND item>0 AND duration>0
        AND ss<1603065600 AND se>1595808000''').fetchall()}
    assert c.execute('SELECT max(ve) FROM views').fetchone()[0]<1595808000
    with Path(m['submission_path']).open(newline='') as f:
        reader=csv.reader(f);header=next(reader);rows=list(reader)
    validate_file_records(users,header,rows,available)
    assert len(rows)==1260 and digest(m['submission_path'])==m['submission_sha256']
    base=Path('cache/rerank/final');extra=Path('cache/focused_v4/final')
    meta=json.loads((base/'features_manifest.json').read_text())
    info=json.loads((extra/('extras_'+block+'.json')).read_text())
    assert meta['users']==users and set(meta['items'])==available
    assert info['cutoff']==1595808000 and not info['future_user_views_used']
    assert info['last_used_view_second']<1595808000
    assert meta['feature_names']+info['columns'][block]==m['feature_names']
    assert digest(base/'X.npy')==m['final_base_feature_hash']
    assert digest(extra/(block+'.npy'))==m['final_extra_feature_hash']
    with Path(m['final_model']['path']).open('rb') as f:model=pickle.load(f)
    assert all(model.get_params()[k]==v for k,v in m['model_parameters'].items())
    X=np.column_stack([np.load(base/'X.npy'),np.load(extra/(block+'.npy'))])
    with threadpool_limits(limits=2):p=model.predict_proba(X)[:,1]
    np.testing.assert_array_equal(p,np.load(extra/(block+'_final_probabilities.npy')))
    with np.load(base/'indices.npz') as z:idx=z['item_index'];cold=z['cold'];fallback=z['fallback']
    starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
    for n,u in enumerate(users):
        lo,hi=starts[n:n+2];position=np.arange(hi-lo)
        learned=np.lexsort((position,-p[lo:hi]));rank=np.empty(len(learned),int);rank[learned]=np.arange(1,len(learned)+1)
        fused=1/(10+position+1)+1/(10+rank)
        picked=idx[lo:hi][np.lexsort((position,-fused))[:5]] if not cold[n] else fallback[:5]
        assert [meta['items'][i] for i in picked]==list(map(int,rows[n][1].split())),f'Ranking mismatch row {n+1}'
    checked=0
    for window,lower,upper in [('main',1589760000,1595808000),('earlier',1586736000,1589760000)]:
        with Path(f'artifacts/stage4/private/{window}_v2_supervised_rrf_equal.csv').open(newline='') as f:
            old={r['user_id']:r for r in csv.DictReader(f)}
        sql='SELECT DISTINCT item FROM airings WHERE item>0 AND duration>0 AND ss<? AND se>?'
        if window=='main':sql+=' AND period=21'
        candidates={i for i, in c.execute(sql,[upper,lower]).fetchall()}
        for fpath in sorted((out/'private').glob(window+'_*.csv')):
            if fpath.name.endswith('_v3_top5_errors.csv'):continue
            with fpath.open(newline='') as f:records=list(csv.DictReader(f))
            validate_file_records(users,['user_id','tv_show_id'],[[r['user_id'],r['predictions']] for r in records],candidates)
            for r in records:
                source=old[r['user_id']];truth=set(map(int,r['truth'].split()));pred=list(map(int,r['predictions'].split()))
                assert truth==set(map(int,source['truth'].split())) and r['group']==source['group']
                assert abs(ap_at_5(truth,pred)-float(r['ap5']))<1e-12
                assert [n for n,i in enumerate(pred,1) if i in truth]==list(map(int,r['hit_ranks'].split()))
                if r['group']=='cold':assert r['predictions']==source['predictions']
            checked+=1
    check_evidence(json.loads((out/'frozen_prior.json').read_text())['files'])
    result={'valid':True,'rows':len(rows),'candidates':len(available),'sha256':m['submission_sha256'],
        'original_ids_order_distinct_available_programmes':True,'independent_model_reload_and_all_row_rank_check':True,
        'temporal_training_roles_verified':True,'offline_ablation_files_checked':checked,
        'frozen_primary_truth_and_all_user_denominator':True,'cold_fallback_unchanged':True,
        'prior_v1_v2_v3_and_evidence_unchanged':True,'kaggle_requests':0}
    save(out/'v4_independent_validation.json',result)
    m['independent_validation']=result;m['verifier_source_sha256']=digest(__file__)
    if (out/'test_results.json').exists():m['tests']=json.loads((out/'test_results.json').read_text())
    save(path,m);c.close();print(result)


if __name__=='__main__':main()
