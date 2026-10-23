"""Independent V3 format, temporal provenance and learned ranking verification."""
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
    out=Path('artifacts/stage4')
    p=Path('artifacts/final/submission_hybrid_v3.manifest.json')
    m=json.loads(p.read_text())
    check_evidence(json.loads((out/'frozen_prior.json').read_text())['files'])
    earliest_holdout=1586736000;test_start=1595808000;test_end=1603065600
    audit=json.loads((out/'supervision_audit.json').read_text())
    assert all(f['cutoff']<f['target_end']<=earliest_holdout for f in audit['folds'])
    assert not audit['heldout_labels_used_in_fit']
    assert all(f['cutoff']<f['target_end']<=test_start for f in m['final_supervised_folds'])
    assert m['reported_holdout_model']['last_training_label_end']==earliest_holdout
    assert m['final_training_label_end']==m['test_features_cutoff']==test_start
    assert digest(m['final_model']['path'])==m['final_model']['sha256']
    assert digest(m['reported_holdout_model']['path'])==m['reported_holdout_model']['sha256']
    folder=Path('cache/rerank/final')
    fm=json.loads((folder/'features_manifest.json').read_text())
    assert fm['signature']['cutoff']==test_start and fm['signature']['upper']==test_end
    assert fm['feature_names']==m['feature_names']
    assert digest(folder/'X.npy')==m['final_features_sha256']
    assert digest(folder/'indices.npz')==m['final_indices_sha256']
    with Path('sweettv-tv-program-recommender/submission.csv').open(newline='') as f:
        reader=csv.reader(f);assert next(reader)==['user_id','tv_show_id'];users=[r[0] for r in reader]
    c=duckdb.connect('cache/data.duckdb',read_only=True)
    available={i for i, in c.execute('''SELECT DISTINCT item FROM airings WHERE period=31 AND item>0 AND duration>0
        AND ss<? AND se>?''',[test_end,test_start]).fetchall()}
    assert c.execute('SELECT max(ve) FROM views').fetchone()[0]<test_start
    with Path(m['submission_path']).open(newline='') as f:
        reader=csv.reader(f);header=next(reader);rows=list(reader)
    validate_file_records(users,header,rows,available)
    assert len(users)==1260 and digest(m['submission_path'])==m['submission_sha256']
    assert fm['users']==users and set(fm['items'])==available
    with Path(m['final_model']['path']).open('rb') as f:
        model=pickle.load(f)
    assert all(model.get_params()[k]==v for k,v in m['model_parameters'].items())
    with threadpool_limits(limits=2):
        probabilities=model.predict_proba(np.load(folder/'X.npy',mmap_mode='r'))[:,1]
    np.testing.assert_array_equal(probabilities,np.load(folder/'final_model_probabilities.npy'))
    with np.load(folder/'indices.npz') as z:
        idx=z['item_index'];cold=z['cold'];fallback=z['fallback']
    offsets=np.r_[0,np.cumsum(fm['pool_lengths'])]
    for n,user in enumerate(users):
        lo,hi=offsets[n:n+2]
        ids=idx[lo:hi]
        pos=np.arange(hi-lo)
        learned=np.lexsort((pos,-probabilities[lo:hi]))
        if m['strategy']=='supervised_only':
            ranked=ids[learned[:5]]
        else:
            inverse=np.empty(len(learned),dtype=int);inverse[learned]=np.arange(1,len(learned)+1)
            fused=1/(10+pos+1)+1/(10+inverse)
            ranked=ids[np.lexsort((pos,-fused))[:5]]
        if cold[n]:
            ranked=fallback[:5]
        assert [fm['items'][i] for i in ranked]==list(map(int,rows[n][1].split())),f'Ranking mismatch row {n+1}'
    checked=0
    for window in ['main','earlier']:
        with Path(f'artifacts/stage3/private/{window}_v1_plus_channel_0.5.csv').open(newline='') as f:
            reference={r['user_id']:r for r in csv.DictReader(f)}
        lower=1589760000 if window=='main' else earliest_holdout
        upper=test_start if window=='main' else 1589760000
        sql='SELECT DISTINCT item FROM airings WHERE item>0 AND duration>0 AND ss<? AND se>?'
        if window=='main':
            sql+=' AND period=21'
        candidates={i for i, in c.execute(sql,[upper,lower]).fetchall()}
        for file in sorted((out/'private').glob(window+'_*.csv')):
            with file.open(newline='') as f:
                values=list(csv.DictReader(f))
            assert [r['user_id'] for r in values]==users
            validate_file_records(users,['user_id','tv_show_id'],
                [[r['user_id'],r['predictions']] for r in values],candidates)
            for r in values:
                source=reference[r['user_id']]
                truth=set(map(int,r['truth'].split()));pred=list(map(int,r['predictions'].split()))
                assert truth==set(map(int,source['truth'].split())) and r['group']==source['group']
                assert abs(ap_at_5(truth,pred)-float(r['ap5']))<1e-12
                assert [rank for rank,item in enumerate(pred,1) if item in truth]==list(map(int,r['hit_ranks'].split()))
                if r['group']=='cold':
                    assert r['predictions']==source['predictions']
            checked+=1
    result={'valid':True,'rows':len(rows),'candidates':len(available),'sha256':m['submission_sha256'],
        'original_template_order_lossless_ids':True,'five_distinct_positive_available_ids':True,
        'evaluated_model_never_fit_main_or_earlier_labels':True,'all_final_supervision_before_hidden_window':True,
        'independent_model_reload_and_all_user_ranking_reconstruction':True,
        'offline_method_files_verified':checked,'v1_v2_and_prior_evidence_unchanged':True,'kaggle_requests':0}
    save(out/'v3_independent_validation.json',result)
    m['independent_validation']=result
    m['verification_source_sha256']=digest(__file__)
    if (out/'test_results.json').exists():
        m['tests']=json.loads((out/'test_results.json').read_text())
    save(p,m)
    print(result)
    c.close()


if __name__=='__main__':
    main()
