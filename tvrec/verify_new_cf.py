"""Independent cutoff, CF score and downstream ordering checks; no V7 output."""
import csv
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from .core import ap_at_5
from .final_refit import digest,check_evidence
from .group_rank import lightgbm_module
from .improve import connect
from .pipeline import save
from .verify_final import validate_file_records


def main():
    out=Path('artifacts/stage8');frozen=json.loads((out/'frozen_prior.json').read_text())['files'];check_evidence(frozen)
    assert json.loads((out/'decision.json').read_text())['selected']=='retain_v5'
    assert not Path('artifacts/final/submission_hybrid_v7.csv').exists()
    with Path('sweettv-tv-program-recommender/submission.csv').open(newline='') as f:
        reader=csv.reader(f);assert next(reader)==['user_id','tv_show_id'];users=[r[0] for r in reader]
    c=connect(Path('cache'));cf_files=downstream_files=0
    for window,cutoff,upper in [('main',1589760000,1595808000),('earlier',1586736000,1589760000)]:
        base=Path('cache/rerank')/window;meta=json.loads((base/'features_manifest.json').read_text())
        assert meta['users']==users and meta['signature']['cutoff']==cutoff
        sql='SELECT DISTINCT item FROM base.airings WHERE item>0 AND duration>0 AND ss<? AND se>?'
        if window=='main':sql+=' AND period=21'
        available={i for i, in c.execute(sql,[upper,cutoff]).fetchall()};assert set(meta['items'])==available
        with np.load(base/'indices.npz') as loaded:z={k:loaded[k] for k in loaded.files}
        with np.load(f'cache/improvements/{window}_v1_scores.npz') as loaded:pop=loaded['popularity'][0]
        fallback=np.lexsort((np.array(meta['items']),-pop));prior=np.empty(len(fallback),int);prior[fallback]=np.arange(len(fallback))
        with Path(f'artifacts/stage6/private/{window}_lambda_top5.csv').open(newline='') as f:
            original={r['user_id']:r for r in csv.DictReader(f)}
        # Independent DISTINCT-airing SQL builds original frequency evidence.
        history=c.execute('''WITH qualified AS(SELECT DISTINCT e.user_id,e.sid,a.item
            FROM continuous_events e JOIN base.airings a USING(sid)
            WHERE a.item>0 AND a.duration>0 AND e.a<? AND e.b>1583712000
            AND 2*(least(e.b,?)-greatest(e.a,1583712000))>=a.duration)
            SELECT user_id,item,count(*) frequency FROM qualified GROUP BY user_id,item''',[cutoff,cutoff]).fetchdf()
        for weighting in ['binary','log_frequency']:
            folder=Path('cache/new_cf')/weighting/window;info=json.loads((folder/'manifest.json').read_text())
            assert info['signature']['cutoff']==cutoff and info['historical_last_second']<=cutoff
            with (folder/'model.pkl').open('rb') as f:model=pickle.load(f)
            um={u:i for i,u in enumerate(info['model_users'])};im={int(i):n for n,i in enumerate(info['model_items'])}
            assert set(um)==set(history.user_id) and set(im)==set(history.item)
            expected=sparse.csr_matrix((history.frequency.to_numpy(),
                ([um[u] for u in history.user_id],[im[int(i)] for i in history.item])),shape=model.matrix.shape)
            assert (model.matrix!=expected).nnz==0
            query=np.array([um.get(u,-1) for u in users]);candidate=np.array([im.get(int(i),-1) for i in meta['items']])
            uk=query>=0;ik=candidate>=0
            reconstructed=np.full((len(users),len(meta['items'])),-np.inf,dtype=np.float32)
            with threadpool_limits(limits=1):
                reconstructed[np.ix_(uk,ik)]=(model.user_factors[query[uk]]@model.item_factors[candidate[ik]].T+
                    model.item_bias[candidate[ik],0][None,:]).astype(np.float32)
            with np.load(folder/'scores.npz') as q:np.testing.assert_array_equal(reconstructed,q['score'])
            with (out/'private'/f'{window}_bpr_{weighting}.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            validate_file_records(users,['user_id','tv_show_id'],[[r['user_id'],r['predictions']] for r in rows],available)
            for n,r in enumerate(rows):
                order=np.lexsort((prior,-reconstructed[n]))[:5] if not z['cold'][n] else fallback[:5]
                assert [meta['items'][i] for i in order]==list(map(int,r['predictions'].split()))
                truth=set(map(int,original[r['user_id']]['truth'].split()));pred=list(map(int,r['predictions'].split()))
                assert set(map(int,r['truth'].split()))==truth
                assert abs(ap_at_5(truth,pred)-float(r['ap5']))<1e-12
                assert r['group']==original[r['user_id']]['group']
                if z['cold'][n]:assert r['predictions']==original[r['user_id']]['predictions']
            cf_files+=1
        for variant,model_path in [('same_pool_features','cache/new_cf/same_pool_ranker_early.txt'),
                                   ('expanded200','cache/new_cf/expanded_ranker_early.txt')]:
            if variant=='same_pool_features':
                X=np.column_stack([np.load(base/'X.npy'),np.load(Path('cache/focused_v4')/window/'loyalty.npy'),
                    np.load(Path('cache/new_cf/log_frequency')/window/'pair_features.npy')])
                current_meta=meta;indices=z
                expected_score=np.load(f'cache/new_cf/{window}_same_pool_ranker_scores.npy')
            else:
                folder=Path('cache/new_cf/log_frequency')/window/'expanded200'
                current_meta=json.loads((folder/'manifest.json').read_text());X=np.load(folder/'X.npy',mmap_mode='r')
                assert current_meta['cutoff']==cutoff and current_meta['existing_features_rebuilt_sample_match']
                with np.load(folder/'indices.npz') as loaded:indices={k:loaded[k] for k in loaded.files}
                expected_score=np.load(folder/'expanded_ranker_scores.npy')
            model=lightgbm_module().Booster(model_file=model_path)
            assert X.shape[1]==model.num_feature()==67
            score=model.predict(X);np.testing.assert_array_equal(score,expected_score)
            with (out/'private'/f'{window}_{variant}.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            validate_file_records(users,['user_id','tv_show_id'],[[r['user_id'],r['predictions']] for r in rows],available)
            starts=np.r_[0,np.cumsum(current_meta['pool_lengths'])]
            assert np.array_equal(indices['user_index'],np.repeat(np.arange(len(users)),current_meta['pool_lengths']))
            for n,r in enumerate(rows):
                lo,hi=starts[n:n+2];positions=np.arange(hi-lo);order=np.lexsort((positions,-score[lo:hi]))
                picked=indices['item_index'][lo:hi][order[:5]] if not indices['cold'][n] else indices['fallback'][:5]
                assert [current_meta['items'][i] for i in picked]==list(map(int,r['predictions'].split()))
                before=original[r['user_id']];truth=set(map(int,before['truth'].split()));pred=list(map(int,r['predictions'].split()))
                assert truth==set(map(int,r['truth'].split())) and r['group']==before['group']
                assert abs(ap_at_5(truth,pred)-float(r['ap5']))<1e-12
                assert [rank for rank,item in enumerate(pred,1) if item in truth]==list(map(int,r['hit_ranks'].split()))
                if indices['cold'][n]:assert r['predictions']==before['predictions']
            downstream_files+=1
    # Rolling models' factors/mappings are their own, never later holdout fits.
    for name,cutoff in [('roll_mar23',1584921600),('roll_apr06',1586131200)]:
        info=json.loads(Path(f'cache/new_cf/log_frequency/{name}/manifest.json').read_text())
        assert info['signature']['cutoff']==cutoff and info['historical_last_second']<=cutoff
    all_results=json.loads((out/'decision.json').read_text())['results']
    assert all(f['target_end']<=1586736000 for r in all_results for f in r['training_folds'])
    check_evidence(frozen);c.close()
    result={'valid':True,'standalone_cf_files_reconstructed':cf_files,'reranker_files_reconstructed':downstream_files,
        'original_users_all1260_and_candidates_preserved':True,'same_frozen_targets_and_cold_fallback':True,
        'independent_sql_positive_matrices_match':True,'new_cf_scores_recomputed_from_factors_and_bias':True,
        'all_learned_rankings_recomputed':True,'rolling_models_before_own_cutoffs':True,
        'no_eval_labels_fit_ranker':True,'prior_evidence_v1_v5_unchanged':True,'no_v7_csv_created':True,
        'retained_v5_sha256':digest('artifacts/final/submission_hybrid_v5.csv'),'kaggle_requests':0}
    save(out/'independent_validation.json',result);print(result)


if __name__=='__main__':main()
