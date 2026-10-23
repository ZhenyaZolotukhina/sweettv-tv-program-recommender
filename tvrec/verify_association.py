"""Independent rule algebra, model predictions and unchanged-cohort checks."""
import csv
import json
from pathlib import Path
import pickle

import numpy as np
from scipy import sparse

from .core import ap_at_5
from .final_refit import digest,check_evidence
from .group_rank import lightgbm_module
from .improve import connect
from .pipeline import save
from .verify_final import validate_file_records


def main():
    out=Path('artifacts/stage9');frozen=json.loads((out/'frozen_prior.json').read_text())['files'];check_evidence(frozen)
    decision=json.loads((out/'decision.json').read_text());assert decision['selected']=='retain_v5' and decision['experiments_run']<=2
    assert not Path('artifacts/final/submission_hybrid_v8.csv').exists()
    with Path('sweettv-tv-program-recommender/submission.csv').open(newline='') as f:
        reader=csv.reader(f);assert next(reader)==['user_id','tv_show_id'];users=[r[0] for r in reader]
    c=connect(Path('cache'));rule_files=integration_files=0
    for window,lower,upper in [('main',1589760000,1595808000),('earlier',1586736000,1589760000)]:
        base=Path('cache/rerank')/window;bm=json.loads((base/'features_manifest.json').read_text())
        assert bm['users']==users
        with np.load(base/'indices.npz') as z:cold=z['cold']
        with Path(f'artifacts/stage6/private/{window}_lambda_top5.csv').open(newline='') as f:
            original={r['user_id']:r for r in csv.DictReader(f)}
        sql='SELECT DISTINCT item FROM base.airings WHERE item>0 AND duration>0 AND ss<? AND se>?'
        if window=='main':sql+=' AND period=21'
        available={i for i, in c.execute(sql,[upper,lower]).fetchall()}
        with np.load(f'cache/improvements/{window}_v1_scores.npz') as z:pop=z['popularity'][0]
        fallback=np.lexsort((np.array(bm['items']),-pop));prior_pos=np.empty(len(fallback),int);prior_pos[fallback]=np.arange(len(fallback))
        for kind in ['confidence','shrink_cosine']:
            folder=Path('cache/association')/window/kind;info=json.loads((folder/'manifest.json').read_text())
            assert info['signature']['cutoff']==lower and info['last_historical_second']<=lower
            with (folder/'model.pkl').open('rb') as f:model=pickle.load(f)
            b=model.frequency.copy();b.data.fill(1)
            support=np.asarray(b.sum(axis=0)).ravel();np.testing.assert_array_equal(support,model.support)
            count=(b.T@b).tocsr();count.setdiag(0);count.eliminate_zeros()
            assert (count!=model.co).nnz==0
            s=model.similarity.tocoo();co=np.asarray(count[s.row,s.col]).ravel()
            assert np.all(co>=info['signature']['config']['min_users']) and np.all(s.row!=s.col)
            expected=co/(support[s.row]+10) if kind=='confidence' else co/np.sqrt(support[s.row]*support[s.col])*co/(co+20)
            np.testing.assert_allclose(s.data,expected,rtol=1e-6,atol=1e-7)
            assert np.all(np.diff(model.similarity.indptr)<=200)
            profile=model.frequency.copy();profile.data=np.log1p(profile.data)
            total=np.asarray(profile.sum(axis=1)).ravel();profile=sparse.diags(np.divide(1,total,out=np.zeros_like(total),where=total>0))@profile
            um={u:n for n,u in enumerate(info['model_users'])};im={int(i):n for n,i in enumerate(info['model_items'])}
            uq=np.array([um.get(u,-1) for u in users]);it=np.array([im.get(int(i),-1) for i in bm['items']]);uk=uq>=0;ik=it>=0
            scores=np.zeros((len(users),len(it)),dtype=np.float32)
            scores[np.ix_(uk,ik)]=(profile[uq[uk]]@model.similarity).toarray()[:,it[ik]]
            with np.load(folder/'scores.npz') as z:np.testing.assert_array_equal(scores,z['score'])
            with (out/'private'/f'{window}_association_{kind}.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            validate_file_records(users,['user_id','tv_show_id'],[[r['user_id'],r['predictions']] for r in rows],available)
            for n,r in enumerate(rows):
                chosen=np.lexsort((prior_pos,-scores[n]))[:5] if not cold[n] else fallback[:5]
                assert [bm['items'][i] for i in chosen]==list(map(int,r['predictions'].split()))
            rule_files+=1
        for variant in ['association','bpr_calibrated']:
            folder=Path('cache/association_integration' if variant=='association' else 'cache/bpr_calibrated')/window
            meta=json.loads((folder/'manifest.json').read_text());X=np.load(folder/'X.npy',mmap_mode='r')
            assert meta['users']==users and meta['cutoff']==lower and set(meta['items'])==available
            with np.load(folder/'indices.npz') as z:idx=z['item_index'];ui=z['user_index'];cc=z['cold'];ff=z['fallback']
            assert np.array_equal(ui,np.repeat(np.arange(len(users)),meta['pool_lengths']))
            if variant=='bpr_calibrated':
                old=Path('cache/new_cf/log_frequency')/window/'expanded200'
                with np.load(old/'indices.npz') as z:np.testing.assert_array_equal(idx,z['item_index'])
                np.testing.assert_array_equal(X[:,:59],np.load(old/'X.npy',mmap_mode='r')[:,:59])
                assert not np.isinf(X).any()  # NaNs are explicitly allowed for missing CF values
            model=lightgbm_module().Booster(model_file=f'cache/association_integration/{variant}_early.txt')
            assert X.shape[1]==model.num_feature()==(64 if variant=='association' else 67)
            score=model.predict(X);np.testing.assert_array_equal(score,np.load(folder/(variant+'_scores.npy')))
            with (out/'private'/f'{window}_{variant}.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            validate_file_records(users,['user_id','tv_show_id'],[[r['user_id'],r['predictions']] for r in rows],available)
            starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
            for n,r in enumerate(rows):
                lo,hi=starts[n:n+2];positions=np.arange(hi-lo)
                chosen=idx[lo:hi][np.lexsort((positions,-score[lo:hi]))[:5]] if not cc[n] else ff[:5]
                assert [meta['items'][i] for i in chosen]==list(map(int,r['predictions'].split()))
            integration_files+=1
        # Every reported file uses identical frozen truth and AP convention.
        # Recovered-candidate audit tables are intentionally not full-cohort
        # prediction files. Verify the four explicitly declared methods here.
        for suffix in ['association_confidence','association_shrink_cosine','association','bpr_calibrated']:
            path=out/'private'/f'{window}_{suffix}.csv'
            with path.open(newline='') as f:rows=list(csv.DictReader(f))
            assert [r['user_id'] for r in rows]==users
            for r in rows:
                old=original[r['user_id']];truth=set(map(int,r['truth'].split()));pred=list(map(int,r['predictions'].split()))
                assert truth==set(map(int,old['truth'].split())) and r['group']==old['group']
                assert abs(ap_at_5(truth,pred)-float(r['ap5']))<1e-12
                assert [i for i,p in enumerate(pred,1) if p in truth]==list(map(int,r['hit_ranks'].split()))
                if r['group']=='cold':assert r['predictions']==old['predictions']
    for r in decision['results']:
        assert all(f['target_end']<=1586736000 for f in r['training_folds'])
    check_evidence(frozen);c.close()
    result={'valid':True,'standalone_rule_files_reconstructed':rule_files,'integration_files_reconstructed':integration_files,
        'all_users1260_same_truth_groups_availability_and_ap':True,'distinct_user_support_and_rule_weights_verified':True,
        'existing_bpr_pool_and_first59_controls_unchanged':True,'all_rule_and_ranker_scores_reloaded':True,
        'holdout_labels_not_used_in_fit':True,'prior_models_submissions_and_evidence_unchanged':True,
        'no_v8_csv_created':True,'retained_v5_sha256':digest('artifacts/final/submission_hybrid_v5.csv'),'kaggle_requests':0}
    save(out/'independent_validation.json',result);print(result)


if __name__=='__main__':main()
