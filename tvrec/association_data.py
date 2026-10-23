"""Candidate/feature data for the two scoped integrations, cutoff-correct only."""
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.stats import rankdata

from .advanced import metadata
from .assoc_study import dataset,association_fallback
from .final_refit import digest
from .focused import load_dataset
from .focused_features import loyalty_features
from .ltr_experiment import matrix
from .models import measured
from .pipeline import save
from .rerank_data import context,build_features,order_row


AR_FEATURES=['association_score_relative','association_reciprocal_rank',
             'association_history_seed_fraction','association_seed_weight_mass','association_item_user_support']
CAL_FEATURES=['cal_bpr_score_z','cal_bpr_score_percentile','cal_bpr_item_bias_percentile','cal_bpr_cosine',
              'cal_bpr_user_norm_percentile','cal_bpr_item_norm_percentile','cal_bpr_user_known','cal_bpr_item_known']


def ecdf(training_values,values):
    reference=np.sort(np.asarray(training_values))
    if not len(reference):return np.full(len(values),np.nan)
    values=np.asarray(values)
    return (np.searchsorted(reference,values,'left')+np.searchsorted(reference,values,'right')+1)/(2*(len(reference)+1))


def calibrated_bpr_features(raw,bias,user_norm,item_norm,user_known,item_known,ui,ii,
                            historical_user_norms,historical_item_norms,historical_biases):
    X=np.full((len(ui),8),np.nan,dtype=np.float32)
    bias_pct=ecdf(historical_biases,bias);item_pct=ecdf(historical_item_norms,item_norm)
    user_pct=ecdf(historical_user_norms,user_norm)
    for user in np.unique(ui):
        at=np.flatnonzero(ui==user);items=ii[at]
        known=np.isfinite(raw[user])&item_known
        if user_known[user] and known.any():
            values=raw[user,known].astype(float)
            mean=values.mean();std=max(values.std(),1e-6)
            supported=item_known[items]&np.isfinite(raw[user,items])
            target=at[supported];selected=items[supported]
            X[target,0]=(raw[user,selected]-mean)/std
            X[target,1]=ecdf(values,raw[user,selected])
            dot=raw[user,selected]-bias[selected];denom=user_norm[user]*item_norm[selected]
            cosine=np.divide(dot,denom,out=np.full(len(target),np.nan),where=denom>0)
            X[target,3]=np.clip(cosine,-1,1)
            X[at,4]=user_pct[user]
        supported=item_known[items]
        X[at[supported],2]=bias_pct[items[supported]]
        X[at[supported],5]=item_pct[items[supported]]
        X[at,6]=user_known[user];X[at,7]=item_known[items]
    assert not np.isinf(X).any()  # NaN is an explicit missing value for LightGBM.
    return X


def ar_columns(scores,count,mass,support,user_index,item_index,history_items,fallback):
    result=np.zeros((len(user_index),5),dtype=np.float32)
    for u in np.unique(user_index):
        at=np.flatnonzero(user_index==u);target=item_index[at]
        maxscore=max(float(scores[u].max()),1e-20)
        order=fallback[np.argsort(-scores[u,fallback],kind='stable')]
        rank=np.empty(len(order),int);rank[order]=np.arange(1,len(order)+1)
        result[at,0]=scores[u,target]/maxscore
        result[at,1]=1/(10+rank[target])
        result[at,2]=count[u,target]/max(int(history_items[u]),1)
        result[at,3]=mass[u,target];result[at,4]=support[target]
    return result


def ar_dataset(c,name,kind):
    folder=Path('cache/association_integration')/name;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'manifest.json'
    signature={'kind':kind,'source':digest(__file__),'score_source':digest('tvrec/association_model.py')}
    if marker.exists() and json.loads(marker.read_text())['signature']==signature:return folder,json.loads(marker.read_text())
    with measured() as resources:
        old_X,meta,z,names=matrix(name)
        score_folder,score_info=dataset(c,name,kind)
        with np.load(score_folder/'scores.npz') as q:
            score=q['score'];seed_count=q['seed_count'];seed_mass=q['seed_mass'];support=q['item_support']
        fallback=association_fallback(name,score_folder,meta)
        starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
        # Reuse any already-correct 59-feature pairs in the previous BPR cache.
        bpr_folder=Path('cache/new_cf/log_frequency')/name/'expanded200'
        bm=json.loads((bpr_folder/'manifest.json').read_text())
        assert bm['cutoff']==meta['signature']['cutoff'] and bm['items']==meta['items']
        BX=np.load(bpr_folder/'X.npy',mmap_mode='r')
        with np.load(bpr_folder/'indices.npz') as q:bui=q['user_index'];bii=q['item_index']
        bs=np.r_[0,np.cumsum(bm['pool_lengths'])]
        # Context scores are historical base models / known availability only.
        ctx=context(c,Path('cache'),meta['users'],name,meta['signature']['cutoff'],meta['signature']['upper'])
        _,ctx['metadata']=metadata(c,ctx['cutoff'],ctx['upper'],ctx['items'])
        missing=[];pools=[];check_rows=[];source_maps=[]
        for n,u in enumerate(meta['users']):
            original=z['item_index'][starts[n]:starts[n+1]]
            graph_order=fallback[np.argsort(-score[n,fallback],kind='stable')][:100]
            graph_order=graph_order[score[n,graph_order]>0] if not z['cold'][n] else np.array([],int)
            combined=np.union1d(original,graph_order)
            global_order=order_row(ctx['scores']['v2'][n],fallback)
            inv=np.empty(len(global_order),int);inv[global_order]=np.arange(len(inv))
            pools.append(combined[np.argsort(inv[combined],kind='stable')])
            oldmap={int(i):('old',starts[n]+j) for j,i in enumerate(original)}
            reused={int(i):('bpr',bs[n]+j) for j,i in enumerate(bii[bs[n]:bs[n+1]])}
            mapping=reused|oldmap
            absent=np.array([i for i in combined if int(i) not in mapping],dtype=int)
            missing.append(np.r_[original[:3],absent]);check_rows.extend((starts[n]+np.arange(3)).tolist());source_maps.append(mapping)
        vocab=json.loads(Path('artifacts/stage4/metadata_feature_vocabulary.json').read_text())
        extra53,fn,mui,mii=build_features(ctx,missing,vocab);assert fn==names[:53]
        pu=np.array(meta['users'],dtype=object)[mui];pi=np.array(meta['items'])[mii]
        channel=loyalty_features(ctx['occurrences'],ctx['future'].drop_duplicates(['channel','ss','duration','item']),pu,pi)
        extra59=np.column_stack([extra53,np.column_stack([channel[k] for k in names[53:]])])
        ms=np.r_[0,np.cumsum([len(p) for p in missing])]
        checks=np.concatenate([ms[n]+np.arange(3) for n in range(len(meta['users']))])
        np.testing.assert_allclose(extra59[checks],old_X[check_rows],rtol=1e-6,atol=1e-5)
        ui=[];ii=[];values=[]
        for n,pool in enumerate(pools):
            fresh={int(i):ms[n]+j for j,i in enumerate(missing[n])}
            for item in pool:
                source=source_maps[n].get(int(item))
                values.append(old_X[source[1]] if source and source[0]=='old' else BX[source[1],:59]
                              if source else extra59[fresh[int(item)]])
            ui.extend([n]*len(pool));ii.extend(pool.tolist())
        ui=np.array(ui,dtype=int);ii=np.array(ii,dtype=int)
        hist=ctx['occurrences'].groupby('user_id').item.nunique()
        nseed=np.array([hist.get(u,0) for u in meta['users']])
        X=np.column_stack([np.asarray(values),ar_columns(score,seed_count,seed_mass,support,ui,ii,nseed,fallback)]).astype(np.float32)
        np.save(folder/'X.npy',X);np.savez_compressed(folder/'indices.npz',user_index=ui,item_index=ii,cold=z['cold'],fallback=fallback)
    record={'signature':signature,'users':meta['users'],'items':meta['items'],'pool_lengths':[len(p) for p in pools],
        'feature_names':names+AR_FEATURES,'cutoff':ctx['cutoff'],'upper':ctx['upper'],
        'existing_feature_identity_checks_passed':True,'resources':resources,'new_asr_top_n':100}
    save(marker,record);return folder,record


def calibrated_dataset(name):
    folder=Path('cache/bpr_calibrated')/name;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'manifest.json';signature={'source':digest(__file__)}
    if marker.exists() and json.loads(marker.read_text())['signature']==signature:return folder,json.loads(marker.read_text())
    old=Path('cache/new_cf/log_frequency')/name/'expanded200';meta=json.loads((old/'manifest.json').read_text())
    with np.load(old/'indices.npz') as q:z={k:q[k] for k in q.files}
    cf=old.parent
    with np.load(cf/'scores.npz') as q:
        raw=q['score'];bias=q['item_bias'];unorm=q['user_norm'];inorm=q['item_norm'];uk=q['user_seen'];ik=q['item_seen']
    with (cf/'model.pkl').open('rb') as f:model=pickle.load(f)
    info=json.loads((cf/'manifest.json').read_text());assert info['signature']['cutoff']==meta['cutoff']
    with measured() as resources:
        new=calibrated_bpr_features(raw,bias,unorm,inorm,uk,ik,z['user_index'],z['item_index'],
            np.linalg.norm(model.user_factors,axis=1),np.linalg.norm(model.item_factors,axis=1),model.item_bias[:,0])
        X=np.column_stack([np.load(old/'X.npy',mmap_mode='r')[:,:59],new]).astype(np.float32)
        np.save(folder/'X.npy',X);np.savez_compressed(folder/'indices.npz',**z)
    record={k:meta[k] for k in ['users','items','pool_lengths','cutoff','upper']}
    record.update(signature=signature,feature_names=meta['feature_names'][:59]+CAL_FEATURES,
        resources=resources,only_bpr_scale_and_missing_preprocessing_changed=True,
        no_future_learned_normalizer=True,base_features_and_candidate_exposure_unchanged=True)
    save(marker,record);return folder,record
