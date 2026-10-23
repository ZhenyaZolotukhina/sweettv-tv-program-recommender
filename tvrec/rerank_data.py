"""Cutoff-specific retrieval and features for leakage-safe supervised reranking.

No validation targets are arguments to feature construction. Histories, ALS,
TF-IDF, popularity and channel models are rebuilt at each new rolling cutoff.
"""
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .advanced import metadata
from .final_refit import digest
from .improve import load_baseline,schedule_frames
from .labels import event_frame,training_features
from .models import (ImplicitALS,ContentProfiles,matrix_from_features,popularity_scores,
                      reciprocal_ranks,measured)
from .pipeline import LOW,SPLIT,TEST,END,save
from .rank_features import qualified_occurrences,channel_scores


EARLIER=int(np.datetime64('2020-04-13','s').astype('int64'))


def available_items(c,cutoff,upper,name):
    sql='SELECT DISTINCT item FROM base.airings WHERE item>0 AND duration>0 AND ss<? AND se>?'
    if name=='main':
        sql+=' AND period=21'
    if name=='final':
        sql+=' AND period=31'
    return sorted(i for i, in c.execute(sql,[upper,cutoff]).fetchall())


def order_row(values,fallback):
    order=fallback[np.argsort(-values[fallback],kind='stable')]
    positive=order[values[order]>0]
    seen=set(positive.tolist())
    return np.concatenate([positive,np.array([i for i in fallback if i not in seen],dtype=int)])


def context(c,cache,users,name,cutoff,upper):
    items=available_items(c,cutoff,upper,name)
    training=event_frame(c,LOW,cutoff)
    assert (training.b<=cutoff).all()
    occurrences=qualified_occurrences(training,cutoff)
    warm=set(training.loc[training.b-training.a>=60,'user_id'])
    f=training_features(training,minimum=.5,strength='frequency',cutoff=cutoff)
    timing={}
    if name in ['main','earlier']:
        scores,_=load_baseline(c,cache,users,items,cutoff,upper,name,training)
    elif name=='final':
        mapping=json.loads((cache/'final/hybrid_v1/mappings.json').read_text())
        assert mapping['template_users']==users and mapping['test_candidates']==items
        with np.load(cache/'final/hybrid_v1/component_scores.npz') as z:
            scores={k:z[k] for k in ['repeat','cf','cbf','popularity']}
    else:
        # Each rolling fold gets its OWN historical MF and metadata fit.
        all_users=sorted(set(training.user_id)|set(users))
        all_items=sorted(set(int(i) for i in training.item)|set(items))
        ui={u:n for n,u in enumerate(all_users)}
        ii={i:n for n,i in enumerate(all_items)}
        scores={'repeat':matrix_from_features(f,users,items).toarray().astype(np.float32),
                'popularity':popularity_scores(f,users,items)}
        p=cache/'rerank'/f'{name}_base_scores.npz'
        p.parent.mkdir(exist_ok=True)
        if p.exists():
            with np.load(p) as z:
                scores['cf']=z['cf'];scores['cbf']=z['cbf']
            timing['cached']=True
        else:
            with measured() as fit_time:
                matrix=matrix_from_features(f,all_users,all_items)
                als=ImplicitALS(factors=32,alpha=10,regularization=1,iterations=12,seed=42).fit(matrix)
                with threadpool_limits(limits=1):
                    scores['cf']=np.maximum((als.user_factors[[ui[u] for u in users]]@
                        als.item_factors[[ii[i] for i in items]].T).astype(np.float32),0)
                mt,ma=metadata(c,cutoff,upper,all_items)
                assert all(row['ss']<cutoff for row in mt)
                cbf=ContentProfiles().fit_items(mt,ma)
                with threadpool_limits(limits=1):
                    scores['cbf'],_=cbf.score(matrix_from_features(f,users,all_items),[ii[i] for i in items])
                np.savez_compressed(p,cf=scores['cf'],cbf=scores['cbf'])
            timing['fresh_models']=fit_time
    scores['channel']=channel_scores(occurrences,users,items)
    fallback=np.lexsort((np.array(items),-scores['popularity'][0]))
    ranks={k:reciprocal_ranks(scores[k],fallback,200,10) for k in ['repeat','cf','cbf','popularity','channel']}
    scores['v2']=ranks['repeat']+.1*ranks['cf']+.1*ranks['cbf']+.1*ranks['popularity']+.5*ranks['channel']
    past,future=schedule_frames(c,cutoff,upper)
    del training,ranks
    gc.collect()
    return {'users':users,'items':items,'cutoff':cutoff,'upper':upper,'occurrences':occurrences,
            'warm':warm,'scores':scores,'fallback':fallback,'past':past,'future':future,'timing':timing}


def retrieve(ctx):
    """Pool = V2 Top-200 UNION positive-score Top-50 of all five components."""
    pools=[];tops={k:[] for k in ['repeat','cf','cbf','popularity','channel','v2']}
    ranks=[]
    for n,u in enumerate(ctx['users']):
        orders={k:order_row(ctx['scores'][k][n],ctx['fallback']) for k in tops}
        if u not in ctx['warm']:
            orders={k:ctx['fallback'] for k in tops}
        for k in tops:
            tops[k].append(orders[k][:200])
        base=orders['v2']
        inverse=np.empty(len(base),dtype=int);inverse[base]=np.arange(len(base))
        proposed=[base[:200]]
        for k in ['repeat','cf','cbf','popularity','channel']:
            top=orders[k][:50]
            proposed.append(top[ctx['scores'][k][n,top]>0])
        pool=np.unique(np.concatenate(proposed))
        pools.append(pool[np.argsort(inverse[pool],kind='stable')])
    return pools,{k:np.asarray(v) for k,v in tops.items()}


def feature_vocabulary(c):
    cutoff=int(np.datetime64('2020-03-23','s').astype('int64'))
    categories=[r[0] for r in c.execute('''SELECT DISTINCT category FROM base.airings
        WHERE item>0 AND ss<? AND category IS NOT NULL ORDER BY category''',[cutoff]).fetchall()]
    genres=[r[0] for r in c.execute('''SELECT genre,count(*) n FROM (
        SELECT tv_show_genre_1 genre FROM base.schedule s JOIN base.airings a USING(sid) WHERE a.ss<? AND a.item>0
        UNION ALL SELECT tv_show_genre_2 FROM base.schedule s JOIN base.airings a USING(sid) WHERE a.ss<? AND a.item>0
        UNION ALL SELECT tv_show_genre_3 FROM base.schedule s JOIN base.airings a USING(sid) WHERE a.ss<? AND a.item>0)
        WHERE genre IS NOT NULL GROUP BY genre ORDER BY n DESC,genre LIMIT 12''',[cutoff]*3).fetchall()]
    return {'categories':categories,'genres':genres,'fit_cutoff':cutoff}


def build_features(ctx,pools,vocab):
    users=ctx['users'];items=ctx['items'];cutoff=ctx['cutoff'];upper=ctx['upper']
    occ=ctx['occurrences'];scores=ctx['scores'];fallback=ctx['fallback']
    user_index=np.repeat(np.arange(len(users)),[len(p) for p in pools])
    item_index=np.concatenate(pools)
    pair_users=np.array(users,dtype=object)[user_index]
    pair_items=np.array(items)[item_index]
    columns={}
    component_names=['repeat','cf','cbf','popularity','channel','v2']
    for name in component_names:
        values=scores[name]
        maxima=np.maximum(values.max(axis=1),1e-20)
        columns[name+'_score']=values[user_index,item_index]/maxima[user_index]
        all_rank=[]
        for n,pool in enumerate(pools):
            order=order_row(values[n],fallback)
            rank=np.empty(len(items),dtype=np.int32);rank[order]=np.arange(1,len(items)+1)
            all_rank.append(rank[pool])
        r=np.concatenate(all_rank)
        columns[name+'_reciprocal_rank']=1/(10+r)
    o=occ.copy()
    o['strong']=(5*o.seconds>=4*o.duration).astype(float)
    o['recent14']=(o.b>=cutoff-14*86400).astype(float)
    pairs=o.groupby(['user_id','item']).agg(frequency=('sid','size'),strong=('strong','sum'),
        seconds=('seconds','sum'),mean_fraction=('fraction','mean'),last=('b','max'),
        first=('a','min'),recent14=('recent14','sum'))
    idx=pd.MultiIndex.from_arrays([pair_users,pair_items],names=['user_id','item'])
    p=pairs.reindex(idx)
    for col in ['frequency','strong','seconds','recent14']:
        columns['history_log_'+col]=np.log1p(p[col].fillna(0).to_numpy())
    columns['history_mean_fraction']=p.mean_fraction.fillna(0).to_numpy()
    columns['history_days_since_last']=np.minimum((cutoff-p['last'].fillna(cutoff-365*86400).to_numpy())/86400,365)
    columns['history_days_since_first']=np.minimum((cutoff-p['first'].fillna(cutoff).to_numpy())/86400,365)
    columns['history_strong_ratio']=p.strong.fillna(0).to_numpy()/np.maximum(p.frequency.fillna(0).to_numpy(),1)
    columns['history_rate']=p.frequency.fillna(0).to_numpy()/((cutoff-LOW)/86400)
    u=o.groupby('user_id').agg(airings=('sid','size'),items=('item','nunique'),recent14=('recent14','sum'))
    uf=u.reindex(pair_users).fillna(0)
    columns['user_log_airings']=np.log1p(uf.airings.to_numpy())
    columns['user_log_items']=np.log1p(uf['items'].to_numpy())
    columns['user_repeat_fraction']=1-uf['items'].to_numpy()/np.maximum(uf.airings.to_numpy(),1)
    columns['user_recent_fraction']=uf.recent14.to_numpy()/np.maximum(uf.airings.to_numpy(),1)
    global_count=o.groupby('item').size()
    columns['item_log_global_count']=np.log1p(global_count.reindex(pair_items).fillna(0).to_numpy())
    for name,frame,days in [('past',ctx['past'],(cutoff-LOW)/86400),('future',ctx['future'],(upper-cutoff)/86400)]:
        counts=frame.groupby('item').size().reindex(pair_items).fillna(0).to_numpy()
        columns[name+'_airing_rate']=np.log1p(counts/days)
        duration=frame.groupby('item').duration.mean().reindex(pair_items).fillna(0).to_numpy()
        columns[name+'_log_duration']=np.log1p(duration)
        channels=frame.groupby('item').channel.nunique().reindex(pair_items).fillna(0).to_numpy()
        columns[name+'_channels']=channels
    columns['history_days']=np.full(len(user_index),(cutoff-LOW)/86400)
    columns['target_days']=np.full(len(user_index),(upper-cutoff)/86400)
    # Canonical known metadata at this cutoff/known target schedule only.
    cmeta=ctx['metadata']
    for category in vocab['categories']:
        mask=np.array([r.get('tv_show_category')==category for r in cmeta],dtype=np.float32)
        columns['category_'+category]=mask[item_index]
    for genre in vocab['genres']:
        mask=np.array([genre in [r.get('tv_show_genre_'+str(k)) for k in [1,2,3]] for r in cmeta],dtype=np.float32)
        columns['genre_'+genre]=mask[item_index]
    names=list(columns)
    X=np.column_stack([columns[n] for n in names]).astype(np.float32)
    assert np.isfinite(X).all()
    return X,names,user_index,item_index


def cached_feature_dataset(c,cache,users,name,cutoff,upper,vocab):
    folder=cache/'rerank'/name
    folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'features_manifest.json'
    signature={'cutoff':cutoff,'upper':upper,'vocab':vocab,'source':digest(__file__),
        'models_source':digest('tvrec/models.py'),'labels_source':digest('tvrec/labels.py')}
    if marker.exists() and json.loads(marker.read_text())['signature']==signature:
        meta=json.loads(marker.read_text())
        return folder,meta
    with measured() as timing:
        ctx=context(c,cache,users,name,cutoff,upper)
        pools,tops=retrieve(ctx)
        _,ctx['metadata']=metadata(c,cutoff,upper,ctx['items'])
        X,names,ui,ii=build_features(ctx,pools,vocab)
        np.save(folder/'X.npy',X)
        np.savez_compressed(folder/'indices.npz',user_index=ui,item_index=ii,
            pair_v2=scores_at_pairs(ctx,pools,'v2'),
            fallback=ctx['fallback'],cold=np.array([u not in ctx['warm'] for u in users]),**tops)
        metadata_record={'users':users,'items':ctx['items'],'feature_names':names,
                         'pool_lengths':[len(p) for p in pools],'model_rebuild':ctx['timing']}
    meta={'signature':signature,**metadata_record,'resources':timing,'rows':len(ui),'columns':len(names)}
    save(marker,meta)
    return folder,meta


def scores_at_pairs(ctx,pools,name):
    return np.concatenate([ctx['scores'][name][n,p] for n,p in enumerate(pools)])
