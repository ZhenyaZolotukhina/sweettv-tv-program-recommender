"""Three coherent extra feature blocks, each tested separately against V3.

Inputs are deduplicated pre-cutoff user/airing observations and known schedules.
There are no labels or future user views in feature function arguments.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from .pipeline import LOW
from .rank_features import time_slots


def check_occurrences(occ,cutoff=None):
    if occ.duplicated(['user_id','sid']).any():
        raise ValueError('Same user/airing counted more than once')
    if cutoff is not None and not ((occ.a<cutoff)&(occ.b<=cutoff)&(occ.b>occ.a)).all():
        raise ValueError('Future or invalid observation in focused feature history')


def consistency_features(occ,pair_users,pair_items,cutoff):
    check_occurrences(occ,cutoff)
    o=occ.copy()
    o['day']=o.a//86400
    o['week']=(o['day']-LOW//86400)//7
    strong=o.loc[5*o.seconds>=4*o.duration]
    soft=o.groupby(['user_id','item']).agg(days=('day','nunique'),weeks=('week','nunique'),n=('sid','size'))
    hard=strong.groupby(['user_id','item']).agg(days=('day','nunique'),weeks=('week','nunique'),
        first_week=('week','min'),last=('b','max'))
    idx=pd.MultiIndex.from_arrays([pair_users,pair_items],names=['user_id','item'])
    s=soft.reindex(idx).fillna(0);h=hard.reindex(idx)
    last_week=((cutoff-1)//86400-LOW//86400)//7
    span=np.maximum(1,last_week-h.first_week.fillna(last_week).to_numpy()+1)
    return {
        'cons_log_days50':np.log1p(s.days.to_numpy()),
        'cons_log_weeks50':np.log1p(s.weeks.to_numpy()),
        'cons_log_days80':np.log1p(h.days.fillna(0).to_numpy()),
        'cons_log_weeks80':np.log1p(h.weeks.fillna(0).to_numpy()),
        'cons_completed_week_share':h.weeks.fillna(0).to_numpy()/span,
        'cons_days_since_completion':np.minimum((cutoff-h['last'].fillna(cutoff-365*86400).to_numpy())/86400,365),
        'cons_log_airings_per_day':np.log1p(s.n.to_numpy()/np.maximum(s.days.to_numpy(),1)),
        'cons_completed_day_fraction':h.days.fillna(0).to_numpy()/np.maximum(s.days.to_numpy(),1)}


def loyalty_features(occ,future,pair_users,pair_items):
    check_occurrences(occ)
    o=occ.copy();o['strong']=(5*o.seconds>=4*o.duration).astype(float)
    users=o.groupby('user_id').agg(total=('sid','size'),seconds=('seconds','sum'))
    uc=o.groupby(['user_id','channel']).agg(n=('sid','size'),seconds=('seconds','sum'),strong=('strong','sum')).reset_index()
    uc=uc.merge(users.reset_index(),on='user_id',suffixes=('_channel','_user'),validate='many_to_one')
    uc['raw_loyalty']=uc.n/uc.total
    uc['duration_loyalty']=uc.seconds_channel/np.maximum(uc.seconds_user,1)
    uc['completion_ratio']=uc.strong/np.maximum(uc.n,1)
    uc['log_channel_intensity']=np.log1p(uc.n)
    global_pair=o.groupby(['channel','item']).strong.sum().rename('complete_item').reset_index()
    global_channel=o.groupby('channel').strong.sum().rename('complete_channel').reset_index()
    global_pair=global_pair.merge(global_channel,on='channel',validate='many_to_one')
    global_pair['complete_popularity']=global_pair.complete_item/np.maximum(global_pair.complete_channel,1)
    mix=future.groupby(['item','channel']).size().rename('airings').reset_index()
    mix['mix']=mix.airings/mix.groupby('item').airings.transform('sum')
    pairs=pd.DataFrame({'pair_id':np.arange(len(pair_users)),'user_id':pair_users,'item':pair_items})
    expanded=pairs.merge(mix[['item','channel','mix']],on='item',how='left')
    expanded=expanded.merge(uc[['user_id','channel','raw_loyalty','duration_loyalty',
        'completion_ratio','log_channel_intensity']],on=['user_id','channel'],how='left',validate='many_to_one')
    expanded=expanded.merge(global_pair[['channel','item','complete_popularity']],
        on=['channel','item'],how='left',validate='many_to_one').fillna(0)
    expanded['loyalty_complete_popularity']=expanded.raw_loyalty*expanded.complete_popularity
    names=['raw_loyalty','duration_loyalty','completion_ratio','log_channel_intensity',
           'complete_popularity','loyalty_complete_popularity']
    return {'loyal_'+name:np.bincount(expanded.pair_id.astype(int),
        weights=expanded[name]*expanded['mix'],minlength=len(pair_users)) for name in names}


def opportunity_features(occ,past,future,users,items,pair_user_index,pair_item_index,cutoff,upper):
    check_occurrences(occ,cutoff)
    strong=occ.loc[5*occ.seconds>=4*occ.duration]
    pair_users=np.array(users,dtype=object)[pair_user_index]
    pair_items=np.array(items)[pair_item_index]
    counts=strong.groupby(['user_id','item']).size()
    n=np.array([counts.get((u,int(i)),0) for u,i in zip(pair_users,pair_items)],dtype=float)
    past_n=past.groupby('item').size().reindex(pair_items).fillna(0).to_numpy()
    future_n=future.groupby('item').size().reindex(pair_items).fillna(0).to_numpy()
    completion_rate=np.minimum(n/np.maximum(past_n,1),1)
    reliability=n/(n+3)
    expected=completion_rate*future_n*reliability
    # Timing is based on the USER'S historical completed-airing midpoints and
    # the known schedule, never future viewing. Laplace prior is fixed at one.
    ui={u:i for i,u in enumerate(users)};ii={int(i):n for n,i in enumerate(items)}
    profile=np.ones((len(users),42),dtype=np.float64)
    for user,slot in zip(strong.user_id,time_slots((strong.a.to_numpy()+strong.b.to_numpy())//2)):
        if user in ui:
            profile[ui[user],slot]+=1
    profile/=profile.sum(axis=1,keepdims=True)
    slots=time_slots(future.ss.to_numpy()+future.duration.to_numpy()//2)
    rows=[];cols=[]
    for item,slot in zip(future.item,slots):
        if int(item) in ii:
            rows.append(int(slot));cols.append(ii[int(item)])
    slot_item=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(42,len(items)))
    compatible=np.asarray(profile@slot_item)
    compatibility=compatible[pair_user_index,pair_item_index]/np.maximum(future_n,1)*42
    # All opportunity scores are gated by personal completion evidence. More
    # scheduled broadcasts alone never creates a positive personal score.
    return {'opportunity_personal_completion_rate':completion_rate,
        'opportunity_log_expected_completions':np.log1p(expected),
        'opportunity_log_timing_expected':np.log1p(expected*compatibility),
        'opportunity_log_supported_broadcast_rate':np.log1p(
            compatible[pair_user_index,pair_item_index]*reliability/((upper-cutoff)/86400))}
