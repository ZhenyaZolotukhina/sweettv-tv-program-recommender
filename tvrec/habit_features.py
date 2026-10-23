"""New habit-relative features, not repeats of rejected generic counts/volume.

Exact channel x weekday x two-hour overlap bins are used for historical completed
viewing and known schedules. No target views/labels enter these functions.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from .focused_features import check_occurrences
from .pipeline import LOW


def active_stability(occ,pair_users,pair_items,cutoff):
    check_occurrences(occ,cutoff)
    o=occ.loc[5*occ.seconds>=4*occ.duration].copy()
    o['day']=((o.a+o.b)//2)//86400
    o['week']=(o.day-LOW//86400)//7
    daily=o.groupby(['user_id','item','day']).size().rename('n').reset_index()
    pairs=daily.groupby(['user_id','item']).agg(n=('n','sum'),days=('day','size'))
    daily=daily.merge(pairs[['n']].rename(columns={'n':'total'}),on=['user_id','item'],validate='many_to_one')
    daily['square']=(daily.n/daily.total)**2
    concentration=daily.groupby(['user_id','item']).square.sum()
    user_days=o.groupby('user_id').day.nunique().to_dict()
    user_weeks={u:sorted(set(g.week)) for u,g in o.groupby('user_id')}
    # A transition is to the NEXT OBSERVED ACTIVE USER WEEK, not necessarily the
    # next calendar week. Inactive weeks do not masquerade as abandoned habits.
    next_week={u:dict(zip(w[:-1],w[1:])) for u,w in user_weeks.items()}
    transitions={}
    for key,g in o.groupby(['user_id','item']):
        u,_=key;weeks=set(g.week);nexts=next_week[u]
        opportunities=[w for w in weeks if w in nexts]
        returns=sum(nexts[w] in weeks for w in opportunities)
        transitions[key]=((returns+1)/(len(opportunities)+2),len(opportunities))
    idx=pd.MultiIndex.from_arrays([pair_users,pair_items],names=['user_id','item'])
    p=pairs.reindex(idx).fillna(0)
    h=concentration.reindex(idx).fillna(0).to_numpy()
    t=np.array([transitions.get((u,int(i)),(.5,0)) for u,i in zip(pair_users,pair_items)])
    denom=np.array([user_days.get(u,0) for u in pair_users])
    return {'stable_completed_active_day_share':p.days.to_numpy()/np.maximum(denom,1),
        'stable_week_return_posterior':t[:,0],
        'stable_log_return_opportunities':np.log1p(t[:,1]),
        'stable_completed_day_concentration':h,
        'stable_log_distributed_completion_mass':np.log1p(p.n.to_numpy()*(1-h))}


def interval_bin_parts(starts,ends,step=7200):
    starts=np.asarray(starts,dtype=np.int64);ends=np.asarray(ends,dtype=np.int64)
    if np.any(ends<=starts):raise ValueError('Nonpositive interval')
    first=starts//step;last=(ends-1)//step;n=last-first+1
    row=np.repeat(np.arange(len(n)),n)
    bins=np.repeat(first,n)+np.arange(len(row))-np.repeat(np.cumsum(n)-n,n)
    seconds=np.minimum(ends[row],(bins+1)*step)-np.maximum(starts[row],bins*step)
    # Epoch Thursday -> Monday-based weekday; 12 two-hour slots per day.
    slots=(bins+36)%84
    return row,slots,seconds


def conditional_profile(occ,users,cutoff):
    check_occurrences(occ,cutoff)
    strong=occ.loc[5*occ.seconds>=4*occ.duration]
    channels=sorted(set(int(c) for c in strong.channel));ci={c:i for i,c in enumerate(channels)}
    ui={u:n for n,u in enumerate(users)}
    row,slot,seconds=interval_bin_parts(strong.a.to_numpy(),strong.b.to_numpy())
    ch=strong.channel.to_numpy()[row]
    user=strong.user_id.to_numpy()[row]
    global_bins=np.zeros((len(channels),84),dtype=np.float64)
    np.add.at(global_bins,([ci[int(c)] for c in ch],slot),seconds)
    total=global_bins.sum(axis=1,keepdims=True)
    np.divide(global_bins,total,out=global_bins,where=total>0)
    mask=np.array([u in ui for u in user],dtype=bool)
    hist=sparse.csr_matrix((seconds[mask],
        ([ui[u] for u in user[mask]],[ci[int(c)]*84+int(s) for c,s in zip(ch[mask],slot[mask])])),
        shape=(len(users),len(channels)*84),dtype=np.float64)
    uc=strong[strong.user_id.isin(ui)].groupby(['user_id','channel']).agg(seconds=('seconds','sum'))
    r=[];cols=[];vals=[]
    for (u,c),record in uc.iterrows():
        channel=ci[int(c)];n=ui[u]
        observed=hist[n,channel*84:(channel+1)*84].toarray().ravel()
        # Fixed one-hour shrinkage to the TRAINING population channel-time prior.
        profile=(observed+3600*global_bins[channel])/(observed.sum()+3600)
        # Store only the correction to the population channel prior. Unseen
        # user/channel pairs back off to that prior, rather than treating absence
        # of history as an incompatible time preference.
        correction=profile-global_bins[channel]
        r.extend([n]*84);cols.extend(range(channel*84,(channel+1)*84));vals.extend(correction)
    profile=sparse.csr_matrix((vals,(r,cols)),shape=hist.shape)
    supported=hist.copy();supported.data=np.ones_like(supported.data)
    return profile,supported,ci,sparse.csr_matrix(global_bins.reshape(1,-1))


def schedule_vectors(frame,items,channel_index,mode,lower,upper):
    ii={int(i):n for n,i in enumerate(items)}
    known=frame[frame.item.isin(ii)].copy()
    known['start']=np.maximum(known.ss,lower);known['end']=np.minimum(known.ss+known.duration,upper)
    known=known[known.end>known.start]
    known['visible_seconds']=known.end-known.start
    f=known[known.channel.isin(channel_index)]
    row,slot,seconds=interval_bin_parts(f.start.to_numpy(),f.end.to_numpy())
    rows=[channel_index[int(c)]*84+int(s) for c,s in zip(f.channel.to_numpy()[row],slot)]
    cols=[ii[int(i)] for i in f.item.to_numpy()[row]]
    if mode=='airing_units':values=seconds/f.duration.to_numpy()[row]
    elif mode=='airtime_share':
        totals=known.groupby('item').visible_seconds.sum().to_dict()
        values=np.array([s/totals[int(i)] for s,i in zip(seconds,f.item.to_numpy()[row])])
    else:raise ValueError('Unknown schedule-vector mode')
    return sparse.csr_matrix((values,(rows,cols)),shape=(len(channel_index)*84,len(items)))


def schedule_habit_blocks(occ,past,future,users,items,ui,ii,cutoff,upper):
    profile,supported,channels,global_profile=conditional_profile(occ,users,cutoff)
    future_share=schedule_vectors(future,items,channels,'airtime_share',cutoff,upper)
    past_share=schedule_vectors(past,items,channels,'airtime_share',LOW,cutoff)
    population_mean=(global_profile@future_share).toarray().ravel()[ii]
    population_past=(global_profile@past_share).toarray().ravel()[ii]
    fmean=population_mean+(profile@future_share).toarray()[ui,ii]
    pmean=population_past+(profile@past_share).toarray()[ui,ii]
    support=(supported@future_share).toarray()[ui,ii]
    # These are mean compatibility features, NOT future volume boosts.
    habitual={'habit_channel_weektime_compatibility':84*fmean,
        'habit_previously_supported_airtime_share':support,
        'habit_channel_specific_lift':84*(fmean-population_mean)}
    future_units=schedule_vectors(future,items,channels,'airing_units',cutoff,upper)
    past_units=schedule_vectors(past,items,channels,'airing_units',LOW,cutoff)
    fmatched=((global_profile@future_units).toarray().ravel()[ii]+(profile@future_units).toarray()[ui,ii])*84/((upper-cutoff)/86400)
    pmatched=((global_profile@past_units).toarray().ravel()[ii]+(profile@past_units).toarray()[ui,ii])*84/((cutoff-LOW)/86400)
    fmatched=np.maximum(fmatched,0);pmatched=np.maximum(pmatched,0)
    # Compare rates with the SAME fixed 0.1/day prior across different horizons.
    drift=np.log((fmatched+.1)/(pmatched+.1))
    strong=occ.loc[5*occ.seconds>=4*occ.duration].groupby(['user_id','item']).size()
    pu=np.array(users,dtype=object)[ui];pi=np.array(items)[ii]
    n=np.array([strong.get((u,int(i)),0) for u,i in zip(pu,pi)],dtype=float)
    confidence=n/(n+3)
    changes={'drift_personal_supported_matched_rate_change':drift*confidence,
        'drift_completion_rate_x_matched_change':np.log1p(n/((cutoff-LOW)/86400))*drift,
        'drift_supported_mean_timing_change':84*(fmean-pmean)*confidence}
    return habitual,changes
