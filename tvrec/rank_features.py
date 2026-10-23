"""Small interpretable ranking features, all learnt from pre-cutoff history.

Only the known future programme schedule enters opportunity/time compatibility.
No target labels or future user observations are arguments to these functions.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from .models import matrix_from_features, score_features


def qualified_occurrences(frame,cutoff):
    if not ((frame.a<cutoff)&(frame.b<=cutoff)&(frame.b>frame.a)).all():
        raise ValueError('Feature history contains post-cutoff or invalid intervals')
    seconds=frame.b-frame.a
    f=frame.loc[2*seconds>=frame.duration].copy()
    f['seconds']=f.b-f.a
    f['fraction']=f.seconds/f.duration
    return f.sort_values(['seconds','b'],ascending=[False,False]).drop_duplicates(['user_id','sid'])


def repeat_variants(occurrences,users,items,cutoff):
    pairs=occurrences.groupby(['user_id','item'],as_index=False).agg(
        frequency=('sid','size'),last_second=('b','max'),mean_fraction=('fraction','mean'))
    strong=occurrences.loc[5*occurrences.seconds>=4*occurrences.duration].groupby(['user_id','item']).size()
    pairs['strong']=np.array([strong.get((u,int(i)),0) for u,i in zip(pairs.user_id,pairs.item)])
    pairs['score']=pairs.frequency.astype(float)
    frequency=score_features(pairs,users,items)
    confidence=pairs.copy()
    confidence['score']=confidence.frequency+confidence.strong
    recency=pairs.copy()
    days=(cutoff-recency.last_second)/86400
    recency['score']=recency.frequency*(.75+.25*np.exp(-np.log(2)*days/35))
    return {'frequency':frequency,'confidence':score_features(confidence,users,items),
            'recency':score_features(recency,users,items)},pairs


def time_slots(timestamps):
    timestamps=np.asarray(timestamps,dtype=np.int64)
    day=timestamps//86400
    weekday=(day+3)%7  # 1970-01-01 was Thursday; Monday=0
    return weekday*6+(timestamps%86400)//14400


def opportunities(train_airings,future_airings,items,train_days,future_days):
    """Smoothed, clipped schedule-volume ratio; not future viewing frequency."""
    past=train_airings.groupby('item').size().to_dict()
    future=future_airings.groupby('item').size().to_dict()
    expected=np.array([past.get(int(i),0)*future_days/train_days for i in items])
    actual=np.array([future.get(int(i),0) for i in items])
    ratio=(actual+4)/(expected+4)
    return np.sqrt(np.clip(ratio,.25,4)).astype(np.float32),past,future


def channel_scores(occurrences,users,items):
    """P(channel|user) * training P(programme|channel), with sqrt user counts."""
    channels=sorted(set(int(ch) for ch in occurrences.channel))
    ci={ch:i for i,ch in enumerate(channels)}
    ui={u:i for i,u in enumerate(users)}
    ii={int(i):n for n,i in enumerate(items)}
    uc=occurrences.groupby(['user_id','channel']).size()
    rows=[]; cols=[]; values=[]
    for (user,ch),count in uc.items():
        if user in ui:
            rows.append(ui[user]); cols.append(ci[ch]); values.append(np.sqrt(count))
    profile=sparse.csr_matrix((values,(rows,cols)),shape=(len(users),len(channels)))
    sums=np.asarray(profile.sum(axis=1)).ravel()
    profile=sparse.diags(np.divide(1,sums,out=np.zeros_like(sums),where=sums>0))@profile
    total=occurrences.groupby('channel').size().to_dict()
    cp=occurrences.groupby(['channel','item']).size()
    rows=[]; cols=[]; values=[]
    for (ch,item),count in cp.items():
        if int(item) in ii:
            rows.append(ci[ch]); cols.append(ii[int(item)]); values.append(count/total[ch])
    popularity=sparse.csr_matrix((values,(rows,cols)),shape=(len(channels),len(items)))
    return (profile@popularity).toarray().astype(np.float32)


def habit_scores(occurrences,future_airings,users,items):
    """Smoothed 42-slot weekly user habit vs target item's airing-midpoint mix."""
    ui={u:i for i,u in enumerate(users)}
    ii={int(i):n for n,i in enumerate(items)}
    profile=np.ones((len(users),42),dtype=np.float64)  # fixed Laplace prior
    slots=time_slots((occurrences.a.to_numpy()+occurrences.b.to_numpy())//2)
    for user,slot in zip(occurrences.user_id,slots):
        if user in ui:
            profile[ui[user],slot]+=1
    profile/=profile.sum(axis=1,keepdims=True)
    item_slots=np.zeros((42,len(items)),dtype=np.float64)
    slots=time_slots(future_airings.ss.to_numpy()+future_airings.duration.to_numpy()//2)
    for item,slot in zip(future_airings.item,slots):
        if int(item) in ii:
            item_slots[slot,ii[int(item)]]+=1
    totals=item_slots.sum(axis=0,keepdims=True)
    np.divide(item_slots,totals,out=item_slots,where=totals>0)
    return (profile@item_slots).astype(np.float32)
