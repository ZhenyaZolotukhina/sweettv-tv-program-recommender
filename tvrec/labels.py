"""Continuous-session audit using cached original views and disjoint schedule.

No cumulative visits across gaps or channel changes can create a positive.
"""
from collections import defaultdict
from itertools import groupby
import hashlib
import inspect
import json
import time

import numpy as np
import pandas as pd

from .core import match_disjoint
from .pipeline import LOW, SPLIT, TEST, save


def continuous_sessions(rows):
    """rows=(channel,start,stop). Deduplicate, union touching same-channel rows.

    Any time with >1 distinct active channel is ambiguous and excluded. With no
    device identifier, this conservative policy avoids claiming channel loyalty
    during overlapping different-channel logs. It does not guess which device wins.
    """
    raw = list(rows)
    unique = set((int(ch),int(a),int(b)) for ch,a,b in raw
                 if ch is not None and a is not None and b is not None and b>a)
    events = defaultdict(lambda: defaultdict(int))
    for channel, start, stop in unique:
        events[start][channel] += 1
        events[stop][channel] -= 1
    active = {}
    sessions = []
    previous = None
    ambiguous = 0
    for timestamp in sorted(events):
        if previous is not None and timestamp>previous:
            if len(active)==1:
                channel = next(iter(active))
                if sessions and sessions[-1][0]==channel and sessions[-1][2]==previous:
                    sessions[-1] = (channel,sessions[-1][1],timestamp)
                else:
                    sessions.append((channel,previous,timestamp))
            elif len(active)>1:
                ambiguous += timestamp-previous
        for channel, delta in events[timestamp].items():
            count = active.get(channel,0)+delta
            if count:
                active[channel] = count
            else:
                active.pop(channel,None)
        previous = timestamp
    return sessions, {'duplicate_intervals':len(raw)-len(unique),
                      'ambiguous_user_seconds':ambiguous}


def qualifies(seconds, duration, strict=False):
    # Integer arithmetic: 80% exactly is never mistaken for strictly >80%.
    return duration>0 and (5*seconds>4*duration if strict else 5*seconds>=4*duration)


def cached_sessions(c, cache, out):
    """Add new tables in stage2 DB, never rewrite or rescan original CSVs."""
    signature = hashlib.sha256((inspect.getsource(continuous_sessions)+
                                inspect.getsource(match_disjoint)).encode()).hexdigest()
    marker = cache/'stage2_sessions_manifest.json'
    base_manifest = json.loads((cache/'manifest.json').read_text())
    expected = {'algorithm_sha256':signature,'original_manifest':base_manifest}
    if marker.exists() and json.loads(marker.read_text())==expected:
        if c.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='continuous_events' AND table_catalog=current_database()").fetchone()[0]:
            print('Reusing audited continuous sessions and events.',flush=True)
            return
    start = time.perf_counter()
    c.execute('DROP TABLE IF EXISTS sessions')
    c.execute('DROP TABLE IF EXISTS continuous_events')
    c.execute('CREATE TABLE sessions(session_id BIGINT,user_id VARCHAR,channel INTEGER,a BIGINT,b BIGINT)')
    reader = c.cursor()
    reader.execute('SELECT user_id,channel,vs,ve FROM base.views WHERE ve>vs AND user_id IS NOT NULL ORDER BY user_id,vs,vid')
    def stream():
        while batch:=reader.fetchmany(100000):
            yield from batch
    buffer = []
    counts = defaultdict(int)
    session_id = 0
    for user, rows in groupby(stream(),key=lambda r:r[0]):
        segments, stats = continuous_sessions((ch,a,b) for _,ch,a,b in rows)
        for k,v in stats.items():
            counts[k] += v
        for channel,a,b in segments:
            session_id += 1
            buffer.append((session_id,user,channel,a,b))
        if len(buffer)>100000:
            frame=pd.DataFrame(buffer,columns=['session_id','user_id','channel','a','b'])
            c.register('session_frame',frame)
            c.execute('INSERT INTO sessions SELECT * FROM session_frame')
            c.unregister('session_frame')
            buffer=[]
    if buffer:
        frame=pd.DataFrame(buffer,columns=['session_id','user_id','channel','a','b'])
        c.register('session_frame',frame)
        c.execute('INSERT INTO sessions SELECT * FROM session_frame')
        c.unregister('session_frame')
    reader.close()
    c.execute('CREATE TABLE continuous_events(session_id BIGINT,user_id VARCHAR,sid BIGINT,channel INTEGER,a BIGINT,b BIGINT)')
    # Reuse the cached, conflict-resolved schedule. Each segment is a CONTIGUOUS
    # run; separated fragments of an interrupted airing are never summed here.
    for channel, in c.execute('SELECT DISTINCT channel FROM sessions ORDER BY channel').fetchall():
        sd=c.execute('SELECT a,b,sid FROM base.segments WHERE channel=? ORDER BY a',[channel]).fetchnumpy()
        if not len(sd['a']):
            continue
        vd=c.execute('SELECT session_id,user_id,a,b FROM sessions WHERE channel=? ORDER BY a',[channel]).fetchnumpy()
        vi,si,first,last=match_disjoint(vd['a'],vd['b'],sd['a'],sd['b'],LOW,TEST)
        frame=pd.DataFrame({'session_id':vd['session_id'][vi],'user_id':vd['user_id'][vi],
                            'sid':sd['sid'][si],'channel':channel,'a':first,'b':last})
        c.register('event_frame',frame)
        c.execute('INSERT INTO continuous_events SELECT * FROM event_frame')
        c.unregister('event_frame')
    c.execute('CHECKPOINT')
    counts['sessions']=session_id
    counts['continuous_schedule_runs']=c.execute('SELECT count(*) FROM continuous_events').fetchone()[0]
    counts['seconds']=time.perf_counter()-start
    save(out/'session_accounting.json',dict(counts))
    save(marker,expected)
    print('Session audit:',dict(counts),flush=True)


def event_frame(c, lower, upper):
    """Clip before any coverage threshold or model feature is derived."""
    return c.execute('''SELECT e.session_id,e.user_id,e.sid,e.channel,
        greatest(e.a,?) a,least(e.b,?) b,a.duration,a.item,
        s.b-s.a session_seconds
        FROM continuous_events e JOIN base.airings a USING(sid) JOIN sessions s USING(session_id)
        WHERE e.b>? AND e.a<? AND a.item>0 AND a.duration>0''',
                     [lower,upper,lower,upper]).fetchdf()


def target_from_events(frame, strict=False, top5=True, long_limit=None, tie='id'):
    frame=frame.copy()
    frame['seconds']=frame.b-frame.a
    mask=5*frame.seconds>4*frame.duration if strict else 5*frame.seconds>=4*frame.duration
    if long_limit is not None:
        mask &= frame.session_seconds<=long_limit
    # Each distinct airing counts at most once, regardless of duplicate logs or runs.
    positives=frame.loc[mask].drop_duplicates(['user_id','sid'])
    pairs=positives.groupby(['user_id','item'],as_index=False).agg(frequency=('sid','size'),last=('b','max'))
    order=['user_id','frequency','item'] if tie=='id' else ['user_id','frequency','last','item']
    ascending=[True,False,True] if tie=='id' else [True,False,False,True]
    pairs=pairs.sort_values(order,ascending=ascending)
    selected=pairs.groupby('user_id',sort=False).head(5) if top5 else pairs
    truth=defaultdict(set)
    for user,item in zip(selected.user_id,selected.item):
        truth[user].add(int(item))
    return dict(truth),pairs,selected


def training_features(frame, minimum=.5, strength='frequency', half_life=0, cutoff=SPLIT, seconds_cut=None):
    """MODEL INPUTS: independent of target's >=80% and Top-5 rules."""
    f=frame.copy()
    f['seconds']=f.b-f.a
    f['fraction']=f.seconds/f.duration
    mask=f.seconds>=seconds_cut if seconds_cut is not None else f.fraction>=minimum
    f=f.loc[mask].copy()
    # One qualifying occurrence per airing, no multiple credit for overlapping logs.
    f=f.sort_values(['seconds','b'],ascending=[False,False]).drop_duplicates(['user_id','sid'])
    value={'frequency':np.ones(len(f)), 'fraction':f.fraction.to_numpy(), 'seconds':f.seconds.to_numpy()}[strength]
    if half_life:
        value=value*np.exp(-np.log(2)*(cutoff-f.b.to_numpy())/(half_life*86400))
    f['score']=value
    return f.groupby(['user_id','item'],as_index=False).agg(score=('score','sum'),frequency=('sid','size'),last_second=('b','max'))
