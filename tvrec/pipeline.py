"""Reproducible local audit, interval join, baseline selection and readiness CSV.

Run: python -m tvrec.pipeline all --data sweettv-tv-program-recommender
"""
import argparse
import hashlib
import html
import json
import platform
import re
import resource
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .core import ap_at_5, disjoint_airings, map_at_5, match_disjoint, write_submission

VERSION = 1
FILES = ['dataset11-30.csv', 'export_arh_11-20-final.csv',
         'export_arh_21-30-final.csv', 'export_arh_31-42-final.csv', 'submission.csv']
LOW = int(datetime(2020, 3, 9).timestamp())
SPLIT = int(datetime(2020, 5, 18).timestamp())
TEST = int(datetime(2020, 7, 27).timestamp())
END = int(datetime(2020, 10, 19).timestamp())
# Treat original timezone-naive clock values identically, independent of host TZ.
LOW, SPLIT, TEST, END = [int(np.datetime64(x, 's').astype('int64')) for x in
                       ['2020-03-09', '2020-05-18', '2020-07-27', '2020-10-19']]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + '\n')


def records(c, sql):
    r = c.execute(sql)
    names = [x[0] for x in r.description]
    return [dict(zip(names, row)) for row in r.fetchall()]


def fingerprint(data):
    out = {}
    for name in FILES:
        p = data/name
        digest = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(4*1024*1024), b''):
                digest.update(chunk)
        out[name] = {'bytes': p.stat().st_size, 'sha256': digest.hexdigest()}
    return out


def prepare(data, cache, out, channel_source=None):
    started = time.perf_counter()
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    marker = cache/'manifest.json'
    db = cache/'data.duckdb'
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    stats = {n: {'bytes':(data/n).stat().st_size,'mtime_ns':(data/n).stat().st_mtime_ns} for n in FILES}
    hashes = previous['files'] if previous.get('stats')==stats else fingerprint(data)
    manifest = {'version': VERSION, 'files': hashes, 'stats': stats}
    if previous.get('version')==VERSION and previous.get('files')==hashes and db.exists():
        print('Using fingerprint-verified cached data.', flush=True)
        save(marker,manifest)
        return duckdb.connect(str(db))
    if db.exists():
        db.unlink()
    c = duckdb.connect(str(db))
    c.execute("SET threads=4; SET memory_limit='3GB'; SET preserve_insertion_order=true")
    audit = {'source_files': manifest['files'], 'archives': 'No archives supplied; five extracted CSV files.',
             'count_provenance': 'SQL statements and source row IDs in tvrec/pipeline.py; header excluded.',
             'encoding': 'UTF-8: all files parsed strictly by DuckDB; no ignored errors.',
             'schemas': {}}
    for i, name in enumerate(FILES):
        table = ['raw_views', 'raw_s11', 'raw_s21', 'raw_s31', 'template'][i]
        c.execute(f"CREATE TABLE {table} AS SELECT row_number() OVER () AS source_row, * FROM read_csv(?, all_varchar=true)", [str(data/name)])
        audit['schemas'][name] = [r[0] for r in c.execute(f'DESCRIBE {table}').fetchall()][1:]
    c.execute("""CREATE TABLE views AS SELECT source_row AS vid, user_id,
        try_cast(vsetv_id AS INTEGER) channel, epoch(try_cast(start_time AS TIMESTAMP))::BIGINT vs,
        epoch(try_cast(stop_time AS TIMESTAMP))::BIGINT ve, try_cast(duraton AS BIGINT) duraton
        FROM raw_views""")
    c.execute("""CREATE TABLE schedule AS SELECT row_number() OVER () sid, * FROM (
        SELECT 11 AS period, * FROM raw_s11 UNION ALL SELECT 21, * FROM raw_s21 UNION ALL SELECT 31, * FROM raw_s31)""")
    c.execute("""CREATE TABLE airings AS SELECT sid, period, source_row, cast(channel_id AS INTEGER) channel,
        channel_title, epoch(try_strptime(start_time, '%d.%m.%Y %H:%M:%S'))::BIGINT ss,
        try_cast(duration AS BIGINT) duration, nullif(try_cast(tv_show_id AS BIGINT),0) item,
        tv_show_title title, tv_show_category category FROM schedule""")
    c.execute('ALTER TABLE airings ADD COLUMN se BIGINT')
    c.execute('UPDATE airings SET se=ss+duration')
    audit['views'] = records(c, """SELECT count(*) AS "rows", count(DISTINCT user_id) users,
        count(DISTINCT channel) channels, min(vs) start_epoch, max(ve) stop_epoch,
        min(duraton) min_duraton, quantile_cont(duraton,[0.5,0.9,0.99]) duraton_quantiles,
        max(duraton) max_duraton, count(*) FILTER(WHERE duraton>86400) over_24h,
        count(*) FILTER(WHERE duraton != ve-vs) duration_mismatches,
        count(*) FILTER(WHERE vs IS NULL OR ve IS NULL OR ve<=vs OR channel IS NULL OR user_id IS NULL) invalid,
        count(*)-count(DISTINCT (user_id,channel,vs,ve,duraton)) duplicate_rows,
        count(*) FILTER(WHERE try_cast(user_id AS UBIGINT)>9223372036854775807) above_signed_int64 FROM views""")[0]
    audit['schedules'] = records(c, """SELECT period, count(*) AS "rows", count(DISTINCT channel) channels,
        count(DISTINCT item) labelled_programmes, count(*) FILTER(WHERE item IS NOT NULL) labelled_rows,
        count(*) FILTER(WHERE item IS NULL) unlabelled_rows,
        count(*) FILTER(WHERE ss IS NULL OR duration IS NULL OR duration<=0) invalid_intervals,
        min(ss) start_epoch, max(ss) last_start_epoch, max(se) end_epoch,
        quantile_cont(duration,[0.5,0.9,0.99]) duration_quantiles, max(duration) max_duration
        FROM airings GROUP BY period ORDER BY period""")
    audit['schedule_missing_vs_zero'] = records(c, """SELECT period,
        count(*) FILTER(WHERE tv_show_id IS NULL) missing_ids,
        count(*) FILTER(WHERE tv_show_id='0') zero_ids,
        count(*) FILTER(WHERE tv_show_id IS NOT NULL AND try_cast(tv_show_id AS BIGINT) IS NULL) malformed_ids
        FROM schedule GROUP BY period ORDER BY period""")
    audit['schedule_duplicate_rows'] = records(c, """SELECT period, count(*)-count(DISTINCT
        (channel,ss,duration,item,title,category)) duplicate_core_rows FROM airings GROUP BY period""")
    audit['repeated_titles'] = records(c, """SELECT count(*) titles_with_multiple_labelled_ids FROM
        (SELECT title FROM airings WHERE item IS NOT NULL GROUP BY title HAVING count(DISTINCT item)>1)""")[0]
    audit['repeated_ids'] = records(c, """SELECT count(*) ids_with_multiple_titles FROM
        (SELECT item FROM airings WHERE item IS NOT NULL GROUP BY item HAVING count(DISTINCT title)>1)""")[0]
    audit['template'] = records(c, 'SELECT count(*) AS "rows",count(DISTINCT user_id) users FROM template')[0]
    assert audit['template'] == {'rows':1260,'users':1260}
    # Full public VseTV channel directory, decoded by the text proxy. Cache source.
    source = out/'sources/vsetv_channels.txt'
    source.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://r.jina.ai/http://www.vsetv.com/channels.html'
    if channel_source:
        source.write_bytes(Path(channel_source).read_bytes())
    if not source.exists():
        source.write_bytes(urllib.request.urlopen(url, timeout=120).read())
    text = source.read_text()
    external = {}
    # Jina Markdown: logo (possibly 1px), title, separator, arrow, schedule URL.
    for match in re.finditer(r'!\[Image \d+\]\(http://www\.vsetv\.com/pic/(?:channel_logos/\d+|1px)\.gif\)([^!\n]+?)\s*!\[Image \d+\]\(http://www\.vsetv\.com/pic/1px\.gif\).*?schedule_channel_(\d+)_week\.html', text):
        title, channel = match.groups()
        external[int(channel)] = html.unescape(title).strip()
    if len(external) < 100:
        raise RuntimeError('External channel directory parsing failed: refuse unverified numeric join')
    def norm(x):
        return re.sub(r'[^a-zа-яёіїєґ0-9+]', '', x.lower())
    channel_rows = records(c, 'SELECT DISTINCT channel,channel_title FROM airings ORDER BY channel')
    exact = [r for r in channel_rows if norm(r['channel_title']) == norm(external.get(r['channel'], ''))]
    # Many independent ID-title matches verify the namespace, rather than mere numeric overlap.
    if len(exact) < 40:
        raise RuntimeError('Insufficient independent ID/title matches to verify VseTV namespace')
    mapping = []
    for r in channel_rows:
        r.update(vsetv_id=r['channel'], external_title=external.get(r['channel']),
                 evidence='exact_normalized_title' if r in exact else 'namespace_identity_not_individually_confirmed')
        mapping.append(r)
    pd.DataFrame(mapping).to_csv(out/'channel_mapping.csv', index=False)
    audit['channel_mapping'] = {'source':url,'original_source':'http://www.vsetv.com/channels.html',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'parsed_external_channels':len(external),'exact_id_title_matches':len(exact),
        'policy':'Identity namespace verified by >=40 independent ID/title checks; all schedule IDs mapped identically. Historical renamed/absent channels not individually verified; see CSV.',
        'view_channels_without_schedule':records(c, 'SELECT channel,count(*) AS "views" FROM views WHERE channel NOT IN (SELECT channel FROM airings) GROUP BY channel ORDER BY channel')}
    # Deterministic interval sweep per channel. No pairwise schedule join.
    segments = []
    conflict_rows = 0
    for (channel,) in c.execute('SELECT DISTINCT channel FROM airings ORDER BY channel').fetchall():
        rows = c.execute('SELECT ss,se,sid FROM airings WHERE channel=? AND ss IS NOT NULL AND se>ss ORDER BY ss,sid', [channel]).fetchall()
        maximum = -1
        for start, stop, sid in rows:
            conflict_rows += start < maximum
            maximum = max(maximum, stop)
        segments.extend((channel,a,b,sid) for a,b,sid in disjoint_airings(rows))
    seg_df = pd.DataFrame(segments, columns=['channel','a','b','sid'])
    c.register('seg_df',seg_df)
    c.execute('CREATE TABLE segments AS SELECT * FROM seg_df')
    c.unregister('seg_df')
    del segments, seg_df
    audit['overlap_resolution'] = {'rows_overlapping_prior_running_end':conflict_rows,
        'rule':'latest schedule start wins; ties lowest global source row ID; zero/missing IDs receive no preference; half-open intervals',
        'disjoint_segments':c.execute('SELECT count(*) FROM segments').fetchone()[0]}
    c.execute('CREATE TABLE matches(vid BIGINT,sid BIGINT,phase VARCHAR,seconds BIGINT,first_second BIGINT,last_second BIGINT)')
    match_started = time.perf_counter()
    counts = defaultdict(int)
    for (channel,) in c.execute('SELECT DISTINCT channel FROM views ORDER BY channel').fetchall():
        sd = c.execute('SELECT a,b,sid FROM segments WHERE channel=? ORDER BY a',[channel]).fetchnumpy()
        if len(sd['a']) == 0:
            continue
        vd = c.execute('SELECT vid,vs,ve FROM views WHERE channel=? AND ve>vs ORDER BY vs',[channel]).fetchnumpy()
        for label, lo, hi in [('train',LOW,SPLIT),('validation',SPLIT,TEST)]:
            idx, si, first, last = match_disjoint(vd['vs'],vd['ve'],sd['a'],sd['b'],lo,hi)
            if not len(idx):
                continue
            df = pd.DataFrame({'vid':vd['vid'][idx],'sid':sd['sid'][si],
                               'phase':label,'seconds':last-first,'first_second':first,'last_second':last})
            assert (df.seconds>0).all()
            c.register('match_df',df)
            c.execute('INSERT INTO matches SELECT * FROM match_df')
            c.unregister('match_df')
            counts[label] += len(df)
    # Aggregate fragments of an airing interrupted by a conflicting schedule row.
    c.execute('''CREATE TABLE interactions AS SELECT m.vid,m.sid,m.phase,v.user_id,a.item,a.duration,
        sum(m.seconds)::BIGINT AS seconds, sum(m.seconds)::DOUBLE/a.duration AS fraction,
        min(m.first_second) first_second,max(m.last_second) last_second
        FROM matches m JOIN views v USING(vid) JOIN airings a USING(sid)
        WHERE a.item IS NOT NULL AND a.item>0 AND a.duration>0
        GROUP BY m.vid,m.sid,m.phase,v.user_id,a.item,a.duration''')
    # No per-view second may be assigned twice, even across zero/nonzero airings.
    assert c.execute('''SELECT count(*) FROM (SELECT vid,sum(seconds) s FROM matches GROUP BY vid) m
        JOIN views v USING(vid) WHERE s>v.ve-v.vs''').fetchone()[0] == 0
    assert c.execute("SELECT count(*) FROM interactions WHERE phase='train' AND last_second>?",[SPLIT]).fetchone()[0] == 0
    assert c.execute("SELECT count(*) FROM interactions WHERE phase='validation' AND (first_second<? OR last_second>?)",[SPLIT,TEST]).fetchone()[0] == 0
    audit['join'] = records(c, '''SELECT (SELECT count(*) FROM views) total_views,
        (SELECT count(DISTINCT vid) FROM matches) matched_schedule_views,
        (SELECT count(DISTINCT vid) FROM interactions) matched_labelled_views,
        (SELECT count(DISTINCT user_id) FROM interactions) usable_users,
        (SELECT count(DISTINCT item) FROM interactions) usable_programmes,
        (SELECT count(*) FROM interactions) interaction_rows''')[0]
    j = audit['join']
    j['no_valid_programme_views'] = j['total_views']-j['matched_labelled_views']
    j['schedule_match_percent'] = 100*j['matched_schedule_views']/j['total_views']
    j['labelled_match_percent'] = 100*j['matched_labelled_views']/j['total_views']
    j['join_seconds'] = time.perf_counter()-match_started
    audit['interaction_windows'] = records(c, '''SELECT phase,count(*) interactions,count(DISTINCT vid) AS "views",
        count(DISTINCT user_id) users,count(DISTINCT item) programmes,sum(seconds) AS seconds
        FROM interactions GROUP BY phase''')
    audit['training_overlap_quantiles'] = records(c, """SELECT quantile_cont(seconds,[0.1,0.25,0.5,0.75,0.9]) AS seconds,
        quantile_cont(fraction,[0.1,0.25,0.5,0.75,0.9]) fractions FROM interactions WHERE phase='train'""")[0]
    audit['availability'] = records(c, '''SELECT period,count(DISTINCT item) candidates,
        count(DISTINCT item) FILTER(WHERE item IN (SELECT item FROM interactions WHERE phase='train')) seen_train,
        count(DISTINCT item) FILTER(WHERE item IN (SELECT item FROM interactions)) seen_all_views
        FROM airings WHERE item>0 AND duration>0 GROUP BY period ORDER BY period''')
    audit['raw_view_windows'] = records(c, f'''SELECT CASE WHEN vs<{SPLIT} THEN 'train' WHEN vs<{TEST} THEN 'validation' ELSE 'outside' END phase,
        count(*) AS "views" FROM views GROUP BY 1''')
    audit['time_split'] = {'train':'[2020-03-09 00:00:00,2020-05-18 00:00:00)',
        'validation':'[2020-05-18 00:00:00,2020-07-27 00:00:00)',
        'test':'[2020-07-27 00:00:00,2020-10-19 00:00:00)',
        'timezone':'original naive timestamps; no inferred UTC conversion'}
    audit['prepare_seconds'] = time.perf_counter()-started
    save(out/'dataset_accounting.json',audit)
    c.execute('CHECKPOINT')
    save(marker,manifest)
    print(json.dumps({'prepare_seconds':audit['prepare_seconds'],'join':j,'channel_mapping':audit['channel_mapping']},default=str),flush=True)
    return c


def candidate_set(c, period):
    lo, hi = (SPLIT,TEST) if period==21 else (TEST,END)
    return {r[0] for r in c.execute('SELECT DISTINCT item FROM airings WHERE period=? AND item>0 AND duration>0 AND ss<? AND se>?',[period,hi,lo]).fetchall()}


def features(c, variant, threshold, half_life, final=False):
    cutoff = TEST if final else SPLIT
    condition = 'TRUE' if final else "phase='train'"
    strength = 'seconds' if variant=='seconds' else 'fraction'
    # Filter on overlap time or programme fraction, NOT negative preference labels.
    predicate = f'{strength}>={float(threshold)}'
    decay = '1.0' if half_life==0 else f'exp(-ln(2)*({cutoff}-last_second)/({half_life}*86400.0))'
    return c.execute(f'''SELECT user_id,item,sum({strength}*{decay}) score,count(*) frequency,
        max(last_second) last_second FROM interactions WHERE {condition} AND {predicate}
        GROUP BY user_id,item ORDER BY user_id,item''').fetchdf()


def rank_baseline(frame, users, candidates, method, alpha=0, min_history=0, signal='strength', fallback=None):
    frame = frame[frame.item.isin(candidates)]
    totals = frame.groupby('item').score.sum().to_dict()
    # Deterministic padding for unseen programmes: training score 0, item ID tie-break.
    popularity = sorted(candidates, key=lambda i:(-totals.get(i,0),i))
    popmax = max(totals.values(),default=1) or 1
    pop = {i:totals.get(i,0)/popmax for i in candidates}
    fallback = popularity if fallback is None else fallback
    histories = {}
    for u,g in frame.groupby('user_id',sort=False):
        if signal=='frequency':
            scores = dict(zip(g.item,g.frequency.astype(float)))
        elif signal=='recency':
            scores = dict(zip(g.item,g.last_second.astype(float)))
            # Monotone rank transform avoids epoch-scale normalization effects.
            order = sorted(scores,key=lambda i:(-scores[i],i))
            scores = {i:1/(r+1) for r,i in enumerate(order)}
        else:
            scores = dict(zip(g.item,g.score))
        histories[u] = scores
    predictions = {}
    for user in users:
        scores = histories.get(user,{})
        if method=='popularity':
            predictions[user] = popularity[:5]
            continue
        if len(scores)<min_history or not scores:
            predictions[user] = fallback[:5]
            continue
        if method=='repeat':
            ordered = sorted(scores,key=lambda i:(-scores[i],i))
            predictions[user] = (ordered+[i for i in fallback[:10] if i not in scores])[:5]
        else:
            maximum = max(scores.values()) or 1
            pool = set(scores)|set(popularity[:10])
            ordered = sorted(pool,key=lambda i:(-(scores.get(i,0)/maximum+alpha*pop[i]),i))
            predictions[user] = ordered[:5]
    return predictions


def evaluate(c, out):
    start = time.perf_counter()
    candidates = candidate_set(c,21)
    template = [r[0] for r in c.execute('SELECT user_id FROM template ORDER BY source_row').fetchall()]
    # Fixed users and labels for every configuration. Weak glances do not create dislikes.
    # Common primary truth: >=60 seconds assigned to one airing; sensitivity at 1s/300s.
    truth_rows = c.execute("SELECT user_id,item,sum(seconds) FROM interactions WHERE phase='validation' AND seconds>=60 GROUP BY user_id,item").fetchall()
    truth = defaultdict(set)
    for user,item,_ in truth_rows:
        if item in candidates:
            truth[user].add(item)
    eligible = [u for u in template if truth.get(u)]
    # Fixed cold group under primary >=60-second training definition.
    warm_set = {r[0] for r in c.execute("SELECT DISTINCT user_id FROM interactions WHERE phase='train' AND seconds>=60").fetchall()}
    warm = [u for u in eligible if u in warm_set]
    cold = [u for u in eligible if u not in warm_set]
    quantiles = c.execute("SELECT quantile_cont(seconds,[0.25,0.5]),quantile_cont(fraction,[0.25,0.5]) FROM interactions WHERE phase='train'").fetchone()
    # Training-derived quartile and median, plus interpretable fixed floors.
    thresholds = {'seconds':sorted(set([60.,300.,float(quantiles[0][0]),float(quantiles[0][1])])),
                  'fraction':sorted(set([.05,.1,float(quantiles[1][0]),float(quantiles[1][1])]))}
    rows=[]
    best = {}
    best_predictions = {}
    # Select A first, then use exactly its ranked list as every model's cold/padding fallback.
    fallback = None
    for pass_name in ['global','personalized']:
        for variant, cuts in thresholds.items():
            for threshold in cuts:
                for half_life in [0,14,35]:
                    t0=time.perf_counter()
                    frame=features(c,variant,threshold,half_life)
                    feature_time=time.perf_counter()-t0
                    settings=[('repeat',0,0,'strength'),
                              ('repeat',0,0,'frequency'),('repeat',0,0,'recency')]
                    settings += [('repeat_popularity',a,n,'strength') for a in [.1,.5,1.] for n in [0,3]]
                    if pass_name=='global':
                        settings=[('popularity',0,0,'strength')]
                    for method,alpha,minimum,signal in settings:
                        t0=time.perf_counter()
                        pred=rank_baseline(frame,template,candidates,method,alpha,minimum,signal,fallback)
                        row={'method':method,'variant':variant,'threshold':threshold,'half_life_days':half_life,
                             'alpha':alpha,'min_history':minimum,'signal':signal,
                             'map5':map_at_5(truth,pred,eligible),'warm_map5':map_at_5(truth,pred,warm),
                             'cold_map5':map_at_5(truth,pred,cold),
                             'all_template_map5_empty_zero':map_at_5(truth,pred,template),
                             'eligible_users':len(eligible),'warm_users':len(warm),'cold_users':len(cold),
                             'feature_seconds':feature_time,'rank_evaluate_seconds':time.perf_counter()-t0}
                        row['runtime_seconds']=row['feature_seconds']+row['rank_evaluate_seconds']
                        rows.append(row)
                        if method not in best or row['map5']>best[method]['map5']:
                            best[method]=row
                            best_predictions[method]=pred
        if pass_name=='global':
            reference=best['popularity']
            reference_frame=features(c,reference['variant'],reference['threshold'],reference['half_life_days'])
            # Full candidate order permits deterministic distinct padding.
            totals=reference_frame[reference_frame.item.isin(candidates)].groupby('item').score.sum().to_dict()
            fallback=sorted(candidates,key=lambda i:(-totals.get(i,0),i))
    df=pd.DataFrame(rows)
    df.to_csv(out/'validation_grid.csv',index=False)
    winner=max(best.values(),key=lambda r:r['map5'])
    chosen=winner['method']
    sensitivity={}
    for seconds in [1,60,300]:
        tt=defaultdict(set)
        for u,i in c.execute("SELECT DISTINCT user_id,item FROM interactions WHERE phase='validation' AND seconds>=?",[seconds]).fetchall():
            if i in candidates:
                tt[u].add(i)
        uu=[u for u in template if tt[u]]
        sensitivity[str(seconds)]={'eligible_users':len(uu), 'map5':map_at_5(tt,best_predictions[chosen],uu)}
    recall_n=sum(len(truth[u]&set(c.execute("SELECT DISTINCT item FROM interactions WHERE user_id=? AND phase='train' AND seconds>=60",[u]).fetchnumpy()['item'])) for u in eligible)
    recall_d=sum(len(truth[u]) for u in eligible)
    stats={'eligible_policy':'official template users with >=1 available validation item at >=60s/airing; common across all methods',
           'empty_truth_policy':'excluded from primary MAP; alternative all-1260 empty=0 reported in grid',
           'metric':'AP=sum(precision@hit)/min(5,number of unique truth items); duplicate predictions consume rank but no repeated credit',
           'competition_metric_verification':'Instructor Markdown confirms MAP@K and K=5; Kaggle formula/empty-user convention unavailable behind JS/auth; local convention may differ.',
           'repeat_policy':'repeats allowed for every baseline; same schedule availability filter; popularity pads all methods',
           'fallback_policy':'all personalized configurations use the same validation-selected A ranking for cold users and padding; A is selected first',
           'truth_seconds':60,'threshold_grid':thresholds,'half_life_grid':[0,14,35],
           'template_users':len(template),'eligible_users':len(eligible),'empty_truth_users':len(template)-len(eligible),
           'warm_users':len(warm),'cold_users':len(cold),'validation_candidates':len(candidates),
           'repeat_truth_item_fraction':recall_n/recall_d if recall_d else None,
           'truth_sensitivity':sensitivity,'best_by_method':best,'selected':winner,
           'configurations':len(rows),'selection_seconds':time.perf_counter()-start}
    save(out/'validation_results.json',stats)
    # Evidence is private local material, excluded from Git.
    private=out/'private'
    private.mkdir(exist_ok=True)
    heavy=max(eligible,key=lambda u:len(truth[u]))
    light=min((u for u in eligible if u!=heavy),key=lambda u:len(truth[u]))
    examples=[]
    for user,reason in [(heavy,'largest distinct validation truth history'),(light,'smallest distinct validation truth history')]:
        history=records(c, f"SELECT phase,item,seconds,fraction,first_second,last_second FROM interactions WHERE user_id='{user}' ORDER BY first_second,item")
        predictions={method:pred[user] for method,pred in best_predictions.items()}
        ids=set(i for p in predictions.values() for i in p)
        metadata=records(c,'SELECT item,first(title ORDER BY ss) title,first(category ORDER BY ss) category FROM airings WHERE item IN ('+','.join(map(str,ids))+') GROUP BY item')
        examples.append({'user_id':user,'selection_reason':reason,'history':history,'predictions':predictions,
                         'metadata':metadata,'truth':sorted(truth[user]),
                         'ap5':{m:ap_at_5(truth[user],p) for m,p in predictions.items()}})
    save(private/'user_examples.json',examples)
    example=records(c,"""SELECT v.vid,v.user_id,v.channel,v.vs,v.ve,a.sid,a.source_row,a.period,
        a.ss,a.se,a.duration,a.item,a.title,i.seconds,i.fraction,i.first_second,i.last_second
        FROM interactions i JOIN views v USING(vid) JOIN airings a USING(sid)
        WHERE i.phase='train' AND i.seconds>=300 AND i.fraction<1 ORDER BY v.vid,a.sid LIMIT 1""")[0]
    for key in ['vs','ve','ss','se','first_second','last_second']:
        example[key+'_timestamp']=str(np.datetime64(example[key],'s'))
    save(private/'interval_example.json',example)
    save(out/'ap5_hand_calculation.json',{'truth':[10,20,30], 'predictions':[10,99,20,88,30],
         'hit_ranks':[1,3,5], 'arithmetic':'(1/1 + 2/3 + 3/5) / min(5,3) = 34/45',
         'ap5':ap_at_5([10,20,30],[10,99,20,88,30])})
    t0=time.perf_counter()
    final_frame=features(c,winner['variant'],winner['threshold'],winner['half_life_days'],final=True)
    test_candidates=candidate_set(c,31)
    reference=best['popularity']
    reference_frame=features(c,reference['variant'],reference['threshold'],reference['half_life_days'],final=True)
    totals=reference_frame[reference_frame.item.isin(test_candidates)].groupby('item').score.sum().to_dict()
    final_fallback=sorted(test_candidates,key=lambda i:(-totals.get(i,0),i))
    final_pred=rank_baseline(final_frame,template,test_candidates,chosen,winner['alpha'],winner['min_history'],winner['signal'],final_fallback)
    path=private/'preliminary_submission.csv'
    write_submission(path,template,final_pred,test_candidates)
    save(out/'submission_checks.json',{'path':str(path),'rows':len(template),'columns':['user_id','tv_show_id'],
         'exact_template_order_and_string_ids':True,'five_distinct_positive_available_ids_each':True,
         'validated_before_atomic_save_and_roundtrip':True,'test_candidates':len(test_candidates),
         'test_candidates_seen_refit':int(final_frame[final_frame.item.isin(test_candidates)].item.nunique()),
         'template_users_without_refit_interactions':len(set(template)-set(final_frame.user_id)),
         'final_refit_and_save_seconds':time.perf_counter()-t0,
         'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'submitted_to_kaggle':False})
    print(json.dumps(stats,default=str),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','evaluate','diagnose','all'])
    parser.add_argument('--data',type=Path,default=Path('sweettv-tv-program-recommender'))
    parser.add_argument('--cache',type=Path,default=Path('cache'))
    parser.add_argument('--out',type=Path,default=Path('artifacts'))
    parser.add_argument('--channel-source',type=Path,help='Saved Jina VseTV channel-directory response, for offline reproducibility')
    args=parser.parse_args()
    started=time.perf_counter()
    c=prepare(args.data,args.cache,args.out,args.channel_source)
    if args.stage in ['evaluate','all']:
        evaluate(c,args.out)
    from .diagnostics import diagnostics
    diagnostics(c,args.out)
    c.close()
    execution={'python':sys.version,'platform':platform.platform(),
         'duckdb':duckdb.__version__,'numpy':np.__version__,'pandas':pd.__version__,
         'elapsed_seconds':time.perf_counter()-started,
         'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'memory_units':'ru_maxrss is bytes on macOS; KiB on Linux',
         'seed':'deterministic; no stochastic model','code_author':'AI assistant; no claim of student verification',
         'stage':args.stage,'recorded_at_local':datetime.now().isoformat(),
         'pipeline_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    history_path=args.out/'execution_history.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    if not history and (args.out/'execution.json').exists():
        history.append(json.loads((args.out/'execution.json').read_text()))
    history.append(execution)
    save(history_path,history)
    save(args.out/'execution.json',execution)


if __name__=='__main__':
    main()
