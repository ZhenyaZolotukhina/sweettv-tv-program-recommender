"""Measured diagnostics from the cached database; no raw CSV rescan."""
import json
import time

import numpy as np
import pandas as pd

from .pipeline import LOW, SPLIT, TEST, END, records, save


def diagnostics(c, out):
    started = time.perf_counter()
    private = out/'private'
    private.mkdir(exist_ok=True)
    result = {}
    result['date_ranges'] = records(c, '''SELECT 'views' AS "source",min(vs) first_start,max(vs) last_start,max(ve) last_end FROM views
        UNION ALL SELECT 'schedule_'||period,min(ss),max(ss),max(se) FROM airings GROUP BY period''')
    for row in result['date_ranges']:
        for key in ['first_start', 'last_start', 'last_end']:
            row[key] = str(np.datetime64(row[key], 's'))
    result['duration_mismatches'] = records(c, '''SELECT duraton-(ve-vs) delta_seconds,count(*) AS count,
        min(vs) first_start,max(vs) last_start FROM views WHERE duraton!=ve-vs GROUP BY 1''')
    for row in result['duration_mismatches']:
        for key in ['first_start', 'last_start']:
            row[key] = str(np.datetime64(row[key], 's'))
    result['duration_policy'] = 'Actual start/stop clock intervals used; duraton audited only. All 117 discrepancies are -3600s, consistent with a clock change but timezone unverified; no inferred correction.'
    result['duplicate_view_policy'] = '17 exact duplicate source rows retained as logged events; may slightly inflate frequency/duration. Schedule conflicts never multiply seconds within a view.'
    result['invalid_airings'] = records(c, 'SELECT sid,period,source_row,channel,ss,se,duration,item FROM airings WHERE ss IS NULL OR duration IS NULL OR duration<=0')
    result['schedule_overlap_examples'] = records(c, '''SELECT * FROM (SELECT sid,period,source_row,channel,ss,se,
        max(se) OVER(PARTITION BY channel ORDER BY ss,sid ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prior_max_end
        FROM airings WHERE se>ss) WHERE ss<prior_max_end ORDER BY channel,ss''')
    result['training_matrix'] = records(c, """SELECT count(DISTINCT user_id) users,count(DISTINCT item) items,
        count(DISTINCT (user_id,item)) nonzero_pairs FROM interactions WHERE phase='train' AND seconds>=60""")[0]
    m = result['training_matrix']
    m['density'] = m['nonzero_pairs']/(m['users']*m['items'])
    result['validation_truth_sizes'] = records(c, """SELECT min(n) min_items,quantile_cont(n,[0.25,0.5,0.75,0.9]) quantiles,
        max(n) max_items FROM (SELECT user_id,count(DISTINCT item) n FROM interactions JOIN template USING(user_id)
        WHERE phase='validation' AND seconds>=60 GROUP BY user_id)""")[0]
    result['candidate_windows'] = []
    for period, lo, hi in [(11,LOW,SPLIT), (21,SPLIT,TEST), (31,TEST,END)]:
        result['candidate_windows'] += records(c, f'''SELECT {period} period,count(DISTINCT item) file_labelled_ids,
            count(DISTINCT item) FILTER(WHERE ss<{hi} AND se>{lo} AND duration>0) in_calendar_window,
            count(*) FILTER(WHERE ss>={hi} OR se<={lo}) out_of_window_rows
            FROM airings WHERE period={period} AND item>0''')
    assert c.execute('SELECT min(ss) FROM airings WHERE period=21').fetchone()[0] >= SPLIT
    assert c.execute('SELECT min(ss) FROM airings WHERE period=31').fetchone()[0] >= TEST
    assert c.execute('SELECT max(ve) FROM views').fetchone()[0] <= TEST
    result['future_view_leakage_checks_passed'] = True
    result['metadata_coverage'] = records(c, '''SELECT a.period,count(DISTINCT item) labelled_ids,
        count(DISTINCT item) FILTER(WHERE tv_show_genre_1 IS NOT NULL) with_first_genre,
        count(DISTINCT item) FILTER(WHERE director IS NOT NULL) with_director,
        count(DISTINCT item) FILTER(WHERE actors IS NOT NULL) with_actors
        FROM airings a JOIN schedule s USING(sid) WHERE item>0 GROUP BY a.period ORDER BY a.period''')
    mapping = pd.read_csv(out/'channel_mapping.csv')
    ids = ','.join(map(str, sorted(set(mapping.loc[mapping.evidence=='exact_normalized_title','channel']))))
    result['mapping_coverage'] = records(c, f'''SELECT count(*) view_rows,
        count(*) FILTER(WHERE channel IN ({ids})) exact_title_verified_channel_views,
        count(*) FILTER(WHERE channel IN (SELECT channel FROM airings)) shared_namespace_schedule_channel_views FROM views''')[0]
    result['count_provenance'] = {
        'duration_mismatches':'views: duraton != ve-vs; group by duraton-(ve-vs)',
        'training_matrix':"interactions phase='train' AND seconds>=60; distinct users/items/pairs",
        'validation_truth_sizes':"interactions JOIN template phase='validation' AND seconds>=60; distinct items/user",
        'candidate_windows':'airings: distinct positive item, valid duration, intersects calendar window',
        'metadata_coverage':'schedule JOIN airings USING(sid); distinct item with nonmissing metadata',
        'mapping_coverage':'views matched to channel_mapping.csv exact_title_verified or shared namespace',
        'other_counts':'Executable SQL in tvrec/pipeline.py; original input SHA256 and source-row IDs saved'}
    result['limitations'] = [
        'Labels reconstructed, not official hidden labels; watch threshold and MAP empty-truth convention unverified against Kaggle.',
        'Ten-week truth sets are large, making popularity hits common. Local MAP is not a Kaggle score.',
        'Current VseTV directory is not a historical 2020 crosswalk; 211 ID/title matches support namespace identity; remaining channels not individually confirmed.',
        'No timezone supplied; both timestamp formats treated as the same naive wall clock.',
        'Single holdout and 240 configurations create selection optimism; eight cold eligible users give noisy subgroup MAP.',
        'Minute-scale glances are positive/unknown only, never dislikes.'
    ]
    result['observed_problems_and_corrections'] = [
        {'problem':'User IDs exceed signed int64', 'diagnosis':'2,918,899 view rows above 2^63-1',
         'correction':'all_varchar ingestion; strings throughout; synthetic large-ID and leading-zero roundtrip test'},
        {'problem':'13 conflicting schedule rows', 'diagnosis':'Per-channel running maximum end; source IDs saved',
         'correction':'Disjoint latest-start sweep, stable ties; per-view second conservation asserted on full data'},
        {'problem':'Schedules spill beyond calendar weeks', 'diagnosis':'candidate_windows and date_ranges',
         'correction':'Clip views to cutoffs and availability to interval intersection with calendar window'},
        {'problem':'duraton differs from stop-start', 'diagnosis':'117 rows differ by -3600s; examples saved privately',
         'correction':'Use timestamps rather than deriving stop_time from duraton; no fabricated timezone correction'}
    ]
    save(private/'data_quality_examples.json', {
        'duration_mismatches':records(c,'SELECT vid,user_id,channel,vs,ve,duraton FROM views WHERE duraton!=ve-vs ORDER BY vid LIMIT 5'),
        'duplicate_view_source_ids':records(c, '''SELECT user_id,channel,vs,ve,list(vid ORDER BY vid) source_rows FROM views
            GROUP BY user_id,channel,vs,ve,duraton HAVING count(*)>1 ORDER BY min(vid)''')})
    if (private/'user_examples.json').exists():
        examples = json.loads((private/'user_examples.json').read_text())
        result['global_top5_validation_hit_prevalence'] = []
        for item in examples[0]['predictions']['popularity']:
            row = records(c,f'SELECT {item} item,first(title ORDER BY ss) title FROM airings WHERE item={item}')[0]
            row['template_users_watching'] = c.execute('''SELECT count(DISTINCT user_id) FROM interactions JOIN template USING(user_id)
                WHERE phase='validation' AND seconds>=60 AND item=?''',[item]).fetchone()[0]
            result['global_top5_validation_hit_prevalence'].append(row)
    result['advanced_method_recommendations'] = [
        {'method':'Implicit-feedback matrix factorization (ALS)',
         'evidence':f"{m['users']} users, {m['items']} items and {m['nonzero_pairs']} nonzero pairs form a compact sparse matrix; compare fraction/frequency confidence with seconds."},
        {'method':'Content-based filtering (metadata TF-IDF user profiles)',
         'evidence':'Many scheduled items have no viewing history, while measured genre/director/actor coverage is high; metadata permits cold-item scoring without future views.'}
    ]
    result['diagnostic_seconds'] = time.perf_counter()-started
    save(out/'diagnostics.json',result)
