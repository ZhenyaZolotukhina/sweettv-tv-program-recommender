"""Independent formatting, channel-count and full-ranking checks for v2.

python -m tvrec.verify_v2
Does not call generation feature functions and never changes v1 evidence.
"""
import csv
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

from .final_refit import digest,check_evidence
from .verify_final import validate_file_records
from .core import ap_at_5


def main():
    out=Path('artifacts/stage3')
    frozen=json.loads((out/'frozen_prior.json').read_text())['files']
    check_evidence(frozen)
    path=Path('artifacts/final/submission_hybrid_v2.csv')
    manifest_path=Path('artifacts/final/submission_hybrid_v2.manifest.json')
    manifest=json.loads(manifest_path.read_text())
    with Path('sweettv-tv-program-recommender/submission.csv').open(newline='') as f:
        reader=csv.reader(f)
        assert next(reader)==['user_id','tv_show_id']
        users=[r[0] for r in reader]
    c=duckdb.connect('cache/stage2.duckdb',read_only=True)
    c.execute("ATTACH 'cache/data.duckdb' AS base (READ_ONLY)")
    c.execute("SET threads=4; SET memory_limit='1GB'")
    items=sorted(i for i, in c.execute('''SELECT DISTINCT item FROM base.airings WHERE period=31 AND item>0
        AND duration>0 AND ss<1603065600 AND se>1595808000''').fetchall())
    with path.open(newline='') as f:
        reader=csv.reader(f)
        header=next(reader)
        rows=list(reader)
    validate_file_records(users,header,rows,set(items))
    assert len(users)==1260 and digest(path)==manifest['submission_sha256']
    # Independently use DISTINCT user/airing instead of generation's longest-run
    # dataframe selection. Visits never sum to reach the >=50% input threshold.
    occurrences=c.execute('''SELECT DISTINCT e.user_id,e.sid,a.channel,a.item
        FROM continuous_events e JOIN base.airings a USING(sid)
        WHERE a.item>0 AND a.duration>0 AND e.a<1595808000 AND e.b>1583712000
        AND 2*(least(e.b,1595808000)-greatest(e.a,1583712000))>=a.duration''').fetchdf()
    assert len(occurrences)==manifest['channel_fit_qualified_airings']
    directory=Path('cache/improvements/final_v2')
    user_counts=occurrences.groupby(['user_id','channel']).size().rename('count').reset_index()
    item_counts=occurrences.groupby(['channel','item']).size().rename('count').reset_index()
    totals=occurrences.groupby('channel').size()
    pd.testing.assert_frame_equal(user_counts,pd.read_csv(directory/'user_channel_counts.csv',dtype={'user_id':str}),check_dtype=False)
    pd.testing.assert_frame_equal(item_counts,pd.read_csv(directory/'channel_item_counts.csv'),check_dtype=False)
    channels=sorted(totals.index)
    ch={x:n for n,x in enumerate(channels)}
    ui={u:n for n,u in enumerate(users)}
    ii={i:n for n,i in enumerate(items)}
    u=user_counts[user_counts.user_id.isin(ui)]
    profile=sparse.csr_matrix((np.sqrt(u['count'].to_numpy()),
        ([ui[x] for x in u.user_id],[ch[x] for x in u.channel])),shape=(len(users),len(channels)))
    mass=np.asarray(profile.sum(axis=1)).ravel()
    profile=sparse.diags(np.divide(1,mass,out=np.zeros_like(mass),where=mass>0))@profile
    p=item_counts[item_counts.item.isin(ii)]
    values=np.array([n/totals.loc[channel] for channel,n in zip(p.channel,p['count'])])
    conditional=sparse.csr_matrix((values,([ch[x] for x in p.channel],[ii[x] for x in p.item])),shape=(len(channels),len(items)))
    channel_score=(profile@conditional).toarray().astype(np.float32)
    with np.load(directory/'scores.npz') as z:
        expected_channel=z['channel']; expected_hybrid=z['hybrid']
    np.testing.assert_array_equal(channel_score,expected_channel)
    with np.load('cache/final/hybrid_v1/component_scores.npz') as z:
        base=z['hybrid']; fallback=z['fallback']
    fallback_position=np.empty(len(items),dtype=int)
    fallback_position[fallback]=np.arange(len(items))
    warm={u for u, in c.execute('''SELECT DISTINCT e.user_id FROM continuous_events e JOIN base.airings a USING(sid)
        WHERE a.item>0 AND a.duration>0 AND least(e.b,1595808000)-greatest(e.a,1583712000)>=60''').fetchall()}
    for n,user in enumerate(users):
        order=np.lexsort((fallback_position,-channel_score[n]))[:200]
        order=order[channel_score[n,order]>0]
        rr=np.zeros(len(items),dtype=np.float32)
        rr[order]=1/(10+np.arange(1,len(order)+1))
        score=base[n]+manifest['addition']['weight']*rr
        np.testing.assert_array_equal(score,expected_hybrid[n])
        predicted=np.lexsort((fallback_position,-score))
        supported=predicted[score[predicted]>0].tolist()
        done=set(supported)
        predicted=(supported+[int(i) for i in fallback if i not in done])[:5] if user in warm else fallback[:5]
        assert [items[i] for i in predicted]==list(map(int,rows[n][1].split())),f'Ranking mismatch at row {n+1}'
    offline_checked=0
    for window,reference,lower,upper,period in [
        ('main','main_fusion_19.csv',1589760000,1595808000,21),
        ('earlier','earlier_fusion_0.csv',int(np.datetime64('2020-04-13','s').astype('int64')),1589760000,None)]:
        with (Path('artifacts/stage2/private')/reference).open(newline='') as f:
            original={r['user_id']:r for r in csv.DictReader(f)}
        query='SELECT DISTINCT item FROM base.airings WHERE item>0 AND duration>0 AND ss<? AND se>?'
        if period is not None:
            query+=' AND period=21'
        available={i for i, in c.execute(query,[upper,lower]).fetchall()}
        for file in sorted((out/'private').glob(f'{window}_*.csv')):
            with file.open(newline='') as f:
                records=list(csv.DictReader(f))
            validate_file_records(users,['user_id','tv_show_id'],
                [[r['user_id'],r['predictions']] for r in records],available)
            for r in records:
                u=r['user_id']
                truth=set(map(int,r['truth'].split()))
                predicted=list(map(int,r['predictions'].split()))
                assert truth==set(map(int,original[u]['truth'].split()))
                assert r['group']==original[u]['group']
                assert abs(ap_at_5(truth,predicted)-float(r['ap5']))<1e-12
                assert [rank for rank,item in enumerate(predicted,1) if item in truth]==list(map(int,r['hit_ranks'].split()))
                if r['group']=='cold':
                    assert r['predictions']==original[u]['predictions']
            offline_checked+=1
    check_evidence(frozen)
    result={'valid':True,'rows':len(rows),'candidates':len(items),'submission_sha256':digest(path),
        'original_ids_and_order':True,'five_distinct_positive_available_ids':True,
        'independent_sql_qualification_and_channel_counts':True,'all_channel_scores_and_final_rankings_recomputed':True,
        'same_v1_cold_fallback':True,'prior_v1_and_evidence_unchanged':True,
        'offline_method_files_checked':offline_checked,'same_frozen_truth_users_availability_warm_cold_groups':True,
        'all_offline_ap5_and_hit_ranks_recomputed':True,'kaggle_requests':0}
    (out/'v2_independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    # Only the NEW v2 manifest is updated.
    manifest['independent_validation']=result
    manifest['source_sha256']={str(p):digest(p) for p in sorted(Path('tvrec').glob('*.py'))}
    manifest['requirements_sha256']=digest('requirements.txt')
    if (out/'test_results.json').exists():
        manifest['test_results']=json.loads((out/'test_results.json').read_text())
        assert manifest['test_results']['passed']
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(result)
    c.close()


if __name__=='__main__':
    main()
