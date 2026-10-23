"""Post-selection user/segment diagnostics; never feeds features or selection.

python -m tvrec.improvement_evidence
"""
import csv
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .final_refit import digest,check_evidence
from .improve import connect
from .labels import event_frame
from .pipeline import LOW,SPLIT,TEST,save
from .rank_features import qualified_occurrences,channel_scores


def prediction_rows(path):
    with Path(path).open(newline='') as f:
        return {r['user_id']:{'predictions':list(map(int,r['predictions'].split())),
            'truth':set(map(int,r['truth'].split())),'ap5':float(r['ap5']),'group':r['group']}
            for r in csv.DictReader(f)}


def ap_arithmetic(truth,pred):
    ranks=[n for n,i in enumerate(pred,1) if i in truth]
    terms=[Fraction(h,r) for h,r in enumerate(ranks,1)]
    denominator=min(5,len(truth))
    value=sum(terms,Fraction(0))/denominator if denominator else Fraction(0)
    return {'hit_ranks':ranks,'arithmetic':f"({' + '.join(f'{h}/{r}' for h,r in enumerate(ranks,1)) or '0'})/{denominator}" if denominator else 'empty truth -> 0',
            'exact_fraction':str(value),'ap5':float(value)}


def main():
    out=Path('artifacts/stage3')
    frozen=json.loads((out/'frozen_prior.json').read_text())['files']
    check_evidence(frozen)
    selection=json.loads((out/'robust_selection.json').read_text())
    assert selection['selected']=='v1_plus_channel'
    weight=selection['channel_weight']
    c=connect(Path('cache'))
    segment_rows=[]
    pair_rows=[]
    examples=[]
    change_counts=[]
    for window,cutoff in [('main',SPLIT),('earlier',int(np.datetime64('2020-04-13','s').astype('int64')))]:
        before=prediction_rows(out/'private'/f'{window}_v1.csv')
        after=prediction_rows(out/'private'/f'{window}_v1_plus_channel_{weight:g}.csv')
        assert before.keys()==after.keys()
        training=event_frame(c,LOW,cutoff)
        occ=qualified_occurrences(training,cutoff)
        uc=occ.groupby(['user_id','channel']).size()
        total=uc.groupby(level=0).sum()
        concentration=uc.groupby(level=0).max()/total
        history=occ.groupby('user_id').item.nunique()
        watched={u:set(g.item) for u,g in occ.groupby('user_id')}
        rows=[]
        introduced_relevant=removed_relevant=changed=reordered_only=0
        introduced_relevant_without_history=0
        for user,r in before.items():
            assert r['truth']==after[user]['truth']
            old,new=r['predictions'],after[user]['predictions']
            delta=after[user]['ap5']-r['ap5']
            count=int(history.get(user,0))
            focus=float(concentration.get(user,0))
            row={'window':window,'user_id':user,'ap5_v1':r['ap5'],'ap5_v2':after[user]['ap5'],
                'delta':delta,'history_items':count,'channel_concentration':focus,
                'history_group':'cold' if r['group']=='cold' else 'sparse_under_10' if count<10 else 'regular_10_plus',
                'channel_group':'cold' if r['group']=='cold' else 'focused_at_least_60pct' if focus>=.6 else 'broad_below_60pct'}
            rows.append(row)
            pair_rows.append(row)
            added=set(new)-set(old)
            removed=set(old)-set(new)
            introduced_relevant+=len(added&r['truth'])
            removed_relevant+=len(removed&r['truth'])
            introduced_relevant_without_history+=len((added&r['truth'])-watched.get(user,set()))
            changed+=old!=new
            reordered_only+=(old!=new and set(old)==set(new))
        frame=pd.DataFrame(rows)
        for field in ['history_group','channel_group']:
            for group,g in frame.groupby(field):
                segment_rows.append({'window':window,'segmentation':field,'segment':group,'users':len(g),
                    'v1_map5':g.ap5_v1.mean(),'v2_map5':g.ap5_v2.mean(),'mean_gain':g.delta.mean(),
                    'improved_users':int((g.delta>0).sum()),'worsened_users':int((g.delta<0).sum())})
        change_counts.append({'window':window,'changed_recommendation_lists':changed,'only_order_changed':reordered_only,
            'introduced_relevant_programme_ids':introduced_relevant,'removed_relevant_programme_ids':removed_relevant,
            'introduced_relevant_programmes_without_qualifying_history':introduced_relevant_without_history})
        if window=='main':
            # Deliberately contrast an actual success and failure. These examples
            # are selected AFTER evaluation and are not representative estimators.
            best=frame.sort_values(['delta','user_id'],ascending=[False,True]).iloc[0].user_id
            worst=frame.sort_values(['delta','user_id'],ascending=[True,True]).iloc[0].user_id
            original=json.loads(Path('artifacts/stage2/private/representative_users.json').read_text())
            chosen=[(best,'largest observed AP improvement'),(worst,'largest observed AP regression')]
            chosen += [(r['user_id'],'same representative user as Stage 2') for r in original if r['user_id'] not in {best,worst}]
            items=sorted(i for i, in c.execute('''SELECT DISTINCT item FROM base.airings WHERE period=21 AND item>0
                AND duration>0 AND ss<? AND se>?''',[TEST,SPLIT]).fetchall())
            chosen_users=[u for u,_ in chosen]
            channel=channel_scores(occ,chosen_users,items)
            item_index={i:n for n,i in enumerate(items)}
            for n,(user,reason) in enumerate(chosen):
                personal=occ[occ.user_id==user]
                top=personal.groupby('item',as_index=False).agg(frequency=('sid','size'),last_second=('b','max')).sort_values(['frequency','item'],ascending=[False,True]).head(15)
                ids=set(top.item)|set(before[user]['predictions'])|set(after[user]['predictions'])|before[user]['truth']
                query=c.execute('''SELECT item,first(title ORDER BY ss,sid) title,first(category ORDER BY ss,sid) category
                    FROM base.airings WHERE item IN ('''+','.join(map(str,sorted(ids)))+') GROUP BY item')
                metadata={int(r[0]):{'item':int(r[0]),'title':r[1],'category':r[2]} for r in query.fetchall()}
                hist={int(i):int(v) for i,v in personal.groupby('item').size().items()}
                lists={method:[metadata[i]|{'training_frequency':hist.get(i,0),
                    'channel_component_score':float(channel[n,item_index[i]])} for i in source[user]['predictions']]
                    for method,source in [('v1',before),('v2',after)]}
                profile=[]
                for ch,count in personal.groupby('channel').size().sort_values(ascending=False).head(5).items():
                    title=c.execute('SELECT first(channel_title ORDER BY ss,sid) FROM base.airings WHERE channel=? AND ss<?',[int(ch),cutoff]).fetchone()[0]
                    profile.append({'channel':int(ch),'title':title,'qualifying_airings':int(count),'raw_share':float(count/len(personal))})
                arithmetic={method:ap_arithmetic(before[user]['truth'],source[user]['predictions'])
                    for method,source in [('v1',before),('v2',after)]}
                assert abs(arithmetic['v1']['ap5']-before[user]['ap5'])<1e-12
                assert abs(arithmetic['v2']['ap5']-after[user]['ap5'])<1e-12
                examples.append({'user_id':user,'selection_reason':reason,'window':'main',
                    'selection_is_posthoc_diagnostic_not_a_training_feature':True,
                    'historical_top_programmes':[metadata[int(r.item)]|{'frequency':int(r.frequency),'last_second':int(r.last_second)} for r in top.itertuples()],
                    'historical_channel_profile':profile,'truth':[metadata[i] for i in sorted(before[user]['truth'])],
                    'recommendations':lists,'ap5_calculations':arithmetic,
                    'observed_delta':after[user]['ap5']-before[user]['ap5']})
    pd.DataFrame(segment_rows).to_csv(out/'segment_results.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(out/'private/paired_user_deltas.csv',index=False)
    save(out/'ranking_change_counts.json',change_counts)
    save(out/'private/contrasting_users.json',{'executor':'AI assistant; not student verification','examples':examples})
    save(out/'contrasting_user_summary.json',[{'alias':f'Example-{n+1}','reason':ex['selection_reason'],
        'v1_ap5':ex['ap5_calculations']['v1']['ap5'],'v2_ap5':ex['ap5_calculations']['v2']['ap5'],
        'delta':ex['observed_delta'],'history_programmes':ex['historical_top_programmes'],
        'recommendations':ex['recommendations']} for n,ex in enumerate(examples)])
    check_evidence(frozen)
    print({'segments':segment_rows,'ranking_changes':change_counts,
        'examples':[(r['selection_reason'],r['ap5_calculations']['v1']['ap5'],r['ap5_calculations']['v2']['ap5']) for r in examples]})
    c.close()


if __name__=='__main__':
    main()
