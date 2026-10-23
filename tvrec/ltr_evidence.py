"""Post-selection V4/group-ranker diagnostics and real ordering examples."""
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn

from .final_refit import digest,check_evidence
from .focused import load_dataset
from .group_rank import lightgbm_module,prediction_lists
from .improve import connect
from .models import ranked_predictions
from .pipeline import LOW,SPLIT,save
from .rerank_evidence import arithmetic


def rows(path):
    with Path(path).open(newline='') as f:return {r['user_id']:r for r in csv.DictReader(f)}


def main():
    out=Path('artifacts/stage6');frozen=json.loads((out/'frozen_prior.json').read_text())['files'];check_evidence(frozen)
    decision=json.loads((out/'decision.json').read_text());chosen=decision['selected']
    candidate=chosen if chosen!='retain_v4' else 'lambda_top5'
    old_segments=pd.read_csv('artifacts/stage4/private/v2_v3_user_deltas.csv',dtype={'user_id':str})
    all_rows=[];segments=[];recall=[]
    for window in ['main','earlier']:
        before=rows(f'artifacts/stage5/private/{window}_loyalty.csv')
        after=rows(out/'private'/f'{window}_{candidate}.csv')
        base,meta,z=load_dataset(window)
        starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
        score=np.load(Path('cache/group_ltr')/(window+'_lambda_top5_scores.npy'))
        records=[];n_relevant=sum(len(v['truth'].split()) for v in before.values());hits200=0
        for n,u in enumerate(meta['users']):
            label=old_segments[(old_segments.window==window)&(old_segments.user_id==u)].iloc[0]
            row={'window':window,'user_id':u,'v4_ap5':float(before[u]['ap5']),'candidate_ap5':float(after[u]['ap5']),
                'gain':float(after[u]['ap5'])-float(before[u]['ap5']),
                'v4_hit_ranks':before[u]['hit_ranks'],'candidate_hit_ranks':after[u]['hit_ranks'],
                'v4_top5':before[u]['predictions'],'candidate_top5':after[u]['predictions'],
                **{field:label[field] for field in ['history_group','repeat_group','channel_group']}}
            records.append(row);all_rows.append(row)
            lo,hi=starts[n:n+2]
            order=np.argsort(-score[lo:hi],kind='stable')[:200] if not z['cold'][n] else np.arange(min(200,hi-lo))
            predicted={meta['items'][i] for i in z['item_index'][lo:hi][order]}
            hits200+=len(predicted&set(map(int,before[u]['truth'].split())))
        df=pd.DataFrame(records)
        for feature in ['history_group','repeat_group','channel_group']:
            for group,g in df.groupby(feature):
                segments.append({'window':window,'segmentation':feature,'group':group,'users':len(g),
                    'v4_map5':float(g.v4_ap5.mean()),'candidate_map5':float(g.candidate_ap5.mean()),'gain':float(g.gain.mean()),
                    'improved':int((g.gain>0).sum()),'regressed':int((g.gain<0).sum())})
        recall.append({'window':window,'candidate_policy_unchanged':True,
            'lambdarank_after_reranking_recall200_micro':hits200/n_relevant,
            'note':'Different Top200 positions reflect ranking, not new candidates. Frozen pool/recall audit is separate.'})
    pd.DataFrame(all_rows).to_csv(out/'private/user_ap_deltas.csv',index=False)
    pd.DataFrame(segments).to_csv(out/'segments.csv',index=False)
    save(out/'candidate_ranking_recall.json',recall)
    main=[r for r in all_rows if r['window']=='main']
    success=max(main,key=lambda r:(r['gain'],r['user_id']))['user_id']
    failure=min(main,key=lambda r:(r['gain'],r['user_id']))['user_id']
    original=json.loads(Path('artifacts/stage2/private/representative_users.json').read_text())
    selected=[(success,'largest observed AP correction'),(failure,'largest observed AP regression')]
    selected += [(r['user_id'],'same Stage-2 representative') for r in original]
    before=rows('artifacts/stage5/private/main_loyalty.csv');after=rows(out/'private'/f'main_{candidate}.csv')
    _,meta,z=load_dataset('main');items=meta['items'];ui={u:n for n,u in enumerate(meta['users'])}
    cold={u for n,u in enumerate(meta['users']) if z['cold'][n]}
    with np.load('cache/improvements/main_v1_scores.npz') as q:components={k:q[k] for k in ['popularity','repeat','cf','cbf']}
    fallback=np.lexsort((np.array(items),-components['popularity'][0]))
    c=connect(Path('cache'));examples=[]
    for user,reason in selected:
        hist=c.execute('''WITH runs AS (SELECT e.sid,a.item,a.duration,
            greatest(e.a,?) a,least(e.b,?) b FROM continuous_events e JOIN base.airings a USING(sid)
            WHERE e.user_id=? AND a.item>0 AND a.duration>0 AND e.a<? AND e.b>?),
            unique_airings AS (SELECT sid,item,duration,max(b-a) AS seconds_watch,max(b) AS last_second
            FROM runs GROUP BY sid,item,duration)
            SELECT item,count(*) frequency50,count(*) FILTER(WHERE 5*seconds_watch>=4*duration) frequency80,
                max(last_second) AS last_second FROM unique_airings WHERE 2*seconds_watch>=duration
            GROUP BY item ORDER BY frequency50 DESC,item''',[LOW,SPLIT,user,SPLIT,LOW]).fetchdf()
        pred={k:ranked_predictions(v[[ui[user]]],[user],items,fallback,cold)[user] for k,v in components.items()}
        pred['v4']=list(map(int,before[user]['predictions'].split()));pred[candidate]=list(map(int,after[user]['predictions'].split()))
        truth=set(map(int,before[user]['truth'].split()))
        ids=set(hist.item)|truth|set(i for p in pred.values() for i in p)
        lookup={int(i):{'item':int(i),'title':title,'category':category} for i,title,category in c.execute('''SELECT item,
            first(title ORDER BY ss,sid),first(category ORDER BY ss,sid) FROM base.airings WHERE item IN ('''+','.join(map(str,sorted(ids)))+') GROUP BY item').fetchall()}
        calculations={k:arithmetic(truth,v) for k,v in pred.items()}
        assert abs(calculations['v4']['ap5']-float(before[user]['ap5']))<1e-12
        assert abs(calculations[candidate]['ap5']-float(after[user]['ap5']))<1e-12
        examples.append({'user_id':user,'reason':reason,'history_cutoff':SPLIT,
            'history':[lookup[int(r.item)]|{'frequency50':int(r.frequency50),'frequency80':int(r.frequency80),
                'last_second':int(r.last_second)} for r in hist.itertuples()],
            'truth':[lookup[i] for i in sorted(truth)],'top5':{k:[lookup[i] for i in v] for k,v in pred.items()},
            'exact_ap5':calculations,'posthoc_diagnostic_not_training_feature':True,
            'executor':'AI assistant; not student verification'})
    save(out/'private/user_examples.json',examples)
    save(out/'example_summary.json',[{'alias':f'User-{n+1}','reason':r['reason'],
        'v4_ap5':r['exact_ap5']['v4']['ap5'],'candidate_ap5':r['exact_ap5'][candidate]['ap5'],
        'v4_top5':r['top5']['v4'],'candidate_top5':r['top5'][candidate]} for n,r in enumerate(examples)])
    # A cheap existing target-tie sensitivity is reporting only, not new labels.
    sensitivity=[]
    for window in ['main','earlier']:
        alt=json.loads(Path(f'artifacts/stage5/private/{window}_tie_sensitivity_truth.json').read_text())
        a=rows(f'artifacts/stage5/private/{window}_loyalty.csv');b=rows(out/'private'/f'{window}_{candidate}.csv')
        from .core import ap_at_5
        delta=np.mean([ap_at_5(alt[u],list(map(int,b[u]['predictions'].split())))-
                       ap_at_5(alt[u],list(map(int,a[u]['predictions'].split()))) for u in a])
        sensitivity.append({'window':window,'frequency_then_recency_tie_gain':float(delta),
            'primary_target_unchanged':True,'not_used_to_train_ranker':True})
    save(out/'tie_sensitivity_reporting.json',sensitivity)
    lgb=lightgbm_module();lib=Path(lgb.__file__).parent/'lib'
    save(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'lightgbm':lgb.__version__,
        'numpy':np.__version__,'sklearn':sklearn.__version__,'optional_requirements_sha256':digest('requirements-ltr.txt'),
        'native_library_hashes':{f.name:digest(f) for f in lib.glob('*') if f.is_file()},
        'install':'binary-only isolated cache/ltr_dependencies; root requirements unchanged'})
    check_evidence(frozen);c.close()
    print({'segments':segments,'recall':recall,'tie_sensitivity':sensitivity,
        'examples':[(r['reason'],r['exact_ap5']['v4']['ap5'],r['exact_ap5'][candidate]['ap5']) for r in examples]})


if __name__=='__main__':main()
