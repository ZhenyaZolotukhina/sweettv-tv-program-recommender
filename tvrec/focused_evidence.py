"""Actual V3/V4 per-user evidence after selection; no model fitting."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .final_refit import digest,check_evidence
from .focused import load_dataset,original_rows
from .improve import connect
from .models import ranked_predictions
from .pipeline import LOW,SPLIT,save
from .rerank_evidence import arithmetic


def read_rows(path):
    with Path(path).open(newline='') as f:return {r['user_id']:r for r in csv.DictReader(f)}


def main():
    out=Path('artifacts/stage5');frozen=json.loads((out/'frozen_prior.json').read_text())['files'];check_evidence(frozen)
    selection=json.loads((out/'selection.json').read_text());block=selection['selected']
    if block=='retain_v3':raise RuntimeError('No selected V4; preserve rejected experiments without a cosmetic candidate')
    prior_segments=pd.read_csv('artifacts/stage4/private/v2_v3_user_deltas.csv',dtype={'user_id':str})
    pairs=[];segments=[];quality=[]
    for window in ['main','earlier']:
        old=original_rows(window);new=read_rows(out/'private'/f'{window}_{block}.csv')
        base,meta,z=load_dataset(window);X=np.load(base/'X.npy',mmap_mode='r')
        fi=meta['feature_names'].index('history_log_frequency')
        starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
        oneoff={method:{'warm_top5':0,'predicted_zero_or_one':0,'correct_zero_or_one':0,
                       'wrong_zero_or_one':0,'hits':0} for method in ['v3','v4']}
        rows=[]
        for n,u in enumerate(meta['users']):
            source=prior_segments[(prior_segments.window==window)&(prior_segments.user_id==u)].iloc[0]
            row={'window':window,'user_id':u,'v3_ap5':float(old[u]['ap5']),'v4_ap5':float(new[u]['ap5']),
                'delta':float(new[u]['ap5'])-float(old[u]['ap5']),
                **{field:source[field] for field in ['history_group','repeat_group','channel_group']}}
            rows.append(row);pairs.append(row)
            lo,hi=starts[n:n+2]
            index={meta['items'][i]:lo+j for j,i in enumerate(z['item_index'][lo:hi])}
            truth=set(map(int,old[u]['truth'].split()))
            for method,values in [('v3',old),('v4',new)]:
                pred=list(map(int,values[u]['predictions'].split()))
                oneoff[method]['hits']+=len(set(pred)&truth)
                if z['cold'][n]:continue
                oneoff[method]['warm_top5']+=5
                for i in pred:
                    count=int(round(np.expm1(X[index[i],fi])))
                    oneoff[method]['predicted_zero_or_one']+=count<=1
                    oneoff[method]['correct_zero_or_one']+=i in truth and count<=1
                    oneoff[method]['wrong_zero_or_one']+=i not in truth and count<=1
        df=pd.DataFrame(rows)
        for feature in ['history_group','repeat_group','channel_group']:
            for group,g in df.groupby(feature):
                segments.append({'window':window,'segmentation':feature,'group':group,'users':len(g),
                    'v3_map5':float(g.v3_ap5.mean()),'v4_map5':float(g.v4_ap5.mean()),
                    'gain':float(g.delta.mean()),'improved':int((g.delta>0).sum()),'regressed':int((g.delta<0).sum())})
        quality.append({'window':window,'counts':oneoff,
            'explanation':'A zero/one historical occurrence is unknown/weak evidence, not a dislike label; counts describe observed errors only.'})
    pd.DataFrame(pairs).to_csv(out/'private/v3_v4_user_deltas.csv',index=False)
    pd.DataFrame(segments).to_csv(out/'segments.csv',index=False)
    save(out/'ranking_quality.json',quality)
    c=connect(Path('cache'))
    main=[r for r in pairs if r['window']=='main']
    success=max(main,key=lambda r:(r['delta'],r['user_id']))['user_id']
    failure=min(main,key=lambda r:(r['delta'],r['user_id']))['user_id']
    motivating=json.loads((out/'private/v3_failure_cases.json').read_text())
    originals=json.loads(Path('artifacts/stage2/private/representative_users.json').read_text())
    chosen=[(r['user_id'],'pre-experiment measured V3 one-off displacement') for r in motivating if r['window']=='main']
    chosen += [(success,'largest observed V4 correction'),(failure,'largest observed V4 regression')]
    chosen += [(r['user_id'],'same original representative user') for r in originals]
    chosen=list(dict.fromkeys(chosen))
    old=original_rows('main');new=read_rows(out/'private'/f'main_{block}.csv')
    _,meta,z=load_dataset('main');users=meta['users'];items=meta['items'];ui={u:n for n,u in enumerate(users)}
    cold={u for n,u in enumerate(users) if z['cold'][n]}
    with np.load('cache/improvements/main_v1_scores.npz') as q:
        baseline={name:q[name] for name in ['popularity','repeat','cf','cbf']}
    fallback=np.lexsort((np.array(items),-baseline['popularity'][0]))
    examples=[]
    for user,reason in chosen:
        history=c.execute('''WITH runs AS (SELECT e.sid,a.item,a.channel,a.duration,
                greatest(e.a,?) a,least(e.b,?) b FROM continuous_events e JOIN base.airings a USING(sid)
                WHERE e.user_id=? AND a.item>0 AND a.duration>0 AND e.a<? AND e.b>?),
            per_airing AS (SELECT sid,item,channel,duration,max(b-a) AS watch_seconds,max(b) AS last_second
                FROM runs GROUP BY sid,item,channel,duration)
            SELECT item,count(*) frequency50,count(*) FILTER(WHERE 5*watch_seconds>=4*duration) frequency80,
                max(last_second) last_second FROM per_airing WHERE 2*watch_seconds>=duration
                GROUP BY item ORDER BY frequency50 DESC,item''',[LOW,SPLIT,user,SPLIT,LOW]).fetchdf()
        pred={name:ranked_predictions(score[[ui[user]]],[user],items,fallback,cold)[user] for name,score in baseline.items()}
        pred.update(v3=list(map(int,old[user]['predictions'].split())),v4=list(map(int,new[user]['predictions'].split())))
        truth=set(map(int,old[user]['truth'].split()))
        ids=set(history.item)|truth|set(i for p in pred.values() for i in p)
        metadata={int(i):{'item':int(i),'title':t,'category':cat} for i,t,cat in c.execute('''SELECT item,
            first(title ORDER BY ss,sid),first(category ORDER BY ss,sid) FROM base.airings WHERE item IN ('''+','.join(map(str,sorted(ids)))+') GROUP BY item').fetchall()}
        ap={name:arithmetic(truth,p) for name,p in pred.items()}
        assert abs(ap['v3']['ap5']-float(old[user]['ap5']))<1e-12 and abs(ap['v4']['ap5']-float(new[user]['ap5']))<1e-12
        examples.append({'user_id':user,'selection_reason':reason,'history_cutoff':SPLIT,
            'historical_programmes':[metadata[int(r.item)]|{'frequency50':int(r.frequency50),
                'frequency80':int(r.frequency80),'last_second':int(r.last_second)} for r in history.itertuples()],
            'truth':[metadata[i] for i in sorted(truth)],
            'method_top5':{name:[metadata[i] for i in p] for name,p in pred.items()},'exact_ap5':ap,
            'posthoc_examples_not_used_to_fit_features':True,'executor':'AI assistant, not student verification'})
    save(out/'private/user_examples.json',examples)
    save(out/'user_summary.json',[{'alias':f'Case-{n+1}','reason':r['selection_reason'],
        'v3_ap5':r['exact_ap5']['v3']['ap5'],'v4_ap5':r['exact_ap5']['v4']['ap5'],
        'v3_top5':r['method_top5']['v3'],'v4_top5':r['method_top5']['v4']} for n,r in enumerate(examples)])
    # Descriptive split-gain ranking; block-level ablation, not individual gain,
    # provides the actual experimental evidence for choosing the six features.
    import pickle
    with Path('cache/focused_v4/loyalty_early.pkl').open('rb') as f:model=pickle.load(f)
    info=json.loads(Path('cache/focused_v4/main/extras.json').read_text())
    names=meta['feature_names']+info['columns']['loyalty'];gains=np.zeros(len(names))
    for iteration in model._predictors:
        for tree in iteration:
            for node in tree.nodes:
                if not node['is_leaf'] and node['gain']>0:gains[int(node['feature_idx'])]+=node['gain']
    save(out/'selected_training_split_gain.json',{'interpretation':'descriptive training split gains only; six-feature block tested as a whole',
        'added_features':[{'name':name,'gain':float(gains[n])} for n,name in enumerate(names) if name.startswith('loyal_')]})
    check_evidence(frozen);c.close()
    print({'ranking_quality':quality,'examples':[(r['selection_reason'],r['exact_ap5']['v3']['ap5'],r['exact_ap5']['v4']['ap5']) for r in examples]})


if __name__=='__main__':main()
