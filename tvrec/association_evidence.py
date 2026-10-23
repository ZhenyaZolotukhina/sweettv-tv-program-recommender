"""Actual rule explanations, rejected integration cases and fixed-user evidence."""
import csv
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .assoc_study import baseline
from .final_refit import digest,check_evidence
from .focused import load_dataset
from .improve import connect
from .pipeline import LOW,SPLIT,save
from .rerank_evidence import arithmetic


def read_rows(path):
    with Path(path).open(newline='') as f:return {r['user_id']:r for r in csv.DictReader(f)}


def main():
    out=Path('artifacts/stage9');frozen=json.loads((out/'frozen_prior.json').read_text())['files'];check_evidence(frozen)
    decision=json.loads((out/'decision.json').read_text());assert decision['selected']=='retain_v5'
    segments_source=pd.read_csv('artifacts/stage4/private/v2_v3_user_deltas.csv',dtype={'user_id':str})
    pairs=[];segments=[];hit_summary=[]
    for window in ['main','earlier']:
        source=baseline(window)
        for method in ['association','bpr_calibrated']:
            new=read_rows(out/'private'/f'{window}_{method}.csv');rows=[]
            for u,r in source.items():
                old=segments_source[(segments_source.window==window)&(segments_source.user_id==u)].iloc[0]
                row={'window':window,'method':method,'user_id':u,'v5_ap5':float(r['ap5']),
                    'candidate_ap5':float(new[u]['ap5']),'delta':float(new[u]['ap5'])-float(r['ap5']),
                    'v5_hit_ranks':r['hit_ranks'],'candidate_hit_ranks':new[u]['hit_ranks'],
                    **{k:old[k] for k in ['history_group','repeat_group','channel_group']}}
                rows.append(row);pairs.append(row)
            df=pd.DataFrame(rows)
            for field in ['history_group','repeat_group','channel_group']:
                for group,g in df.groupby(field):
                    segments.append({'window':window,'method':method,'segmentation':field,'group':group,'users':len(g),
                        'v5_map5':float(g.v5_ap5.mean()),'candidate_map5':float(g.candidate_ap5.mean()),'gain':float(g.delta.mean()),
                        'improved':int((g.delta>0).sum()),'regressed':int((g.delta<0).sum())})
            count=lambda table:sum(len(set(map(int,r['predictions'].split()))&set(map(int,r['truth'].split()))) for r in table.values())
            hit_summary.append({'window':window,'method':method,'v5_relevant_top5_hits':count(source),'candidate_relevant_top5_hits':count(new)})
    pd.DataFrame(pairs).to_csv(out/'private/per_user_changes.csv',index=False)
    pd.DataFrame(segments).to_csv(out/'segments.csv',index=False)
    save(out/'relevant_top5_hits.json',hit_summary)
    c=connect(Path('cache'));kind=decision['ar_kind']
    folder=Path('cache/association/main')/kind;info=json.loads((folder/'manifest.json').read_text())
    with (folder/'model.pkl').open('rb') as f:model=pickle.load(f)
    users={u:n for n,u in enumerate(info['model_users'])};items={int(i):n for n,i in enumerate(info['model_items'])}
    _,meta,z=load_dataset('main');ui={u:n for n,u in enumerate(meta['users'])}
    qitems={int(i):n for n,i in enumerate(meta['items'])}
    with np.load(folder/'scores.npz') as q:ar_score=q['score']
    standalone=read_rows(out/'private'/f'main_association_{kind}.csv')
    def lookup(ids):
        return {int(i):{'item':int(i),'title':t,'category':cat} for i,t,cat in c.execute('''SELECT item,
            first(title ORDER BY ss,sid),first(category ORDER BY ss,sid) FROM base.airings WHERE item IN ('''+','.join(map(str,sorted(ids)))+') GROUP BY item').fetchall()}
    rule_examples=[]
    for useful in [True,False]:
        for u,r in sorted(standalone.items(),key=lambda pair:(-float(pair[1]['ap5']) if useful else float(pair[1]['ap5']),pair[0])):
            truth=set(map(int,r['truth'].split()));target=int(r['predictions'].split()[0])
            if not truth or z['cold'][ui[u]] or u not in users or target not in items:continue
            if (target in truth)!=useful or ar_score[ui[u],qitems[target]]<=0:continue
            profile=model.profiles().getrow(users[u]);j=items[target]
            weights=model.similarity[profile.indices,j].toarray().ravel()
            contributions=profile.data*weights
            if contributions.max()<=0:continue
            seed=int(profile.indices[np.argmax(contributions)]);antecedent=info['model_items'][seed]
            metadata=lookup({antecedent,target})
            rule_examples.append({'user_id':u,'case':'useful' if useful else 'misleading',
                'antecedent':metadata[antecedent],'consequent':metadata[target],
                'shared_distinct_users':int(model.co[seed,j]),'antecedent_distinct_users':int(model.support[seed]),
                'consequent_distinct_users':int(model.support[j]),
                'training_user_baskets':len(info['model_users']),
                'consequent_basket_prevalence':float(model.support[j]/len(info['model_users'])),
                'empirical_lift':float((model.co[seed,j]/model.support[seed])/(model.support[j]/len(info['model_users']))),
                'empirical_confidence':float(model.co[seed,j]/model.support[seed]),
                'smoothed_rule_weight':float(model.similarity[seed,j]),
                'user_seed_mass':float(profile.data[np.argmax(contributions)]),
                'contribution_to_score':float(contributions.max()),'total_rule_score':float(ar_score[ui[u],qitems[target]]),
                'future_top5_relevant':target in truth,'future_target':sorted(truth),
                'rule_is_not_causal_or_a_future_probability':True,'posthoc_diagnostic_example':True})
            break
    assert len(rule_examples)==2
    save(out/'private/association_rule_examples.json',rule_examples)
    save(out/'rule_summary.json',[{k:v for k,v in r.items() if k!='user_id'} for r in rule_examples])
    # Actual corrected ordering and regression from experiment A, not cherry-picked
    # as representative performance. Include complete before-cutoff history.
    candidates=[r for r in pairs if r['window']=='main' and r['method']=='association']
    chosen=[(max(candidates,key=lambda r:(r['delta'],r['user_id']))['user_id'],'largest association-integration correction'),
            (min(candidates,key=lambda r:(r['delta'],r['user_id']))['user_id'],'largest association-integration regression')]
    old=baseline('main');new=read_rows(out/'private/main_association.csv');examples=[]
    for u,reason in chosen:
        hist=c.execute('''WITH runs AS(SELECT e.sid,a.item,a.duration,greatest(e.a,?) a,least(e.b,?) b
            FROM continuous_events e JOIN base.airings a USING(sid) WHERE e.user_id=? AND a.item>0 AND a.duration>0 AND e.a<? AND e.b>?),
            per_airing AS(SELECT sid,item,duration,max(b-a) AS watch_seconds FROM runs GROUP BY sid,item,duration)
            SELECT item,count(*) frequency50,count(*) FILTER(WHERE 5*watch_seconds>=4*duration) frequency80
            FROM per_airing WHERE 2*watch_seconds>=duration GROUP BY item ORDER BY frequency50 DESC,item''',[LOW,SPLIT,u,SPLIT,LOW]).fetchdf()
        a=list(map(int,old[u]['predictions'].split()));b=list(map(int,new[u]['predictions'].split()));truth=set(map(int,old[u]['truth'].split()))
        names=lookup(set(hist.item)|set(a)|set(b)|truth)
        exact={'v5':arithmetic(truth,a),'association_integration':arithmetic(truth,b)}
        assert abs(exact['v5']['ap5']-float(old[u]['ap5']))<1e-12 and abs(exact['association_integration']['ap5']-float(new[u]['ap5']))<1e-12
        examples.append({'user_id':u,'reason':reason,'rejected_candidate':True,'history_cutoff':SPLIT,
            'history':[names[int(r.item)]|{'frequency50':int(r.frequency50),'frequency80':int(r.frequency80)} for r in hist.itertuples()],
            'v5_top5':[names[i] for i in a],'association_top5':[names[i] for i in b],'truth':[names[i] for i in sorted(truth)],
            'exact_ap5':exact,'executor':'AI assistant; no student verification claimed'})
    save(out/'private/integration_examples.json',examples)
    save(out/'integration_example_summary.json',[{k:v for k,v in r.items() if k not in ['user_id','history']} for r in examples])
    cases=json.loads((out/'private/bpr_actual_cases.json').read_text())
    save(out/'bpr_case_summary.json',[{'alias':f'Case-{n+1}',**{k:v for k,v in r.items() if k not in ['user_id','flat_feature_row']}}
         for n,r in enumerate(cases)])
    original=json.loads(Path('artifacts/stage6/private/user_examples.json').read_text())
    save(out/'private/original_two_method_comparisons.json',[r for r in original if r['reason']=='same Stage-2 representative'])
    check_evidence(frozen);c.close()
    save(out/'conclusion.json',{'selected':'V5','v8_created':False,'kaggle_requests':0,
        'diagnosis':'No verified implementation bug; existing training exposure/old features consistent. Measured statistical drift justified B but calibration did not rescue MAP.',
        'source_attribution':PLAN_SOURCE,'required_methods_preserved':'artifacts/stage2/leaderboard.json',
        'reproduction':'assoc_study audit/standalone; assoc_integrate; association_evidence; verify_association'})
    print({'rules':[(r['case'],r['antecedent']['item'],r['consequent']['item'],r['shared_distinct_users'],r['smoothed_rule_weight']) for r in rule_examples],
        'examples':[(r['reason'],r['exact_ap5']['v5']['ap5'],r['exact_ap5']['association_integration']['ap5']) for r in examples],
        'relevant_hits':hit_summary})


PLAN_SOURCE='https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/; independently written SciPy implementation'
if __name__=='__main__':main()
