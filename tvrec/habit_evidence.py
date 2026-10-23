"""Post-experiment evidence for retaining V5, including independent manual cases."""
import csv
from fractions import Fraction
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .final_refit import digest,check_evidence
from .habits import baseline
from .improve import connect
from .pipeline import LOW,SPLIT,save
from .rerank_evidence import arithmetic


def read_rows(path):
    with Path(path).open(newline='') as f:return {r['user_id']:r for r in csv.DictReader(f)}


def main():
    out=Path('artifacts/stage7');frozen=json.loads((out/'frozen_prior.json').read_text())['files'];check_evidence(frozen)
    result=json.loads((out/'results.json').read_text())
    means={name:np.mean([r['gain'] for r in result if r['experiment']==name]) for name in {r['experiment'] for r in result}}
    best=max(means,key=means.get)  # diagnostic only; NOT a selected V6
    decision=json.loads((out/'decision.json').read_text());assert decision['selected']=='retain_v5'
    prior=pd.read_csv('artifacts/stage4/private/v2_v3_user_deltas.csv',dtype={'user_id':str})
    paired=[];segments=[]
    for window in ['main','earlier']:
        old=baseline(window);new=read_rows(out/'private'/f'{window}_{best}.csv')
        values=[]
        for u,r in old.items():
            s=prior[(prior.window==window)&(prior.user_id==u)].iloc[0]
            row={'window':window,'user_id':u,'v5_ap5':float(r['ap5']),'candidate_ap5':float(new[u]['ap5']),
                'gain':float(new[u]['ap5'])-float(r['ap5']),
                'v5_hits':r['hit_ranks'],'candidate_hits':new[u]['hit_ranks'],
                **{k:s[k] for k in ['history_group','repeat_group','channel_group']}}
            values.append(row);paired.append(row)
        df=pd.DataFrame(values)
        for category in ['history_group','repeat_group','channel_group']:
            for group,g in df.groupby(category):
                segments.append({'window':window,'segmentation':category,'group':group,'users':len(g),
                    'v5_map5':float(g.v5_ap5.mean()),'rejected_candidate_map5':float(g.candidate_ap5.mean()),
                    'gain':float(g.gain.mean()),'improved':int((g.gain>0).sum()),'regressed':int((g.gain<0).sum())})
    pd.DataFrame(paired).to_csv(out/'private/paired_user_changes.csv',index=False)
    pd.DataFrame(segments).to_csv(out/'segments.csv',index=False)
    main=[r for r in paired if r['window']=='main']
    chosen=[(max(main,key=lambda r:(r['gain'],r['user_id']))['user_id'],'largest correction by rejected candidate'),
            (min(main,key=lambda r:(r['gain'],r['user_id']))['user_id'],'largest regression by rejected candidate')]
    old=baseline('main');new=read_rows(out/'private'/f'main_{best}.csv')
    meta=json.loads(Path('cache/rerank/main/features_manifest.json').read_text())
    with np.load('cache/rerank/main/indices.npz') as z:idx=z['item_index']
    starts=np.r_[0,np.cumsum(meta['pool_lengths'])];ui={u:n for n,u in enumerate(meta['users'])}
    extra=np.load(Path('cache/habit_v6/main')/(best+'.npy'),mmap_mode='r')
    extra_info=json.loads(Path('cache/habit_v6/main/extras.json').read_text())
    c=connect(Path('cache'));examples=[]
    for user,reason in chosen:
        n=ui[user];lo,hi=starts[n:n+2]
        at={meta['items'][item]:lo+j for j,item in enumerate(idx[lo:hi])}
        a=list(map(int,old[user]['predictions'].split()));b=list(map(int,new[user]['predictions'].split()))
        truth=set(map(int,old[user]['truth'].split()))
        hist=c.execute('''WITH events AS(SELECT e.sid,a.item,a.duration,
            greatest(e.a,?) s,least(e.b,?) t FROM continuous_events e JOIN base.airings a USING(sid)
            WHERE e.user_id=? AND a.item>0 AND a.duration>0 AND e.a<? AND e.b>?),
            unique_events AS(SELECT sid,item,duration,max(t-s) AS seconds_watch FROM events GROUP BY sid,item,duration)
            SELECT item,count(*) frequency50,count(*) FILTER(WHERE 5*seconds_watch>=4*duration) frequency80
            FROM unique_events WHERE 2*seconds_watch>=duration GROUP BY item ORDER BY frequency50 DESC,item''',
            [LOW,SPLIT,user,SPLIT,LOW]).fetchdf()
        ids=set(hist.item)|set(a)|set(b)|truth
        lookup={int(i):{'item':int(i),'title':title,'category':cat} for i,title,cat in c.execute('''SELECT item,
            first(title ORDER BY ss,sid),first(category ORDER BY ss,sid) FROM base.airings WHERE item IN ('''+','.join(map(str,sorted(ids)))+') GROUP BY item').fetchall()}
        calculations={'v5':arithmetic(truth,a),'rejected_candidate':arithmetic(truth,b)}
        assert abs(calculations['v5']['ap5']-float(old[user]['ap5']))<1e-12
        assert abs(calculations['rejected_candidate']['ap5']-float(new[user]['ap5']))<1e-12
        examples.append({'user_id':user,'reason':reason,'candidate':best,'candidate_rejected':True,
            'history_cutoff':SPLIT,'historical_programmes':[lookup[int(r.item)]|{'frequency50':int(r.frequency50),
                'frequency80':int(r.frequency80)} for r in hist.itertuples()],
            'truth':[lookup[i] for i in sorted(truth)],'v5_top5':[lookup[i] for i in a],
            'candidate_top5':[lookup[i] for i in b],'exact_ap5':calculations,
            'new_feature_values':{str(i):dict(zip(extra_info['columns'][best],map(float,extra[at[i]]))) for i in set(a)|set(b)},
            'posthoc_diagnostic_not_training_feature':True,'executor':'AI assistant; not student verification'})
    save(out/'private/rejected_candidate_examples.json',examples)
    save(out/'rejected_candidate_summary.json',[{'alias':f'Example-{n+1}','reason':r['reason'],
        'v5_ap5':r['exact_ap5']['v5']['ap5'],'candidate_ap5':r['exact_ap5']['rejected_candidate']['ap5'],
        'v5_top5':r['v5_top5'],'candidate_top5':r['candidate_top5']} for n,r in enumerate(examples)])
    # Copy genuine already-executed source checks; do not alter historical evidence.
    evidence=out/'private/report_evidence';evidence.mkdir(exist_ok=True)
    copied={}
    for name in ['coverage_witness.json','source_line_checks.json']:
        src=Path('artifacts/stage2/private')/name;dst=evidence/name
        if not dst.exists():shutil.copyfile(src,dst)
        assert digest(src)==digest(dst);copied[name]={'source':str(src),'copy':str(dst),'sha256':digest(dst)}
    original=json.loads(Path('artifacts/stage6/private/user_examples.json').read_text())
    representatives=[r for r in original if r['reason']=='same Stage-2 representative']
    assert len(representatives)==2
    bundle=[]
    for r in representatives:
        truth={t['item'] for t in r['truth']};pred=r['top5']['lambda_top5'];hits=0;steps=[];s=Fraction(0)
        for rank,item in enumerate(pred,1):
            hit=item['item'] in truth
            if hit:hits+=1;s+=Fraction(hits,rank)
            steps.append({'rank':rank,**item,'relevant':hit,'cumulative_hits':hits,
                'precision_term':str(Fraction(hits,rank)) if hit else '0'})
        denominator=min(5,len(truth));exact=s/denominator if denominator else Fraction(0)
        assert abs(float(exact)-r['exact_ap5']['lambda_top5']['ap5'])<1e-12
        bundle.append(r|{'recommended_final_method':'V5','v5_rank_steps':steps,
            'ap5_denominator':denominator,'exact_v5_ap5':str(exact),'student_verification_performed':False})
    save(evidence/'same_two_users_all_methods.json',bundle)
    save(out/'evidence_index.json',{'best_rejected_diagnostic':best,'selected_final':'V5','v6_created':False,
        'source_copies':copied,'representatives':str(evidence/'same_two_users_all_methods.json'),
        'progression_and_private_grading_correction':'artifacts/monitoring/submission_history_grading_correction.json',
        'manual_instruction':'Use original CSV record numbers in source_line_checks; independently compute clipped interval intersection and divide by actual airing duration. For AP, use the five explicit per-rank relevance/precision steps. These are AI checks, not claims of student verification.'})
    c.close();check_evidence(frozen)
    print({'best_rejected':best,'representative_users':len(bundle),
        'examples':[(r['reason'],r['exact_ap5']['v5']['ap5'],r['exact_ap5']['rejected_candidate']['ap5']) for r in examples]})


if __name__=='__main__':main()
