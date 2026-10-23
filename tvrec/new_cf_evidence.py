"""Post-selection diagnosis/examples for the unsuccessful complementary-CF study."""
import csv
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .core import ap_at_5
from .final_refit import digest,check_evidence
from .focused import load_dataset
from .improve import connect
from .new_cf import baseline
from .pipeline import LOW,SPLIT,save
from .rerank_evidence import arithmetic


def read_rows(path):
    with Path(path).open(newline='') as f:return {r['user_id']:r for r in csv.DictReader(f)}


def main():
    out=Path('artifacts/stage8');frozen=json.loads((out/'frozen_prior.json').read_text())['files'];check_evidence(frozen)
    decision=json.loads((out/'decision.json').read_text());assert decision['selected']=='retain_v5'
    old_segments=pd.read_csv('artifacts/stage4/private/v2_v3_user_deltas.csv',dtype={'user_id':str})
    paired=[];segments=[];ranking=[];retrieval_baselines=[]
    for window in ['main','earlier']:
        source=baseline(window)
        base,meta,z=load_dataset(window)
        truth_pairs=sum(len(r['truth'].split()) for r in source.values())
        als_hits=sum(len({meta['items'][i] for i in z['cf'][n,:200]}&set(map(int,source[u]['truth'].split())))
                     for n,u in enumerate(meta['users']))
        existing_hits=0
        offsets=np.r_[0,np.cumsum(meta['pool_lengths'])]
        for n,u in enumerate(meta['users']):
            existing_hits+=len({meta['items'][i] for i in z['item_index'][offsets[n]:offsets[n+1]]}&set(map(int,source[u]['truth'].split())))
        prior_rec=next(r['lambdarank_after_reranking_recall200_micro'] for r in
            json.loads(Path('artifacts/stage6/candidate_ranking_recall.json').read_text()) if r['window']==window)
        retrieval_baselines.append({'window':window,'als_recall200_micro':als_hits/truth_pairs,
            'v5_postrank_recall200_micro':prior_rec,'v5_existing_pool_recall_micro':existing_hits/truth_pairs})
        expanded=Path('cache/new_cf/log_frequency')/window/'expanded200'
        expmeta=json.loads((expanded/'manifest.json').read_text())
        with np.load(expanded/'indices.npz') as q:expidx=q['item_index']
        expstarts=np.r_[0,np.cumsum(expmeta['pool_lengths'])]
        oldstarts=np.r_[0,np.cumsum(meta['pool_lengths'])]
        for variant in ['same_pool_features','expanded200']:
            candidate=read_rows(out/'private'/f'{window}_{variant}.csv')
            rows=[];hits=novel_hit=0
            for n,u in enumerate(meta['users']):
                s=old_segments[(old_segments.window==window)&(old_segments.user_id==u)].iloc[0]
                row={'window':window,'variant':variant,'user_id':u,'v5_ap5':float(source[u]['ap5']),
                    'candidate_ap5':float(candidate[u]['ap5']),'gain':float(candidate[u]['ap5'])-float(source[u]['ap5']),
                    'v5_hit_ranks':source[u]['hit_ranks'],'candidate_hit_ranks':candidate[u]['hit_ranks'],
                    **{k:s[k] for k in ['history_group','repeat_group','channel_group']}}
                rows.append(row);paired.append(row)
                target=set(map(int,source[u]['truth'].split()));pred=set(map(int,candidate[u]['predictions'].split()))
                old_pool={meta['items'][i] for i in z['item_index'][oldstarts[n]:oldstarts[n+1]]}
                hits+=len(target&pred);novel_hit+=len((target&pred)-old_pool)
            df=pd.DataFrame(rows)
            for feature in ['history_group','repeat_group','channel_group']:
                for group,g in df.groupby(feature):
                    segments.append({'window':window,'variant':variant,'segmentation':feature,'group':group,'users':len(g),
                        'v5_map5':float(g.v5_ap5.mean()),'candidate_map5':float(g.candidate_ap5.mean()),'gain':float(g.gain.mean()),
                        'improved':int((g.gain>0).sum()),'regressed':int((g.gain<0).sum())})
            ranking.append({'window':window,'variant':variant,'relevant_top5_hits':hits,
                'correct_top5_programmes_absent_from_original_pool':novel_hit,
                'v5_relevant_top5_hits':sum(len(set(map(int,r['predictions'].split()))&set(map(int,r['truth'].split()))) for r in source.values())})
    pd.DataFrame(paired).to_csv(out/'private/user_deltas.csv',index=False)
    pd.DataFrame(segments).to_csv(out/'segments.csv',index=False)
    save(out/'ranking_quality.json',ranking)
    save(out/'baseline_candidate_recall.json',retrieval_baselines)
    # Scale diagnostics are observations, not proof of the mechanism of failure.
    scale=[]
    for name in ['roll_mar23','roll_apr06','earlier','main']:
        folder=Path('cache/new_cf/log_frequency')/name
        with np.load(folder/'scores.npz') as q:
            norm=q['user_norm'][q['user_seen']]
            values=q['score'][np.isfinite(q['score'])]
        scale.append({'task':name,'user_norm_quantiles':np.quantile(norm,[.1,.5,.9]).tolist(),
                      'ranking_score_quantiles':np.quantile(values,[.1,.5,.9]).tolist()})
    save(out/'latent_scale_diagnostics.json',{'tasks':scale,
        'interpretation':'Different history lengths can shift raw latent scales. Rank/cosine features are also supplied, but no calibration/extra search was run.'})
    main=[r for r in paired if r['window']=='main' and r['variant']=='expanded200']
    chosen=[(max(main,key=lambda r:(r['gain'],r['user_id']))['user_id'],'largest rejected-expansion AP correction'),
            (min(main,key=lambda r:(r['gain'],r['user_id']))['user_id'],'largest rejected-expansion AP regression')]
    original=json.loads(Path('artifacts/stage6/private/user_examples.json').read_text())
    c=connect(Path('cache'));old=baseline('main');new=read_rows(out/'private/main_expanded200.csv');examples=[]
    for user,reason in chosen:
        hist=c.execute('''WITH runs AS(SELECT e.sid,a.item,a.duration,greatest(e.a,?) a,least(e.b,?) b
            FROM continuous_events e JOIN base.airings a USING(sid) WHERE e.user_id=? AND a.item>0 AND a.duration>0 AND e.a<? AND e.b>?),
            unique_airings AS(SELECT sid,item,duration,max(b-a) AS overlap_seconds FROM runs GROUP BY sid,item,duration)
            SELECT item,count(*) frequency50,count(*) FILTER(WHERE 5*overlap_seconds>=4*duration) frequency80
            FROM unique_airings WHERE 2*overlap_seconds>=duration GROUP BY item ORDER BY frequency50 DESC,item''',[LOW,SPLIT,user,SPLIT,LOW]).fetchdf()
        a=list(map(int,old[user]['predictions'].split()));b=list(map(int,new[user]['predictions'].split()));truth=set(map(int,old[user]['truth'].split()))
        ids=set(hist.item)|set(a)|set(b)|truth
        lookup={int(i):{'item':int(i),'title':title,'category':cat} for i,title,cat in c.execute('''SELECT item,
            first(title ORDER BY ss,sid),first(category ORDER BY ss,sid) FROM base.airings WHERE item IN ('''+','.join(map(str,sorted(ids)))+') GROUP BY item').fetchall()}
        exact={'v5':arithmetic(truth,a),'rejected_expansion':arithmetic(truth,b)}
        assert abs(exact['v5']['ap5']-float(old[user]['ap5']))<1e-12 and abs(exact['rejected_expansion']['ap5']-float(new[user]['ap5']))<1e-12
        examples.append({'user_id':user,'reason':reason,'candidate_rejected':True,'history_cutoff':SPLIT,
            'historical_programmes':[lookup[int(r.item)]|{'frequency50':int(r.frequency50),'frequency80':int(r.frequency80)} for r in hist.itertuples()],
            'truth':[lookup[i] for i in sorted(truth)],'v5_top5':[lookup[i] for i in a],
            'candidate_top5':[lookup[i] for i in b],'exact_ap5':exact,
            'executor':'AI assistant, not student verification','posthoc_example_selection':True})
    save(out/'private/rejected_candidate_examples.json',examples)
    save(out/'example_summary.json',[{'alias':f'Example-{n+1}','reason':r['reason'],
        'v5_ap5':r['exact_ap5']['v5']['ap5'],'candidate_ap5':r['exact_ap5']['rejected_expansion']['ap5'],
        'v5_top5':r['v5_top5'],'candidate_top5':r['candidate_top5']} for n,r in enumerate(examples)])
    # Keep the original academically justified users without changing old files.
    save(out/'private/original_representatives.json',[r for r in original if r['reason']=='same Stage-2 representative'])
    gate=json.loads((out/'information_gate.json').read_text())
    save(out/'conclusion.json',{'selected':'V5','new_cf':'implicit BPR, not installed LightFM/WARP',
        'selected_positive_weighting':gate['selected_weighting'],'lightfm_failure_source':'artifacts/sources/new_cf_methodology.json',
        'genuine_new_information':'Selected BPR Top200 adds220/134 relevant pairs outside the old pool; Top50 adds only3/7.',
        'why_retained':'Neither unchanged-pool feature addition nor one expanded-pool follow-up improves both holdouts robustly.',
        'v7_created':False,'kaggle_requests':0,'source_sha256':{str(p):digest(p) for p in Path('tvrec').glob('*.py')},
        'all_cf_mappings_rebuilt_at_own_cutoff':True,'final_refit_not_performed':True})
    c.close();check_evidence(frozen)
    print({'ranking_quality':ranking,'scale':scale,
        'examples':[(r['reason'],r['exact_ap5']['v5']['ap5'],r['exact_ap5']['rejected_expansion']['ap5']) for r in examples]})


if __name__=='__main__':main()
