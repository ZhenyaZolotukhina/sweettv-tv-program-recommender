"""Controlled ranking-only improvements against frozen v1; no Kaggle requests.

python -m tvrec.improve experiment
Only the fixed feature plan is evaluated. Existing ALS factors/scores are reused.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .advanced import metadata, metrics, save_predictions
from .core import ap_at_5
from .final_refit import digest, check_evidence
from .labels import event_frame, target_from_events, training_features
from .models import (ContentProfiles,matrix_from_features,popularity_scores,reciprocal_ranks,
    ranked_predictions,measured)
from .pipeline import LOW,SPLIT,TEST,END,save
from .rank_features import qualified_occurrences,repeat_variants,opportunities,channel_scores,habit_scores


PLAN={
    'selection_uses':'local main and earlier targets only; Kaggle monitoring is not read by experiment code',
    'frozen_ground_truth':'uninterrupted >=80% coverage, distinct-airing frequency Top-5, frequency ties ID ascending',
    'users':1260,'rrf_k':10,'rrf_top':200,'new_component_weight':.1,
    'recency':'frequency * (0.75 + 0.25 * exp(-ln(2)*days_since_last/35))',
    'confidence':'frequency of >=50% airings + frequency of >=80% airings (training only)',
    'opportunity':'repeat frequency * sqrt(clip((future_airings+4)/(past_airings*future_days/past_days+4),0.25,4))',
    'channel':'sqrt user channel frequencies normalized; training channel-conditional programme popularity',
    'habit':'42 weekday/four-hour bins; Laplace-1 user prior; known future airing midpoint distribution',
    'configs':['v1','recency','confidence','opportunity','channel_alone','habit_alone',
               'v1_plus_channel','v1_plus_habit','confidence_opportunity'],
    'robust_selection':'Both windows must improve at least 0.001 absolute MAP; choose largest mean gain, then simpler name. Paired bootstrap reported, not treated as an independent test.'
}


def freeze_prior(out):
    p=out/'frozen_prior.json'
    if not p.exists():
        files={str(f):{'sha256':digest(f),'bytes':f.stat().st_size}
            for f in sorted(Path('artifacts').rglob('*')) if f.is_file()
            and 'stage3' not in f.parts and 'monitoring' not in f.parts}
        for f in sorted(Path('repository_artifacts').glob('*')):
            if f.is_file():
                files[str(f)]={'sha256':digest(f),'bytes':f.stat().st_size}
        save(p,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'files':files})
    frozen=json.loads(p.read_text())['files']
    check_evidence(frozen)
    assert digest('artifacts/final/submission_hybrid_v1.csv')=='d4026ac73d6f940010d7be3f8241731bd5975054fc3127eebd0cc1cfaebfed72'
    return frozen


def connect(cache):
    c=duckdb.connect(str(cache/'stage2.duckdb'),read_only=True)
    c.execute(f"ATTACH '{str((cache/'data.duckdb').resolve())}' AS base (READ_ONLY)")
    c.execute("SET threads=4; SET memory_limit='1GB'")
    return c


def schedule_frames(c,cutoff,upper):
    # Only historical schedule for historical exposure; target schedule for opportunities.
    past=c.execute('SELECT sid,item,channel,ss,duration FROM base.airings WHERE item>0 AND duration>0 AND ss<? AND se>?',[cutoff,LOW]).fetchdf()
    future=c.execute('SELECT sid,item,channel,ss,duration FROM base.airings WHERE item>0 AND duration>0 AND ss<? AND se>?',[upper,cutoff]).fetchdf()
    return past,future


def load_baseline(c,cache,users,items,cutoff,upper,prefix,training):
    target=cache/'improvements'
    target.mkdir(exist_ok=True)
    p=target/f'{prefix}_v1_scores.npz'
    ff=training_features(training,minimum=.5,strength='frequency',half_life=0,cutoff=cutoff)
    if p.exists():
        with np.load(p) as z:
            scores={n:z[n] for n in ['repeat','cf','cbf','popularity']}
        return scores,ff
    repeat=matrix_from_features(ff,users,items).toarray().astype(np.float32)
    pop=popularity_scores(ff,users,items)
    cfg={'minimum':.5,'factors':32,'alpha':10.,'regularization':1.,'iterations':12,'seed':42}
    key=hashlib.sha256(json.dumps({'config':cfg,'window':[LOW,cutoff,upper],
        'model_source':digest('tvrec/models.py'),
        'sessions':json.loads((cache/'stage2_sessions_manifest.json').read_text())},sort_keys=True).encode()).hexdigest()[:20]
    existing=cache/f'als_{key}.npz'
    if not existing.exists():
        raise RuntimeError('Expected frozen ALS cache absent; refuse unnecessary retraining')
    with np.load(existing) as z:
        cf=z['scores']
    model_items=sorted(set(int(i) for i in training.item)|set(items))
    idx={i:n for n,i in enumerate(model_items)}
    mt,ma=metadata(c,cutoff,upper,model_items)
    cbf=ContentProfiles(rich=True,max_features=40000).fit_items(mt,ma)
    with threadpool_limits(limits=1):
        content,_=cbf.score(matrix_from_features(ff,users,model_items),[idx[i] for i in items])
    scores={'repeat':repeat,'cf':cf,'cbf':content,'popularity':pop}
    np.savez_compressed(p,**scores)
    return scores,ff


def ranking_features(c,training,users,items,cutoff,upper):
    with measured() as timing:
        occurrences=qualified_occurrences(training,cutoff)
        variants,pairs=repeat_variants(occurrences,users,items,cutoff)
        past,future=schedule_frames(c,cutoff,upper)
        multiplier,_,_=opportunities(past,future,items,(cutoff-LOW)/86400,(upper-cutoff)/86400)
        variants['opportunity']=variants['frequency']*multiplier[None,:]
        variants['confidence_opportunity']=variants['confidence']*multiplier[None,:]
        variants['channel']=channel_scores(occurrences,users,items)
        with threadpool_limits(limits=1):
            variants['habit']=habit_scores(occurrences,future,users,items)
    details={'timing':timing,'qualified_airings':len(occurrences),'training_last_observation':int(training.b.max()),
        'cutoff':cutoff,'schedule_feature_target_end':upper,'future_user_features':False,
        'opportunity_multiplier_min':float(multiplier.min()),'opportunity_multiplier_max':float(multiplier.max())}
    return variants,pairs,details


def fused_variants(scores,variants,fallback):
    ranks={name:reciprocal_ranks(values,fallback,top=200,k=10) for name,values in scores.items()}
    # Exact order matches frozen v1 arithmetic.
    v1=ranks['repeat']+.1*ranks['cf']+.1*ranks['cbf']+.1*ranks['popularity']
    yield 'v1',v1
    for name in ['recency','confidence','opportunity']:
        r=reciprocal_ranks(variants[name],fallback,200,10)
        yield name,r+.1*ranks['cf']+.1*ranks['cbf']+.1*ranks['popularity']
    yield 'channel_alone',variants['channel']
    yield 'habit_alone',variants['habit']
    channel=reciprocal_ranks(variants['channel'],fallback,200,10)
    habit=reciprocal_ranks(variants['habit'],fallback,200,10)
    yield 'v1_plus_channel',v1+.1*channel
    yield 'v1_plus_habit',v1+.1*habit
    r=reciprocal_ranks(variants['confidence_opportunity'],fallback,200,10)
    yield 'confidence_opportunity',r+.1*ranks['cf']+.1*ranks['cbf']+.1*ranks['popularity']


def window_experiment(c,cache,out,users,cutoff,upper,prefix):
    private=out/'private'
    private.mkdir(exist_ok=True)
    if prefix=='main':
        items=sorted(i for i, in c.execute('SELECT DISTINCT item FROM base.airings WHERE period=21 AND item>0 AND duration>0 AND ss<? AND se>?',[upper,cutoff]).fetchall())
        truth=json.loads(Path('artifacts/stage2/private/truth_versions.json').read_text())['official_ge80_top5']
        truth={u:set(v) for u,v in truth.items()}
        frozen_path=Path('artifacts/stage2/private/main_fusion_19.csv')
        expected=.37363117283950614
    else:
        items=sorted(i for i, in c.execute('SELECT DISTINCT item FROM base.airings WHERE item>0 AND duration>0 AND ss<? AND se>?',[upper,cutoff]).fetchall())
        truth,_,_=target_from_events(event_frame(c,cutoff,upper))
        frozen_path=Path('artifacts/stage2/private/earlier_fusion_0.csv')
        expected=.38793562610229276
    training=event_frame(c,LOW,cutoff)
    warm=set(training.loc[training.b-training.a>=60,'user_id'])
    scores,ff=load_baseline(c,cache,users,items,cutoff,upper,prefix,training)
    fallback=np.lexsort((np.array(items),-scores['popularity'][0]))
    variants,pairs,feature_details=ranking_features(c,training,users,items,cutoff,upper)
    np.testing.assert_array_equal(variants['frequency'],scores['repeat'])
    baseline_pred=None
    board=[]
    all_preds={}
    contributions={}
    for name,fused in fused_variants(scores,variants,fallback):
        with measured() as timing:
            pred=ranked_predictions(fused,users,items,fallback,set(users)-warm)
            result={'window':prefix,'name':name,**metrics(truth,pred,users,warm),'ranking_resources':timing}
        aps=np.array([ap_at_5(truth.get(u,()),pred[u]) for u in users])
        if name=='v1':
            assert abs(result['map5']-expected)<1e-12,'Frozen v1 was not reproduced'
            with frozen_path.open(newline='') as f:
                original={r['user_id']:[int(i) for i in r['predictions'].split()] for r in csv.DictReader(f)}
            assert pred==original,'Frozen v1 predictions changed'
            baseline_pred=pred
            baseline_aps=aps
        delta=aps-baseline_aps
        result.update(mean_delta=float(delta.mean()),improved_users=int((delta>0).sum()),worsened_users=int((delta<0).sum()))
        board.append(result)
        all_preds[name]=pred
        contributions[name]=delta
        save_predictions(private/f'{prefix}_{name}.csv',pred,truth,users,warm)
        print(prefix,name,f"MAP={result['map5']:.6f} delta={result['mean_delta']:+.6f}",flush=True)
    save(out/f'{prefix}_results.json',{'features':feature_details,'candidates':len(items),'users':len(users),'results':board})
    # Training-only segments, fixed before reading any per-user AP difference.
    stats=ff[ff.user_id.isin(users)].groupby('user_id').agg(history_items=('item','size'),occurrences=('frequency','sum'))
    save(private/f'{prefix}_history_segments.json',{u:{'history_items':int(stats.loc[u,'history_items']) if u in stats.index else 0,
        'group':'cold' if u not in warm else 'sparse' if u not in stats.index or stats.loc[u,'history_items']<10 else 'regular'} for u in users})
    return board,contributions,all_preds,truth,items,warm


def channel_refinement(c,cache,out,users):
    """Two follow-ups justified by channel gains in BOTH initial windows."""
    plan={'reason':'Channel addition improved both windows; initial main paired CI crossed zero.',
        'extra_weights':[.3,.5],'maximum_additional_configurations':2,
        'no_new_features_or_mf_fits':True,
        'final_rule':'Both gains >=0.001 and both paired-bootstrap 95% lower bounds >0; otherwise retain v1.'}
    save(out/'channel_refinement_plan.json',plan)
    rows=[]
    for prefix,cutoff,upper in [('main',SPLIT,TEST),
            ('earlier',int(np.datetime64('2020-04-13','s').astype('int64')),SPLIT)]:
        training=event_frame(c,LOW,cutoff)
        warm=set(training.loc[training.b-training.a>=60,'user_id'])
        if prefix=='main':
            items=sorted(i for i, in c.execute('SELECT DISTINCT item FROM base.airings WHERE period=21 AND item>0 AND duration>0 AND ss<? AND se>?',[upper,cutoff]).fetchall())
            truth=json.loads(Path('artifacts/stage2/private/truth_versions.json').read_text())['official_ge80_top5']
            truth={u:set(v) for u,v in truth.items()}
        else:
            items=sorted(i for i, in c.execute('SELECT DISTINCT item FROM base.airings WHERE item>0 AND duration>0 AND ss<? AND se>?',[upper,cutoff]).fetchall())
            truth,_,_=target_from_events(event_frame(c,cutoff,upper))
        scores,_=load_baseline(c,cache,users,items,cutoff,upper,prefix,training)
        fallback=np.lexsort((np.array(items),-scores['popularity'][0]))
        with measured() as feature_timing:
            channels=channel_scores(qualified_occurrences(training,cutoff),users,items)
            channel_rank=reciprocal_ranks(channels,fallback,200,10)
        ranks={k:reciprocal_ranks(scores[k],fallback,200,10) for k in ['repeat','cf','cbf','popularity']}
        v1=ranks['repeat']+.1*ranks['cf']+.1*ranks['cbf']+.1*ranks['popularity']
        base=ranked_predictions(v1,users,items,fallback,set(users)-warm)
        base_ap=np.array([ap_at_5(truth.get(u,()),base[u]) for u in users])
        for weight in plan['extra_weights']:
            name=f'v1_plus_channel_{weight:g}'
            with measured() as timing:
                predictions=ranked_predictions(v1+weight*channel_rank,users,items,fallback,set(users)-warm)
            delta=np.array([ap_at_5(truth.get(u,()),predictions[u]) for u in users])-base_ap
            rng=np.random.default_rng(42)
            bootstrap=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)])
            row={'window':prefix,'name':name,'channel_weight':weight,**metrics(truth,predictions,users,warm),
                'mean_delta':float(delta.mean()),'improved_users':int((delta>0).sum()),
                'worsened_users':int((delta<0).sum()),
                'bootstrap_95_percentile_interval':np.quantile(bootstrap,[.025,.975]).tolist(),
                'feature_resources':feature_timing,'ranking_resources':timing,
                'selection_optimism_not_corrected':True}
            save_predictions(out/'private'/f'{prefix}_{name}.csv',predictions,truth,users,warm)
            rows.append(row)
            print(prefix,name,f"MAP={row['map5']:.6f}, delta={row['mean_delta']:+.6f}, CI={row['bootstrap_95_percentile_interval']}",flush=True)
    save(out/'channel_refinement_results.json',rows)
    original=[r for window in ['main','earlier'] for r in json.loads((out/f'{window}_results.json').read_text())['results']]
    original_ci={(r['window'],r['name']):r for r in json.loads((out/'paired_bootstrap.json').read_text())}
    considered=[r|{'channel_weight':.1,'bootstrap_95_percentile_interval':original_ci[r['window'],r['name']]['bootstrap_95_percentile_interval']}
        for r in original if r['name']=='v1_plus_channel']+rows
    by={(r['window'],r['channel_weight']):r for r in considered}
    qualified=[w for w in [.1,.3,.5] if all(by[window,w]['mean_delta']>=.001
        and by[window,w]['bootstrap_95_percentile_interval'][0]>0 for window in ['main','earlier'])]
    weight=max(qualified,key=lambda w:sum(by[window,w]['mean_delta'] for window in ['main','earlier'])) if qualified else None
    selection={'selected':'v1' if weight is None else 'v1_plus_channel',
        'channel_weight':weight,'qualified_weights':qualified,'rule':plan['final_rule'],
        'kaggle_scores_used':False,'kaggle_requests':0,
        'main':by['main',weight] if weight is not None else next(r for r in original if r['window']=='main' and r['name']=='v1'),
        'earlier':by['earlier',weight] if weight is not None else next(r for r in original if r['window']=='earlier' and r['name']=='v1')}
    save(out/'robust_selection.json',selection)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['feature_resources','ranking_resources']} for r in original+rows]).to_csv(out/'complete_comparison.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['experiment','refine'])
    parser.add_argument('--cache',type=Path,default=Path('cache'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/stage3'))
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    frozen=freeze_prior(args.out)
    plan=args.out/'experiment_plan.json'
    if plan.exists():
        assert json.loads(plan.read_text())==PLAN
    else:
        save(plan,PLAN)
    with measured() as timing:
        c=connect(args.cache)
        users=[u for u, in c.execute('SELECT user_id FROM base.template ORDER BY source_row').fetchall()]
        if args.stage=='refine':
            with measured() as refinement_timing:
                channel_refinement(c,args.cache,args.out,users)
                c.close()
                check_evidence(frozen)
            save(args.out/'refinement_execution.json',{'resources':refinement_timing,'new_als_fits':0,
                'source_sha256':{str(p):digest(p) for p in Path('tvrec').glob('*.py')}})
            return
        main_results=window_experiment(c,args.cache,args.out,users,SPLIT,TEST,'main')
        earlier_cut=int(np.datetime64('2020-04-13','s').astype('int64'))
        earlier_results=window_experiment(c,args.cache,args.out,users,earlier_cut,SPLIT,'earlier')
        all_rows=main_results[0]+earlier_results[0]
        pd.DataFrame([{k:v for k,v in r.items() if k!='ranking_resources'} for r in all_rows]).to_csv(args.out/'comparison.csv',index=False)
        by={(r['window'],r['name']):r for r in all_rows}
        qualified=[n for n in PLAN['configs'] if n not in ['v1','channel_alone','habit_alone']
            and by['main',n]['mean_delta']>=.001 and by['earlier',n]['mean_delta']>=.001]
        chosen=max(qualified,key=lambda n:((by['main',n]['mean_delta']+by['earlier',n]['mean_delta'])/2,n)) if qualified else 'v1'
        uncertainty=[]
        for prefix,result in [('main',main_results),('earlier',earlier_results)]:
            for name in PLAN['configs']:
                delta=result[1][name]
                rng=np.random.default_rng(42)
                bootstrap=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)])
                uncertainty.append({'window':prefix,'name':name,'mean_delta':float(delta.mean()),
                    'bootstrap_95_percentile_interval':np.quantile(bootstrap,[.025,.975]).tolist(),
                    'selection_optimism_not_corrected':True})
        save(args.out/'paired_bootstrap.json',uncertainty)
        save(args.out/'selection.json',{'selected':chosen,'qualified_candidates':qualified,
            'rule':PLAN['robust_selection'],'main':by['main',chosen],'earlier':by['earlier',chosen],
            'kaggle_scores_used_for_selection':False,'kaggle_requests':0})
        c.close()
        check_evidence(frozen)
    save(args.out/'execution.json',{'resources':timing,'platform':platform.platform(),
        'source_sha256':{str(p):digest(p) for p in Path('tvrec').glob('*.py')},
        'prior_files_unchanged':True,'executor':'AI assistant','new_als_fits':0})


if __name__=='__main__':
    main()
