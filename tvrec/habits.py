"""At most three new habit/context ablations against frozen V5. No Kaggle calls.

python -m tvrec.habits audit
python -m tvrec.habits experiment
"""
import argparse
import csv
from datetime import datetime,timezone
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .advanced import metrics,save_predictions
from .core import ap_at_5
from .final_refit import digest,check_evidence
from .group_rank import PARAMETERS,ROUNDS,fit_ranker,labels_for_pool,validate_fold_groups,prediction_lists
from .habit_features import active_stability,schedule_habit_blocks
from .improve import connect,schedule_frames
from .labels import event_frame
from .ltr_experiment import matrix
from .models import measured
from .pipeline import LOW,SPLIT,TEST,save
from .rank_features import qualified_occurrences
from .rerank import PLAN as ROLLING,audit_training_cutoffs,truth_for
from .rerank_data import EARLIER


PLAN={
    'experiments':{
        'active_stability':'Completion persistence relative to user active days/weeks, transition returns and burst concentration; not old raw distinct-day/week counts.',
        'channel_calendar':'Exact interval-overlap channel x weekday x 2-hour completed-viewing profiles, population-channel shrinkage and known future compatibility; not rejected global midpoint habits.',
        'matched_drift':'Change of personalized channel/calendar-matched opportunity rates between history and forecast, gated by individual completions; not raw future volume.'},
    'maximum_experiments':3,'combinations_or_weight_tuning':False,
    'parameters':PARAMETERS|{'lambdarank_truncation_level':5},'rounds':ROUNDS,
    'training_folds':ROLLING['training_folds'],'features_control':'frozen V5 59 columns, same pools and learned preprocessing cutoffs',
    'weights':'unchanged V5 normalized LambdaRank user-task groups; no classifier class weighting',
    'smoothing':{'slot_width_seconds':7200,'population_channel_prior_seconds':3600,'matched_rate_prior_per_day':.1},
    'selection':'Both gains >=0.001 and paired95 lower>0; heavy gain>=0, light and repeat-heavy gains>=-0.003 both; cold lists identical. Else retain V5.',
    'unchanged_target':'>=80% uninterrupted original airing duration; distinct-airing frequency Top5; ID ties; all1260, empty=0',
    'kaggle_or_private_score_use':False}


def freeze(out):
    f=out/'frozen_prior.json'
    if not f.exists():
        files={str(p):{'sha256':digest(p),'bytes':p.stat().st_size}
            for p in Path('artifacts').rglob('*') if p.is_file() and 'stage7' not in p.parts}
        for directory,pattern in [('repository_artifacts','*'),('cache/group_ltr','*.txt'),
                                  ('cache/focused_v4','*.pkl'),('cache/rerank','*.pkl')]:
            for p in Path(directory).glob(pattern):
                if p.is_file():files[str(p)]={'sha256':digest(p),'bytes':p.stat().st_size}
        save(f,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'files':files})
    files=json.loads(f.read_text())['files'];check_evidence(files)
    assert digest('artifacts/final/submission_hybrid_v5.csv')=='249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f'
    return files


def baseline(window):
    with Path(f'artifacts/stage6/private/{window}_lambda_top5.csv').open(newline='') as f:
        return {r['user_id']:r for r in csv.DictReader(f)}


def cache_extra(c,name,out,blocks=None):
    base=Path('cache/rerank')/name;meta=json.loads((base/'features_manifest.json').read_text())
    with np.load(base/'indices.npz') as z:ui=z['user_index'];ii=z['item_index']
    cutoff=meta['signature']['cutoff'];upper=meta['signature']['upper']
    folder=Path('cache/habit_v6')/name;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/('extras.json' if blocks is None else 'extras_'+'_'.join(blocks)+'.json')
    signature={'source':digest('tvrec/habit_features.py'),'base_manifest':digest(base/'features_manifest.json')}
    if marker.exists() and json.loads(marker.read_text())['signature']==signature:
        return folder,json.loads(marker.read_text())
    with measured() as resources:
        occ=qualified_occurrences(event_frame(c,LOW,cutoff),cutoff)
        pu=np.array(meta['users'],dtype=object)[ui];pi=np.array(meta['items'])[ii]
        requested=list(PLAN['experiments']) if blocks is None else blocks
        values={}
        if 'active_stability' in requested:values['active_stability']=active_stability(occ,pu,pi,cutoff)
        if any(k in requested for k in ['channel_calendar','matched_drift']):
            past,future=schedule_frames(c,cutoff,upper)
            past=past.drop_duplicates(['channel','ss','duration','item']);future=future.drop_duplicates(['channel','ss','duration','item'])
            calendar,drift=schedule_habit_blocks(occ,past,future,meta['users'],meta['items'],ui,ii,cutoff,upper)
            if 'channel_calendar' in requested:values['channel_calendar']=calendar
            if 'matched_drift' in requested:values['matched_drift']=drift
        names={}
        for block,cols in values.items():
            names[block]=list(cols);X=np.column_stack([cols[k] for k in cols]).astype(np.float32)
            assert np.isfinite(X).all();np.save(folder/(block+'.npy'),X)
    result={'signature':signature,'columns':names,'cutoff':cutoff,'target_end':upper,'resources':resources,
        'last_historical_second':int(occ.b.max()),'qualified_distinct_airings':len(occ),'future_views_used':False,
        'timing_exposure_clipped_to_windows':True}
    save(marker,result)
    return folder,result


def evaluator_evidence(out):
    truth={'A','B','C'}
    cases=[(['A','B','D','C','E'],Fraction(11,12),'(1 + 1 + 0.75)/3'),
           (['D','A','E','B','C'],Fraction(8,15),'(0.5 + 0.5 + 0.6)/3')]
    result=[]
    for pred,expected,expression in cases:
        value=ap_at_5(truth,pred);assert abs(value-float(expected))<1e-12
        result.append({'truth':sorted(truth),'predictions':pred,'arithmetic':expression,'exact_fraction':str(expected),'actual_ap5':value,'passed':True})
    save(out/'instructor_ap_regressions.json',{'source':'Instructor calculations supplied by user',
        'executor':'AI assistant; no student verification claimed','cases':result,'evaluator_unchanged':True})


def audit(out):
    v5=json.loads(Path('artifacts/final/submission_hybrid_v5.manifest.json').read_text())
    features=v5['feature_names'];assert len(features)==59
    forbidden=[f for f in features if f in ['truth','label','user_id','item_id','target_history_share']]
    assert not forbidden
    save(out/'feature_audit.json',{'baseline_features':features,'count':len(features),
        'already_present':['history_log_strong','history_strong_ratio','future_airing_rate','past_airing_rate',
                           'loyal_completion_ratio','loyal_loyalty_complete_popularity'],
        'not_retested':['Stage5 raw distinct days/weeks consistency block','Stage3 generic midpoint/global time habits',
                        'Stage5 generic completion-gated future volume forecast'],
        'new_information':'Relative active-day/week persistence, exact conditional channel/time overlap, and personalized opportunity drift',
        'original_supervision_and_target_unchanged':True,'no_identifier_or_label_feature':True})
    # Directly assess remaining missed labels under frozen V5 full-pool ranks.
    diagnostics=[]
    for window in ['main','earlier']:
        base=Path('cache/rerank')/window;meta=json.loads((base/'features_manifest.json').read_text())
        with np.load(base/'indices.npz') as z:indices={k:z[k] for k in z.files}
        score=np.load(Path('cache/group_ltr')/(window+'_lambda_top5_scores.npy'))
        source=baseline(window);starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
        missed={k:0 for k in ['rank6_20','rank21_50','rank51_200','beyond200_in_pool','absent_pool']}
        for n,u in enumerate(meta['users']):
            lo,hi=starts[n:n+2];order=np.argsort(-score[lo:hi],kind='stable') if not indices['cold'][n] else np.arange(hi-lo)
            ranked=[meta['items'][i] for i in indices['item_index'][lo:hi][order]]
            assert ranked[:5]==list(map(int,source[u]['predictions'].split()))
            positions={i:r for r,i in enumerate(ranked,1)}
            for item in set(map(int,source[u]['truth'].split()))-set(ranked[:5]):
                r=positions.get(item,100000)
                key='rank6_20' if r<=20 else 'rank21_50' if r<=50 else 'rank51_200' if r<=200 else 'beyond200_in_pool' if item in positions else 'absent_pool'
                missed[key]+=1
        diagnostics.append({'window':window,'missed_relevant_items':missed,'original_v5_lists_exactly_reproduced':True})
    save(out/'v5_error_audit.json',diagnostics)
    print(json.dumps(diagnostics),flush=True)


def experiment(c,out):
    audit_training_cutoffs(PLAN['training_folds'],EARLIER)
    rows=[];seg=pd.read_csv('artifacts/stage4/private/v2_v3_user_deltas.csv',dtype={'user_id':str})
    for block in PLAN['experiments']:
        xs=[];ys=[];groups=[];records=[]
        for fold in PLAN['training_folds']:
            X,meta,z,names=matrix(fold['name']);extra,info=cache_extra(c,fold['name'],out)
            assert info['cutoff']==fold['cutoff'] and info['last_historical_second']<=fold['cutoff']
            truth,_=truth_for(c,fold['name'],fold['cutoff'],fold['target_end'],meta['users'])
            y=labels_for_pool(meta,z,truth)
            g=validate_fold_groups(z['user_index'],z['item_index'],np.array(meta['pool_lengths']),y)
            records.append(fold|g)
            xs.append(np.column_stack([X,np.load(extra/(block+'.npy'))]));ys.append(y);groups.extend(meta['pool_lengths'])
        X=np.concatenate(xs);y=np.concatenate(ys);del xs,ys
        with measured() as fit:model=fit_ranker(X,y,groups,parameters=PLAN['parameters'])
        model_file=Path('cache/habit_v6')/(block+'_early.txt');model.save_model(str(model_file))
        del X,y
        import gc;gc.collect()
        for window in ['main','earlier']:
            X,meta,z,names=matrix(window);extra,info=cache_extra(c,window,out)
            assert max(r['target_end'] for r in records)<=meta['signature']['cutoff']
            with measured() as inference:
                X=np.column_stack([X,np.load(extra/(block+'.npy'))]);score=model.predict(X)
                pred=prediction_lists(meta,z,score)
            np.save(extra/(block+'_early_scores.npy'),score)
            source=baseline(window);users=meta['users'];truth={u:set(map(int,source[u]['truth'].split())) for u in users}
            warm={u for n,u in enumerate(users) if not z['cold'][n]}
            delta=np.array([ap_at_5(truth[u],pred[u])-float(source[u]['ap5']) for u in users])
            rng=np.random.default_rng(42);boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)])
            s=seg[seg.window==window].set_index('user_id')
            means={}
            for key,field,value in [('light','history_group','light_under10'),('heavy','history_group','heavy10plus'),('repeat_heavy','repeat_group','repeat_heavy')]:
                index=[n for n,u in enumerate(users) if s.loc[u,field]==value]
                means[key]=float(delta[index].mean())
            row={'experiment':block,'window':window,**metrics(truth,pred,users,warm),
                'gain':float(delta.mean()),'paired_95_ci':np.quantile(boot,[.025,.975]).tolist(),
                'improved':int((delta>0).sum()),'regressed':int((delta<0).sum()),'subgroup_gains':means,
                'fit_resources':fit,'inference_resources':inference,'training_folds':records,
                'base_feature_count':59,'new_feature_count':len(info['columns'][block]),
                'selection_intervals_not_multiplicity_corrected':True}
            rows.append(row);save(out/'results.json',rows)
            save_predictions(out/'private'/f'{window}_{block}.csv',pred,truth,users,warm)
            print(window,block,row['map5'],row['gain'],row['paired_95_ci'],means,flush=True)
        del model,X,score;gc.collect()
    by={(r['experiment'],r['window']):r for r in rows}
    eligible=[k for k in PLAN['experiments'] if all(by[k,w]['gain']>=.001 and by[k,w]['paired_95_ci'][0]>0
        and by[k,w]['subgroup_gains']['heavy']>=0 and by[k,w]['subgroup_gains']['light']>=-.003
        and by[k,w]['subgroup_gains']['repeat_heavy']>=-.003 for w in ['main','earlier'])]
    chosen=max(eligible,key=lambda k:sum(by[k,w]['gain'] for w in ['main','earlier'])) if eligible else 'retain_v5'
    save(out/'decision.json',{'selected':chosen,'eligible':eligible,'rule':PLAN['selection'],
        'experiments':3,'results':rows,'kaggle_requests':0,'target_changed':False})
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['fit_resources','inference_resources','training_folds']}
                  for r in rows]).to_csv(out/'comparison.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['audit','experiment']);args=parser.parse_args()
    out=Path('artifacts/stage7');out.mkdir(exist_ok=True);(out/'private').mkdir(exist_ok=True)
    frozen=freeze(out);p=out/'plan.json'
    if p.exists():assert json.loads(p.read_text())==PLAN
    else:save(p,PLAN)
    with measured() as resources:
        evaluator_evidence(out)
        if args.stage=='audit':audit(out)
        else:
            assert (out/'feature_audit.json').exists()
            c=connect(Path('cache'));experiment(c,out);c.close()
        check_evidence(frozen)
    save(out/f'execution_{args.stage}.json',{'resources':resources,'executor':'AI assistant',
        'source_sha256':{str(p):digest(p) for p in Path('tvrec').glob('*.py')},'kaggle_requests':0})


if __name__=='__main__':main()
