"""Final focused investigation: audit, three single-block V3 ablations, select.

python -m tvrec.focused audit
python -m tvrec.focused experiment
No previous feature caches, models, results or submissions are overwritten.
"""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .advanced import metrics,save_predictions
from .core import ap_at_5
from .final_refit import digest,check_evidence
from .focused_features import consistency_features,loyalty_features,opportunity_features
from .improve import connect,schedule_frames
from .labels import event_frame
from .models import measured
from .pipeline import LOW,SPLIT,TEST,END,save
from .rank_features import qualified_occurrences
from .rerank import PLAN as V3_PLAN,audit_training_cutoffs,truth_for,query_predictions
from .rerank_data import EARLIER


PLAN={
    'experiments':{
        'consistency':'Distinct >=50% and >=80% observed days/weeks, completed-week coverage, last completion age; sustained completion separates repeats from bursts/one-offs.',
        'loyalty':'Raw and watch-duration channel loyalty, channel completion/intensity and >=80% conditional programme popularity; differs from existing sqrt/soft-count channel score.',
        'opportunity':'Personally completed-event yield per historical broadcast and reliability-gated expected future completions/timing; future air count alone never boosts unseen items.'},
    'maximum_experiments':3,'combinations_or_weight_search':False,
    'model_parameters':V3_PLAN['model'],'training_folds':V3_PLAN['training_folds'],
    'output_policy':'unchanged V3 equal V2/GB reciprocal-rank fusion within unchanged pool; common cold fallback',
    'primary_target_unchanged':True,'all_users':1260,
    'selection':'Main gain >=0.002, earlier >=0.001, paired lower95>0 both; light-history subgroup gain >=-0.003 both; no reversal under frequency-tie sensitivity. Otherwise retain V3.',
    'kaggle_scores_read_for_selection':False}


def freeze(out):
    p=out/'frozen_prior.json'
    if not p.exists():
        files={str(f):{'sha256':digest(f),'bytes':f.stat().st_size}
            for f in sorted(Path('artifacts').rglob('*')) if f.is_file() and 'stage5' not in f.parts}
        for f in Path('repository_artifacts').glob('*'):
            if f.is_file():files[str(f)]={'sha256':digest(f),'bytes':f.stat().st_size}
        for f in Path('cache/rerank').glob('*.pkl'):
            files[str(f)]={'sha256':digest(f),'bytes':f.stat().st_size}
        save(p,{'files':files,'created_at_utc':datetime.now(timezone.utc).isoformat()})
    frozen=json.loads(p.read_text())['files'];check_evidence(frozen)
    assert digest('artifacts/final/submission_hybrid_v3.csv')=='5ebb58dfb26662ea2b10bc3ab0f2e5641f2b846f5cea1a10b604e0d0d9d7c304'
    return frozen


def load_dataset(name):
    folder=Path('cache/rerank')/name
    meta=json.loads((folder/'features_manifest.json').read_text())
    with np.load(folder/'indices.npz') as z:indices={k:z[k] for k in z.files}
    return folder,meta,indices


def original_rows(window):
    with Path(f'artifacts/stage4/private/{window}_v2_supervised_rrf_equal.csv').open(newline='') as f:
        return {r['user_id']:r for r in csv.DictReader(f)}


def labels_audit(c,out):
    results=[]
    users=[u for u, in c.execute('SELECT user_id FROM base.template ORDER BY source_row').fetchall()]
    for window,lo,hi in [('main',SPLIT,TEST),('earlier',EARLIER,SPLIT)]:
        # Independent SQL construction, not a changed label definition.
        pairs=c.execute('''SELECT e.user_id,a.item,count(DISTINCT e.sid) frequency,
            max(least(e.b,?)) AS last_second FROM continuous_events e JOIN base.airings a USING(sid)
            WHERE a.item>0 AND a.duration>0 AND e.a<? AND e.b>?
            AND 5*(least(e.b,?)-greatest(e.a,?))>=4*a.duration
            GROUP BY e.user_id,a.item''',[hi,hi,lo,hi,lo]).fetchdf()
        ordered=pairs.sort_values(['user_id','frequency','item'],ascending=[True,False,True]).groupby('user_id').head(5)
        truth={u:set(g.item) for u,g in ordered.groupby('user_id')}
        alternative=pairs.sort_values(['user_id','frequency','last_second','item'],ascending=[True,False,False,True]).groupby('user_id').head(5)
        alt={u:set(g.item) for u,g in alternative.groupby('user_id')}
        source=original_rows(window)
        assert all(truth.get(u,set())==set(map(int,source[u]['truth'].split())) for u in users)
        folder,meta,_=load_dataset(window)
        assert all(set(truth.get(u,()))<=set(meta['items']) for u in users)
        counts=[len(truth.get(u,())) for u in users]
        result={'window':window,'target_pairs':sum(counts),'zero_users':counts.count(0),
            'fewer_than_five_users':sum(n<5 for n in counts),'exactly_five_users':counts.count(5),
            'original_targets_exactly_reproduced':True,'users':len(users),'available_candidates':len(meta['items']),
            'changed_users_recency_tie_sensitivity':sum(truth.get(u,set())!=alt.get(u,set()) for u in users)}
        results.append(result)
        save(out/'private'/f'{window}_tie_sensitivity_truth.json',{u:sorted(alt.get(u,set())) for u in users})
    duplicate=c.execute('''SELECT count(*) FROM (SELECT channel,ss,se,item FROM base.airings
        WHERE item>0 GROUP BY channel,ss,se,item HAVING count(*)>1)''').fetchone()[0]
    save(out/'target_audit.json',{'windows':results,'coverage':'>=80% of original programme duration in one uninterrupted run',
        'frequency_unit':'distinct user-airing ID; no accumulated separated visits',
        'tie_order':'frequency descending, programme ID ascending; unchanged',
        'empty_truth_ap':0,'denominator':'all1260, including empty truth','labelled_physical_schedule_duplicate_keys':duplicate,
        'id_mapping':'Official page explicitly confirms vsetv_id == channel_id; no guessed bridge.',
        'session_policy':'Existing deduplicated single-channel timeline and disjoint conflicting-airing policy retained.',
        'ambiguity':'Official target-frequency tie/session/device details incomplete. Recency ties sensitivity only; no target redesign.'})


def diagnose(c,out):
    summaries=[];cases=[];redundancy=[]
    for window in ['main','earlier']:
        folder,meta,z=load_dataset(window)
        X=np.load(folder/'X.npy',mmap_mode='r')
        p=np.load(folder/'early_model_probabilities.npy')
        source=original_rows(window)
        offsets=np.r_[0,np.cumsum(meta['pool_lengths'])]
        fi=meta['feature_names'].index('history_log_frequency');si=meta['feature_names'].index('history_log_strong')
        pos_total={k:0 for k in [20,50,200]}
        missed={k:0 for k in ['rank6_20','rank21_50','rank51_200','rank201_plus_in_pool','absent_pool']}
        observations=[];oneoff_cases=[]
        for n,u in enumerate(meta['users']):
            lo,hi=offsets[n:n+2];ids=z['item_index'][lo:hi];prob=p[lo:hi]
            order=np.argsort(-prob,kind='stable');rank=np.empty(len(order),int);rank[order]=np.arange(1,len(order)+1)
            fused=1/(10+np.arange(1,len(order)+1))+1/(10+rank)
            ranked=np.argsort(-fused,kind='stable') if not z['cold'][n] else np.arange(len(order))
            predicted=[meta['items'][ids[j]] for j in ranked[:5]]
            assert predicted==list(map(int,source[u]['predictions'].split()))
            truth=set(map(int,source[u]['truth'].split()))
            frequencies=np.rint(np.expm1(X[lo:hi,fi])).astype(int)
            strong=np.rint(np.expm1(X[lo:hi,si])).astype(int)
            row_for_item={meta['items'][i]:j for j,i in enumerate(ids)}
            pool_rank={meta['items'][ids[j]]:r for r,j in enumerate(ranked,1)}
            for k in [20,50,200]:pos_total[k]+=sum(pool_rank.get(i,100000)<=k for i in truth)
            missed_repeats=[]
            for item in truth-set(predicted):
                r=pool_rank.get(item,100000)
                key='rank6_20' if r<=20 else 'rank21_50' if r<=50 else 'rank51_200' if r<=200 else 'rank201_plus_in_pool' if item in row_for_item else 'absent_pool'
                missed[key]+=1
                if item in row_for_item and strong[row_for_item[item]]>=2:
                    missed_repeats.append({'item':item,'rank':r,'history50':int(frequencies[row_for_item[item]]),'history80':int(strong[row_for_item[item]])})
            wrong_oneoffs=[]
            for j in ranked[:5]:
                item=meta['items'][ids[j]]
                observations.append({'window':window,'user_id':u,'item':item,'relevant':item in truth,
                    'history50':int(frequencies[j]),'history80':int(strong[j]),'cold':bool(z['cold'][n])})
                if not z['cold'][n] and item not in truth and frequencies[j]<=1:
                    wrong_oneoffs.append({'item':item,'history50':int(frequencies[j]),'history80':int(strong[j])})
            if missed_repeats and wrong_oneoffs:
                oneoff_cases.append({'user_id':u,'v3_ap5':float(source[u]['ap5']),
                    'wrong_oneoffs':wrong_oneoffs,'missed_completed_repeats':missed_repeats})
        df=pd.DataFrame(observations)
        df['frequency_group']=np.where(df.history50<=1,'zero_or_one','sustained_at_least_two')
        precision=[{'group':g,'rows':len(v),'precision':float(v.relevant.mean()),'wrong':int((~v.relevant).sum())}
                   for g,v in df[~df.cold].groupby('frequency_group')]
        denom=sum(len(r['truth'].split()) for r in source.values())
        summaries.append({'window':window,'micro_recall_after_v3':{str(k):v/denom for k,v in pos_total.items()},
            'misses':missed,'warm_top5_precision_by_training_frequency':precision,
            'users_wrong_oneoff_above_missed_completed_repeat':len(oneoff_cases)})
        cases.extend({'window':window,**r} for r in sorted(oneoff_cases,key=lambda r:(r['v3_ap5'],r['user_id']))[:2])
        # Correlation is descriptive, not a feature-removal experiment.
        names=['history_log_frequency','history_rate','history_log_strong','history_log_seconds',
               'repeat_score','repeat_reciprocal_rank','v2_score','v2_reciprocal_rank','channel_score','popularity_score']
        sample=np.arange(0,len(X),max(1,len(X)//40000))
        data=pd.DataFrame(np.asarray(X[sample])[:,[meta['feature_names'].index(k) for k in names]],columns=names)
        data=data[data.history_log_frequency>0]
        corr=data.corr(method='spearman')
        for a in range(len(names)):
            for b in range(a+1,len(names)):
                if abs(corr.iloc[a,b])>=.9:
                    redundancy.append({'window':window,'a':names[a],'b':names[b],'spearman':float(corr.iloc[a,b]),'sample_rows_with_history':len(data)})
        pd.DataFrame(observations).to_csv(out/'private'/f'{window}_v3_top5_errors.csv',index=False)
    save(out/'v3_error_analysis.json',{'windows':summaries,'interpretation':'One-off precision and displacement are measured, not assumed to imply dislike.',
        'hypotheses_are_fixed_in_plan':True})
    save(out/'redundancy.json',redundancy)
    save(out/'private/v3_failure_cases.json',cases)
    print(json.dumps(summaries),flush=True)


def extra_dataset(c,name,out,blocks=None):
    base,meta,z=load_dataset(name)
    cutoff=meta['signature']['cutoff'];upper=meta['signature']['upper']
    folder=Path('cache/focused_v4')/name;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/('extras.json' if blocks is None else 'extras_'+'_'.join(blocks)+'.json')
    signature={'feature_source':digest('tvrec/focused_features.py'),'base_manifest':digest(base/'features_manifest.json')}
    if marker.exists() and json.loads(marker.read_text())['signature']==signature:
        return folder,json.loads(marker.read_text())
    with measured() as timing:
        training=event_frame(c,LOW,cutoff);occ=qualified_occurrences(training,cutoff)
        assert occ.b.max()<=cutoff
        pair_users=np.array(meta['users'],dtype=object)[z['user_index']]
        pair_items=np.array(meta['items'])[z['item_index']]
        past,future=schedule_frames(c,cutoff,upper)
        # Known physical broadcast keys, not repeated schedule copies.
        past=past.drop_duplicates(['channel','ss','duration','item'])
        future=future.drop_duplicates(['channel','ss','duration','item'])
        builders={
            'consistency':lambda:consistency_features(occ,pair_users,pair_items,cutoff),
            'loyalty':lambda:loyalty_features(occ,future,pair_users,pair_items),
            'opportunity':lambda:opportunity_features(occ,past,future,meta['users'],meta['items'],
                z['user_index'],z['item_index'],cutoff,upper)}
        columns={}
        for block in (list(builders) if blocks is None else blocks):
            values=builders[block]()
            names=list(values);X=np.column_stack([values[n] for n in names]).astype(np.float32)
            assert np.isfinite(X).all()
            np.save(folder/(block+'.npy'),X);columns[block]=names
    result={'signature':signature,'cutoff':cutoff,'target_end':upper,'columns':columns,'resources':timing,
            'unique_user_airings':len(occ),'last_used_view_second':int(occ.b.max()),'future_user_views_used':False}
    save(marker,result)
    return folder,result


def train_arrays(c,block,out):
    xs=[];ys=[];ws=[];records=[]
    for fold in PLAN['training_folds']:
        base,meta,z=load_dataset(fold['name']);extra,info=extra_dataset(c,fold['name'],out)
        truth,_=truth_for(c,fold['name'],fold['cutoff'],fold['target_end'],meta['users'])
        y=np.array([int(meta['items'][i] in truth[meta['users'][u]]) for u,i in zip(z['user_index'],z['item_index'])],dtype=np.uint8)
        X=np.column_stack([np.load(base/'X.npy'),np.load(extra/(block+'.npy'))])
        weights=1/np.array(meta['pool_lengths'])[z['user_index']]*np.where(y,10.,1.);weights/=weights.mean()
        xs.append(X);ys.append(y);ws.append(weights)
        records.append(fold|{'rows':len(y),'positives':int(y.sum()),'extra_cutoff':info['cutoff']})
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(ws),records


def experiment(c,out):
    audit_training_cutoffs(PLAN['training_folds'],EARLIER)
    rows=[]
    for block in PLAN['experiments']:
        X,y,w,records=train_arrays(c,block,out)
        with measured() as fit:
            with threadpool_limits(limits=2):model=HistGradientBoostingClassifier(**PLAN['model_parameters']).fit(X,y,sample_weight=w)
        folder=Path('cache/focused_v4');folder.mkdir(exist_ok=True)
        with (folder/(block+'_early.pkl')).open('wb') as f:pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)
        del X,y,w
        import gc;gc.collect()
        for window in ['main','earlier']:
            base,meta,z=load_dataset(window);extra,info=extra_dataset(c,window,out)
            assert max(f['target_end'] for f in records)<=meta['signature']['cutoff']
            with measured() as prediction:
                X=np.column_stack([np.load(base/'X.npy'),np.load(extra/(block+'.npy'))])
                with threadpool_limits(limits=2):p=model.predict_proba(X)[:,1]
                predictions=query_predictions(meta,z,p,'v2_supervised_rrf_equal')
            np.save(extra/(block+'_early_probabilities.npy'),p)
            source=original_rows(window);users=meta['users'];truth={u:set(map(int,source[u]['truth'].split())) for u in users}
            warm={u for n,u in enumerate(users) if not z['cold'][n]}
            delta=np.array([ap_at_5(truth[u],predictions[u])-float(source[u]['ap5']) for u in users])
            rng=np.random.default_rng(42);boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)])
            light=json.loads(Path(f'artifacts/stage3/private/{window}_history_segments.json').read_text())
            sparse=[n for n,u in enumerate(users) if u in warm and light[u]['history_items']<10]
            alternative=json.loads((out/'private'/f'{window}_tie_sensitivity_truth.json').read_text())
            alternative_delta=np.mean([ap_at_5(alternative[u],predictions[u])-ap_at_5(alternative[u],list(map(int,source[u]['predictions'].split()))) for u in users])
            row={'experiment':block,'window':window,**metrics(truth,predictions,users,warm),
                'gain':float(delta.mean()),'paired_95_ci':np.quantile(boot,[.025,.975]).tolist(),
                'improved_users':int((delta>0).sum()),'regressed_users':int((delta<0).sum()),
                'light_history_gain':float(delta[sparse].mean()) if sparse else None,
                'recency_tie_sensitivity_gain':float(alternative_delta),'fit_resources':fit,
                'prediction_resources':prediction,'training_folds':records}
            rows.append(row)
            save_predictions(out/'private'/f'{window}_{block}.csv',predictions,truth,users,warm)
            save(out/'results.json',rows)
            print(window,block,row['map5'],row['gain'],row['paired_95_ci'],'light',row['light_history_gain'],'alt',row['recency_tie_sensitivity_gain'],flush=True)
        del model,X,p
        gc.collect()
    by={(r['experiment'],r['window']):r for r in rows}
    eligible=[b for b in PLAN['experiments'] if by[b,'main']['gain']>=.002 and by[b,'earlier']['gain']>=.001
        and all(by[b,w]['paired_95_ci'][0]>0 and by[b,w]['light_history_gain']>=-.003
                and by[b,w]['recency_tie_sensitivity_gain']>=0 for w in ['main','earlier'])]
    chosen=max(eligible,key=lambda b:sum(by[b,w]['gain'] for w in ['main','earlier'])) if eligible else 'retain_v3'
    save(out/'selection.json',{'selected':chosen,'eligible':eligible,'rule':PLAN['selection'],
        'experiments_run':len(PLAN['experiments']),'kaggle_requests':0,'target_changed':False,
        'main':by[chosen,'main'] if eligible else None,'earlier':by[chosen,'earlier'] if eligible else None})
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['fit_resources','prediction_resources','training_folds']}
                  for r in rows]).to_csv(out/'comparison.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['audit','experiment'])
    args=parser.parse_args();out=Path('artifacts/stage5');out.mkdir(parents=True,exist_ok=True);(out/'private').mkdir(exist_ok=True)
    frozen=freeze(out)
    plan=out/'plan.json'
    if plan.exists():assert json.loads(plan.read_text())==PLAN
    else:save(plan,PLAN)
    with measured() as resources:
        c=connect(Path('cache'))
        if args.stage=='audit':labels_audit(c,out);diagnose(c,out)
        else:
            assert (out/'target_audit.json').exists(),'Run audit before experiments'
            experiment(c,out)
        c.close();check_evidence(frozen)
    save(out/f'execution_{args.stage}.json',{'resources':resources,'executor':'AI assistant',
        'source_sha256':{str(p):digest(p) for p in Path('tvrec').glob('*.py')},'kaggle_requests':0})


if __name__=='__main__':main()
