"""One shallow supervised reranker with strictly earlier rolling supervision.

python -m tvrec.rerank diagnose
python -m tvrec.rerank experiment
Never reads Kaggle monitoring scores and never sends network requests.
"""
import argparse
import csv
from datetime import datetime,timezone
from itertools import combinations
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
from .improve import connect
from .labels import event_frame,target_from_events
from .models import measured
from .pipeline import LOW,SPLIT,TEST,END,save
from .rerank_data import EARLIER,cached_feature_dataset,feature_vocabulary


def epoch(date):
    return int(np.datetime64(date,'s').astype('int64'))


PLAN={
    'training_folds':[
        {'name':'roll_mar23','cutoff':epoch('2020-03-23'),'target_end':epoch('2020-04-06')},
        {'name':'roll_apr06','cutoff':epoch('2020-04-06'),'target_end':EARLIER}],
    'model':{'learning_rate':.06,'max_iter':80,'max_leaf_nodes':15,'max_bins':63,
        'min_samples_leaf':50,'l2_regularization':10.,'early_stopping':False,'random_state':42},
    'estimator':'sklearn HistGradientBoostingClassifier, binary log loss; existing dependency',
    'training_weights':'per-query sum=1; positive rows multiplied by 10; no validation class balancing',
    'candidates':'V2 Top200 union positive-score Top50 repeat/ALS/CBF/popularity/channel; never inject truth',
    'prediction_variants':['supervised_only','v2_supervised_rrf_equal'],
    'fit_heldout_targets':False,'kaggle_score_use':False,'new_model_configurations':1,
    'selection_rule':'main gain >=0.003, earlier gain >=0.001, both paired 95% lower bounds >0; else retain V2',
    'cold_policy':'unchanged frozen V2 popularity fallback',
    'training_item_metadata_vocabulary_cutoff':epoch('2020-03-23')}


def freeze(out):
    p=out/'frozen_prior.json'
    if not p.exists():
        files={str(f):{'sha256':digest(f),'bytes':f.stat().st_size}
            for f in sorted(Path('artifacts').rglob('*')) if f.is_file()
            and 'stage4' not in f.parts and 'monitoring' not in f.parts}
        for f in Path('repository_artifacts').glob('*'):
            if f.is_file():
                files[str(f)]={'sha256':digest(f),'bytes':f.stat().st_size}
        save(p,{'files':files,'created_at_utc':datetime.now(timezone.utc).isoformat()})
    files=json.loads(p.read_text())['files']
    check_evidence(files)
    assert digest('artifacts/final/submission_hybrid_v2.csv')=='987f8fde48654ba982c20b36e25bf78017749c58b15d96bccdbc8eca5f7d2671'
    return files


def truth_for(c,name,cutoff,upper,users):
    if name in ['main','earlier']:
        with Path(f'artifacts/stage3/private/{name}_v1_plus_channel_0.5.csv').open(newline='') as f:
            original={r['user_id']:r for r in csv.DictReader(f)}
        return {u:set(map(int,original[u]['truth'].split())) for u in users},original
    truth,_,_=target_from_events(event_frame(c,cutoff,upper))
    return {u:truth.get(u,set()) for u in users},None


def audit_training_cutoffs(folds,prediction_cutoff):
    if any(f['cutoff']<LOW or f['cutoff']>=f['target_end'] or f['target_end']>prediction_cutoff for f in folds):
        raise ValueError('Supervised training target overlaps or follows the prediction cutoff')
    ordered=sorted(folds,key=lambda f:f['cutoff'])
    if any(a['target_end']>b['cutoff'] for a,b in zip(ordered,ordered[1:])):
        raise ValueError('Training target folds overlap; avoid duplicated-label weighting')
    return True


def reference_predictions(meta,z):
    starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
    return {u:[int(meta['items'][i]) for i in z['item_index'][starts[n]:starts[n]+5]]
            for n,u in enumerate(meta['users'])}


def diagnostics(c,cache,out,users,vocab):
    summaries=[];overlaps=[];segments=[];losses=[]
    datasets={}
    for name,cutoff,upper in [('main',SPLIT,TEST),('earlier',EARLIER,SPLIT)]:
        folder,meta=cached_feature_dataset(c,cache,users,name,cutoff,upper,vocab)
        datasets[name]=(folder,meta)
        with np.load(folder/'indices.npz') as loaded:
            z={k:loaded[k] for k in loaded.files}
        truth,original=truth_for(c,name,cutoff,upper,users)
        reference=reference_predictions(meta,z)
        assert all(reference[u]==list(map(int,original[u]['predictions'].split())) for u in users)
        starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
        training=event_frame(c,LOW,cutoff)
        from .rank_features import qualified_occurrences
        occ=qualified_occurrences(training,cutoff)
        user_stats=occ.groupby('user_id').agg(airings=('sid','size'),items=('item','nunique'))
        uc=occ.groupby(['user_id','channel']).size()
        focus=uc.groupby(level=0).max()/uc.groupby(level=0).sum()
        seen={u:set(g.item) for u,g in occ.groupby('user_id')}
        by_segment=[]
        denom=sum(len(truth[u]) for u in users)
        for k in [20,50,200]:
            hits=[]
            for n,u in enumerate(users):
                ids={int(meta['items'][i]) for i in z['v2'][n,:k]}
                hits.append(len(ids&truth[u]))
            summaries.append({'window':name,'generator':'V2','k':k,'micro_recall':sum(hits)/denom,
                'macro_recall_all_1260':float(np.mean([h/len(truth[u]) if truth[u] else 0 for h,u in zip(hits,users)])),
                'truth_pairs':denom})
        pooled_hits=miss_pool=below5=0
        strata={key:0 for key in ['rank6_20','rank21_50','rank51_200','outside_v2_top200']}
        for n,u in enumerate(users):
            target=truth[u]
            top200=[int(meta['items'][i]) for i in z['v2'][n]]
            pool={int(meta['items'][i]) for i in z['item_index'][starts[n]:starts[n+1]]}
            hit=len(pool&target)
            pooled_hits+=hit
            ap=ap_at_5(target,reference[u])
            missing=target-set(reference[u])
            absent=missing-pool
            miss_pool+=len(absent);below5+=len(missing&pool)
            for i in missing:
                rank=top200.index(i)+1 if i in top200 else 10000
                key='rank6_20' if rank<=20 else 'rank21_50' if rank<=50 else 'rank51_200' if rank<=200 else 'outside_v2_top200'
                strata[key]+=1
            h=int(user_stats.loc[u,'items']) if u in user_stats.index else 0
            count=int(user_stats.loc[u,'airings']) if u in user_stats.index else 0
            repeat=1-h/max(count,1)
            row={'window':name,'user_id':u,'v2_ap5':ap,'loss':1-ap if target else None,
                'truth_items':len(target),'pool_recall':hit/len(target) if target else 0,
                'missing_outside_pool':len(absent),'missing_inside_pool':len(missing&pool),
                'history_group':'cold' if z['cold'][n] else 'light_under10' if h<10 else 'heavy10plus',
                'repeat_group':'repeat_heavy' if repeat>=.75 else 'exploratory_history',
                'channel_group':'focused60pct' if focus.get(u,0)>=.6 else 'broad',
                'target_history_share':len(target&seen.get(u,set()))/len(target) if target else None}
            losses.append(row);by_segment.append(row)
        summaries.append({'window':name,'generator':'union_pool','k':None,'micro_recall':pooled_hits/denom,
            'pool_mean_size':float(np.mean(meta['pool_lengths'])),'pool_max_size':max(meta['pool_lengths']),
            'missed_relevant_outside_pool':miss_pool,'missed_relevant_inside_pool_below5':below5,**strata})
        for a,b in combinations(['repeat','cf','cbf','popularity','channel','v2'],2):
            values=[len(set(z[a][n,:50])&set(z[b][n,:50]))/len(set(z[a][n,:50])|set(z[b][n,:50])) for n in range(len(users))]
            overlaps.append({'window':name,'generator_a':a,'generator_b':b,'mean_top50_jaccard':float(np.mean(values)),
                'padding':'positive model ranks then common-popularity padding; cold users common fallback'})
        df=pd.DataFrame(by_segment)
        for feature in ['history_group','repeat_group','channel_group']:
            for group,g in df.groupby(feature):
                segments.append({'window':name,'segmentation':feature,'group':group,'users':len(g),
                    'v2_map5':g.v2_ap5.mean(),'mean_pool_recall_empty_zero':g.pool_recall.mean()})
    pd.DataFrame(summaries).to_csv(out/'candidate_recall.csv',index=False)
    pd.DataFrame(overlaps).to_csv(out/'generator_overlap.csv',index=False)
    pd.DataFrame(segments).to_csv(out/'error_segments.csv',index=False)
    pd.DataFrame(losses).to_csv(out/'private/v2_losses.csv',index=False)
    save(out/'candidate_diagnosis.json',{'summaries':summaries,
        'definition':'Recall: micro over unique target pairs; macro uses all1260, empty=0. MAP always all1260.',
        'posthoc_target_history_share_never_feature':True,
        'decision':'Reranking is worth testing if union micro recall >=0.85 on both windows and ranking misses dominate absent-pool misses.'})
    print(json.dumps(summaries),flush=True)
    return datasets,summaries


def query_predictions(meta,z,probabilities,variant):
    starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
    result={}
    for n,u in enumerate(meta['users']):
        lo,hi=starts[n:n+2]
        ids=z['item_index'][lo:hi]
        if z['cold'][n]:
            result[u]=[int(meta['items'][i]) for i in z['fallback'][:5]]
            continue
        p=probabilities[lo:hi]
        order=np.argsort(-p,kind='stable')
        if variant=='supervised_only':
            chosen=ids[order[:5]]
        else:
            # Equal reciprocal-rank fusion; candidate order is frozen V2 order.
            learned_rank=np.empty(len(order),dtype=int);learned_rank[order]=np.arange(1,len(order)+1)
            score=1/(10+np.arange(1,len(order)+1))+1/(10+learned_rank)
            chosen=ids[np.argsort(-score,kind='stable')[:5]]
        result[u]=[int(meta['items'][i]) for i in chosen]
    return result


def supervised_experiment(c,cache,out,users,vocab,datasets):
    audit_training_cutoffs(PLAN['training_folds'],EARLIER)
    xs=[];ys=[];weights=[];fold_records=[]
    for fold in PLAN['training_folds']:
        folder,meta=cached_feature_dataset(c,cache,users,fold['name'],fold['cutoff'],fold['target_end'],vocab)
        with np.load(folder/'indices.npz') as loaded:
            z={k:loaded[k] for k in loaded.files}
        truth,_=truth_for(c,fold['name'],fold['cutoff'],fold['target_end'],users)
        y=np.array([int(meta['items'][i] in truth[users[u]]) for u,i in zip(z['user_index'],z['item_index'])],dtype=np.uint8)
        X=np.load(folder/'X.npy')
        w=1/np.array(meta['pool_lengths'])[z['user_index']]
        w=w*np.where(y,10.,1.)
        w=w/w.mean()
        xs.append(X);ys.append(y);weights.append(w)
        fold_records.append(fold|{'candidate_rows':len(y),'positive_rows':int(y.sum()),
            'users_with_positive_target':sum(bool(truth[u]) for u in users),'feature_resources':meta['resources'],
            'features_before':fold['cutoff'],'target_labels_end':fold['target_end']})
    X=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(weights)
    with measured() as fit_resources:
        with threadpool_limits(limits=2):
            model=HistGradientBoostingClassifier(**PLAN['model']).fit(X,y,sample_weight=w)
    with (cache/'rerank/model_early_only.pkl').open('wb') as f:
        pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)
    save(out/'supervision_audit.json',{'folds':fold_records,'training_rows':len(y),'positives':int(y.sum()),
        'last_training_label_end':max(f['target_end'] for f in PLAN['training_folds']),
        'earlier_evaluation_starts':EARLIER,'main_evaluation_starts':SPLIT,
        'heldout_labels_used_in_fit':False,'random_internal_validation':False,'fit_resources':fit_resources})
    del X,y,w,xs,ys,weights
    import gc;gc.collect()
    rows=[]
    for name,(folder,meta) in datasets.items():
        cutoff=SPLIT if name=='main' else EARLIER
        upper=TEST if name=='main' else SPLIT
        audit_training_cutoffs(PLAN['training_folds'],cutoff)
        with np.load(folder/'indices.npz') as loaded:
            z={k:loaded[k] for k in loaded.files}
        truth,original=truth_for(c,name,cutoff,upper,users)
        X=np.load(folder/'X.npy',mmap_mode='r')
        with measured() as predict_resources:
            with threadpool_limits(limits=2):
                p=model.predict_proba(X)[:,1]
        np.save(folder/'early_model_probabilities.npy',p)
        base=reference_predictions(meta,z)
        warm={u for n,u in enumerate(users) if not z['cold'][n]}
        for variant in PLAN['prediction_variants']:
            pred=query_predictions(meta,z,p,variant)
            result={'window':name,'variant':variant,**metrics(truth,pred,users,warm),'prediction_resources':predict_resources}
            delta=np.array([ap_at_5(truth[u],pred[u])-ap_at_5(truth[u],base[u]) for u in users])
            rng=np.random.default_rng(42)
            boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)])
            result.update(mean_gain=float(delta.mean()),improved_users=int((delta>0).sum()),
                regressed_users=int((delta<0).sum()),paired_95_ci=np.quantile(boot,[.025,.975]).tolist())
            rows.append(result)
            save_predictions(out/'private'/f'{name}_{variant}.csv',pred,truth,users,warm)
            print(name,variant,result['map5'],result['mean_gain'],result['paired_95_ci'],flush=True)
    save(out/'supervised_results.json',rows)
    pd.DataFrame([{k:v for k,v in r.items() if k!='prediction_resources'} for r in rows]).to_csv(out/'comparison.csv',index=False)
    # Training split gains are descriptive, not an out-of-sample ablation.
    importance=np.zeros(model.n_features_in_)
    for iteration in model._predictors:
        for tree in iteration:
            nodes=tree.nodes
            for node in nodes:
                if not node['is_leaf'] and node['gain']>0:
                    importance[int(node['feature_idx'])]+=node['gain']
    feature_names=next(iter(datasets.values()))[1]['feature_names']
    save(out/'training_split_gain.json',{'interpretation':'training tree split gain only, not causal or holdout permutation importance',
        'features':sorted([{'name':n,'gain':float(g)} for n,g in zip(feature_names,importance)],key=lambda r:-r['gain'])})
    by={(r['window'],r['variant']):r for r in rows}
    eligible=[v for v in PLAN['prediction_variants'] if by['main',v]['mean_gain']>=.003
        and by['earlier',v]['mean_gain']>=.001 and all(by[n,v]['paired_95_ci'][0]>0 for n in ['main','earlier'])]
    selected=max(eligible,key=lambda v:sum(by[n,v]['mean_gain'] for n in ['main','earlier'])) if eligible else 'retain_v2'
    save(out/'selection.json',{'selected':selected,'eligible':eligible,'rule':PLAN['selection_rule'],
        'supervised_training_valid':True,'training_label_end':EARLIER,'kaggle_score_use':False,
        'results':rows,'kaggle_requests':0})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['diagnose','experiment'])
    parser.add_argument('--cache',type=Path,default=Path('cache'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/stage4'))
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'private').mkdir(exist_ok=True)
    frozen=freeze(args.out)
    plan=args.out/'plan.json'
    if plan.exists():
        assert json.loads(plan.read_text())==PLAN
    else:
        save(plan,PLAN)
    with measured() as resources:
        c=connect(args.cache)
        users=[u for u, in c.execute('SELECT user_id FROM base.template ORDER BY source_row').fetchall()]
        vocab=feature_vocabulary(c)
        save(args.out/'metadata_feature_vocabulary.json',vocab)
        if args.stage=='experiment' and (args.out/'candidate_diagnosis.json').exists():
            summaries=json.loads((args.out/'candidate_diagnosis.json').read_text())['summaries']
            datasets={name:cached_feature_dataset(c,args.cache,users,name,cutoff,upper,vocab)
                for name,cutoff,upper in [('main',SPLIT,TEST),('earlier',EARLIER,SPLIT)]}
        else:
            datasets,summaries=diagnostics(c,args.cache,args.out,users,vocab)
        if args.stage=='experiment':
            pool=[r for r in summaries if r['generator']=='union_pool']
            if not all(r['micro_recall']>=.85 and r['missed_relevant_inside_pool_below5']>r['missed_relevant_outside_pool'] for r in pool):
                save(args.out/'selection.json',{'selected':'retain_v2','reason':'Retrieval coverage inadequate for supervised reranking','kaggle_requests':0})
            else:
                supervised_experiment(c,args.cache,args.out,users,vocab,datasets)
        c.close();check_evidence(frozen)
    save(args.out/f'execution_{args.stage}.json',{'resources':resources,'executor':'AI assistant',
        'source_sha256':{str(p):digest(p) for p in Path('tvrec').glob('*.py')},'kaggle_requests':0})


if __name__=='__main__':
    main()
