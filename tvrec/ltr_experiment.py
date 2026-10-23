"""One genuine group-aware LambdaRank comparison against frozen V4.

python -m tvrec.ltr_experiment audit
python -m tvrec.ltr_experiment experiment
No data preprocessing, candidate search or Kaggle access.
"""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .advanced import metrics,save_predictions
from .core import ap_at_5
from .final_refit import digest,check_evidence
from .focused import load_dataset
from .group_rank import (PARAMETERS,ROUNDS,lightgbm_module,labels_for_pool,
    validate_fold_groups,fit_ranker,prediction_lists)
from .improve import connect
from .models import measured
from .pipeline import SPLIT,TEST,save
from .rerank import PLAN as V3_PLAN,audit_training_cutoffs,truth_for
from .rerank_data import EARLIER


PLAN={'hypothesis':'Within-user ranking loss may order binary frequency-selected Top5 better than weighted per-row classification.',
    'candidate_a':'standalone LambdaRank, same candidate pool and 59 V4 features',
    'candidate_b':'one equal RRF with frozen V4, only if A has useful complementary corrections',
    'fusion_gate':'A within 0.02 MAP of V4 on both windows; >=50 improved users and >=20 newly retrieved relevant IDs on each window',
    'parameters':PARAMETERS,'rounds':ROUNDS,'configurations':1,
    'training_folds':V3_PLAN['training_folds'],'label':'binary membership of frequency-selected qualifying Top5; not graded count or every >=80% show',
    'groups':'one user x one historical prediction task; never mix the same user across folds into one query',
    'ranking_weights':'no classifier positive-times10 weights; LambdaRank query normalization and binary gains [0,1]',
    'validation_labels_in_fit':False,'early_stopping':False,'kaggle_score_use':False,
    'selection':'Both gain >=0.001, paired95 lower>0 both, light-history gain>=-0.003 both; else retain V4',
    'objective_caveat':'LambdaRank uses an NDCG-based pairwise surrogate at truncation8, not an exact differentiable MAP5 objective. Selection only uses external exact MAP5.',
    'sources':['https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective',
               'https://lightgbm.readthedocs.io/en/stable/Parameters.html#lambdarank_truncation_level',
               'https://lightgbm.readthedocs.io/en/stable/Python-API.html']}


def freeze(out):
    p=out/'frozen_prior.json'
    if not p.exists():
        files={str(f):{'sha256':digest(f),'bytes':f.stat().st_size}
            for f in sorted(Path('artifacts').rglob('*')) if f.is_file() and 'stage6' not in f.parts}
        for folder in ['repository_artifacts','cache/focused_v4']:
            for f in Path(folder).glob('*'):
                if f.is_file() and (folder=='repository_artifacts' or f.suffix=='.pkl'):
                    files[str(f)]={'sha256':digest(f),'bytes':f.stat().st_size}
        save(p,{'files':files,'created_at_utc':datetime.now(timezone.utc).isoformat()})
    frozen=json.loads(p.read_text())['files'];check_evidence(frozen)
    assert digest('artifacts/final/submission_hybrid_v4.csv')=='08dbcca1aa312f03397447205684143b20c9a632d5fc8868c4c2916210d44ce2'
    return frozen


def matrix(name):
    base,meta,z=load_dataset(name)
    extra=Path('cache/focused_v4')/name
    marker=extra/'extras.json'
    if not marker.exists():marker=extra/'extras_loyalty.json'
    info=json.loads(marker.read_text())
    assert info['cutoff']==meta['signature']['cutoff'] and info['last_used_view_second']<=info['cutoff']
    X=np.column_stack([np.load(base/'X.npy'),np.load(extra/'loyalty.npy')])
    names=meta['feature_names']+info['columns']['loyalty']
    assert X.shape==(len(z['user_index']),59) and len(names)==59
    return X,meta,z,names


def v4_rows(window):
    with Path(f'artifacts/stage5/private/{window}_loyalty.csv').open(newline='') as f:
        return {r['user_id']:r for r in csv.DictReader(f)}


def audit(c,out):
    audit_training_cutoffs(PLAN['training_folds'],EARLIER)
    fold_records=[]
    for fold in PLAN['training_folds']:
        _,meta,z=load_dataset(fold['name'])
        info=json.loads((Path('cache/focused_v4')/fold['name']/'extras.json').read_text())
        assert meta['signature']['cutoff']==info['cutoff']==fold['cutoff']
        assert info['last_used_view_second']<=fold['cutoff'] and not info['future_user_views_used']
        truth,_=truth_for(c,fold['name'],fold['cutoff'],fold['target_end'],meta['users'])
        y=labels_for_pool(meta,z,truth)
        record=validate_fold_groups(z['user_index'],z['item_index'],np.array(meta['pool_lengths']),y)
        record.update(fold)
        fold_records.append(record)
    with Path('cache/focused_v4/loyalty_early.pkl').open('rb') as f:
        classifier=pickle.load(f)
    assert classifier.get_params()['loss']=='log_loss'
    result={'classifier':'HistGradientBoostingClassifier, weighted binary log_loss; no pairwise/listwise objective',
        'labels_are_top5_membership_not_any_completed_programme':True,
        'coverage_threshold':'>=80% uninterrupted original airing duration, then frequency Top5',
        'classifier_weighting':'per-query size correction; positive rows x10; within-fold normalization',
        'classifier_actual_loss':classifier.get_params()['loss'],
        'classifier_model_sha256':digest('cache/focused_v4/loyalty_early.pkl'),
        'ranking_group_definition':'user x fold; contiguous candidates in cached order',
        'folds':fold_records,'last_training_label_end':EARLIER,'main_eval_cutoff':SPLIT,'earlier_eval_cutoff':EARLIER,
        'no_eval_labels_fit_or_preprocessing':True,'features':59,'pools_unchanged':True}
    save(out/'target_group_audit.json',result)
    diagnoses=[]
    for window in ['main','earlier']:
        base,meta,z=load_dataset(window);source=v4_rows(window)
        p=np.load(Path('cache/focused_v4')/window/'loyalty_early_probabilities.npy')
        starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
        truth_pairs=sum(len(r['truth'].split()) for r in source.values())
        recall={k:0 for k in [20,50,200]};inside=outside=0
        for n,u in enumerate(meta['users']):
            lo,hi=starts[n:n+2];ids=z['item_index'][lo:hi]
            learned=np.argsort(-p[lo:hi],kind='stable');r=np.empty(len(learned),int);r[learned]=np.arange(1,len(learned)+1)
            score=1/(10+np.arange(1,len(r)+1))+1/(10+r)
            order=np.argsort(-score,kind='stable') if not z['cold'][n] else np.arange(len(r))
            ranked=[meta['items'][i] for i in ids[order]]
            assert ranked[:5]==list(map(int,source[u]['predictions'].split()))
            truth=set(map(int,source[u]['truth'].split()))
            for k in recall:recall[k]+=len(truth&set(ranked[:k]))
            missing=truth-set(ranked[:5]);pool=set(ranked)
            inside+=len(missing&pool);outside+=len(missing-pool)
        diagnoses.append({'window':window,'candidate_recall_micro':{str(k):v/truth_pairs for k,v in recall.items()},
            'missing_relevant_inside_pool_below5':inside,'missing_relevant_outside_pool':outside,
            'all_v4_top5_exactly_reproduced':True,'mean_pool_size':float(np.mean(meta['pool_lengths']))})
    save(out/'v4_ranking_audit.json',diagnoses)
    print(json.dumps({'groups':fold_records,'ranking':diagnoses}),flush=True)


def evaluate(out,window,name,meta,z,score,baseline_classifier=None,fusion=False):
    source=v4_rows(window);users=meta['users'];truth={u:set(map(int,source[u]['truth'].split())) for u in users}
    with measured() as resources:pred=prediction_lists(meta,z,score,baseline_classifier,fusion)
    warm={u for n,u in enumerate(users) if not z['cold'][n]}
    delta=np.array([ap_at_5(truth[u],pred[u])-float(source[u]['ap5']) for u in users])
    rng=np.random.default_rng(42);boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)])
    segments=pd.read_csv('artifacts/stage4/private/v2_v3_user_deltas.csv',dtype={'user_id':str})
    segments=segments[segments.window==window].set_index('user_id')
    light=[i for i,u in enumerate(users) if segments.loc[u,'history_group']=='light_under10']
    new_relevant=sum(len((set(pred[u])-set(map(int,source[u]['predictions'].split())))&truth[u]) for u in users)
    row={'window':window,'candidate':name,**metrics(truth,pred,users,warm),
        'gain':float(delta.mean()),'paired_95_ci':np.quantile(boot,[.025,.975]).tolist(),
        'improved':int((delta>0).sum()),'regressed':int((delta<0).sum()),
        'new_relevant_programme_ids':new_relevant,'light_history_gain':float(delta[light].mean()),
        'ranking_resources':resources,'surrogate_metric_not_used_for_selection':True}
    save_predictions(out/'private'/f'{window}_{name}.csv',pred,truth,users,warm)
    print(window,name,row['map5'],row['gain'],row['paired_95_ci'],flush=True)
    return row,pred


def experiment(c,out):
    xs=[];ys=[];groups=[];records=[]
    for fold in PLAN['training_folds']:
        X,meta,z,names=matrix(fold['name'])
        if xs:assert names==feature_names
        feature_names=names
        truth,_=truth_for(c,fold['name'],fold['cutoff'],fold['target_end'],meta['users'])
        y=labels_for_pool(meta,z,truth)
        record=validate_fold_groups(z['user_index'],z['item_index'],np.array(meta['pool_lengths']),y)
        record.update(fold)
        xs.append(X);ys.append(y);groups.extend(meta['pool_lengths']);records.append(record)
    X=np.concatenate(xs);y=np.concatenate(ys);del xs,ys
    folder=Path('cache/group_ltr');folder.mkdir(exist_ok=True)
    with measured() as fit_resources:model=fit_ranker(X,y,groups)
    model.save_model(str(folder/'lambda_early.txt'))
    save(out/'fit_audit.json',{'queries':len(groups),'rows':len(y),'positives':int(y.sum()),'folds':records,
        'last_training_label_end':EARLIER,'fit_resources':fit_resources,'lightgbm':lightgbm_module().__version__,
        'actual_boosting_iterations':model.current_iteration(),'no_holdout_training_or_early_stopping':True})
    del X,y
    import gc;gc.collect()
    rows=[];a_rows=[];evaluation={}
    for window in ['main','earlier']:
        X,meta,z,names=matrix(window)
        assert names==feature_names
        with measured() as inference:score=model.predict(X)
        np.save(folder/(window+'_lambda_scores.npy'),score)
        row,pred=evaluate(out,window,'lambda',meta,z,score)
        row['inference_resources']=inference
        rows.append(row);a_rows.append(row);evaluation[window]=(meta,z,score)
        del X;gc.collect()
    complementary=all(r['gain']>=-.02 and r['improved']>=50 and r['new_relevant_programme_ids']>=20 for r in a_rows)
    save(out/'fusion_gate.json',{'run_b':complementary,'rule':PLAN['fusion_gate'],'a_results':a_rows})
    if complementary:
        for window,(meta,z,score) in evaluation.items():
            old=np.load(Path('cache/focused_v4')/window/'loyalty_early_probabilities.npy')
            row,_=evaluate(out,window,'lambda_v4_rrf',meta,z,score,old,True);rows.append(row)
    save(out/'results.json',rows)
    by={(r['candidate'],r['window']):r for r in rows}
    candidates=['lambda']+(['lambda_v4_rrf'] if complementary else [])
    eligible=[k for k in candidates if all(by[k,w]['gain']>=.001 and by[k,w]['paired_95_ci'][0]>0
        and by[k,w]['light_history_gain']>=-.003 for w in ['main','earlier'])]
    selected=max(eligible,key=lambda k:sum(by[k,w]['gain'] for w in ['main','earlier'])) if eligible else 'retain_v4'
    save(out/'selection.json',{'selected':selected,'eligible':eligible,'rule':PLAN['selection'],
        'results':rows,'configurations_trained':1,'features_and_pools_changed':False,'kaggle_requests':0})
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['ranking_resources','inference_resources']} for r in rows]).to_csv(out/'comparison.csv',index=False)
    save(out/'feature_gain.json',{'interpretation':'descriptive LambdaRank split gain, not causal feature ablation',
        'features':sorted([{'name':n,'gain':float(g)} for n,g in zip(names,model.feature_importance('gain'))],key=lambda r:-r['gain'])})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['audit','experiment']);args=parser.parse_args()
    out=Path('artifacts/stage6');out.mkdir(exist_ok=True);(out/'private').mkdir(exist_ok=True)
    frozen=freeze(out)
    plan=out/'plan.json'
    if plan.exists():assert json.loads(plan.read_text())==PLAN
    else:save(plan,PLAN)
    with measured() as resources:
        c=connect(Path('cache'))
        if args.stage=='audit':audit(c,out)
        else:
            assert (out/'target_group_audit.json').exists()
            audit_training_cutoffs(PLAN['training_folds'],EARLIER)
            experiment(c,out)
        c.close();check_evidence(frozen)
    save(out/f'execution_{args.stage}.json',{'resources':resources,'lightgbm':lightgbm_module().__version__,
        'source_sha256':{str(p):digest(p) for p in Path('tvrec').glob('*.py')},'kaggle_requests':0})


if __name__=='__main__':main()
