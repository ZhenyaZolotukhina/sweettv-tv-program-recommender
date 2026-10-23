"""One same-pool collaborative-feature ablation, with a bounded expansion gate.

python -m tvrec.new_cf_rerank
Rolling BPR scores are fitted independently before their own prediction cutoffs.
"""
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .advanced import metrics,save_predictions
from .core import ap_at_5
from .final_refit import check_evidence
from .focused import load_dataset
from .group_rank import PARAMETERS,fit_ranker,labels_for_pool,validate_fold_groups,prediction_lists
from .improve import connect
from .ltr_experiment import matrix
from .models import measured
from .new_cf import CONFIGS,freeze,fit_context,ordered,baseline
from .pipeline import save
from .rerank import PLAN as ROLLING,audit_training_cutoffs,truth_for
from .rerank_data import EARLIER


FEATURES=['bpr_raw_score','bpr_reciprocal_rank','bpr_item_bias','bpr_latent_cosine',
          'bpr_user_factor_norm','bpr_item_factor_norm','bpr_user_known','bpr_item_known']


def cutoff_fallback(name,folder,meta):
    existing=Path(f'cache/improvements/{name}_v1_scores.npz')
    if existing.exists():
        with np.load(existing) as q:pop=q['popularity'][0]
    else:
        # Early tasks have no later-model cache. Recover frequency-only global
        # popularity from THIS cutoff's original positive matrix, not the BPR
        # sampling weights or a later evaluation/final model.
        info=json.loads((folder/'manifest.json').read_text())
        with (folder/'model.pkl').open('rb') as f:model=pickle.load(f)
        totals=np.asarray(model.matrix.sum(axis=0)).ravel()
        mapping={int(i):n for n,i in enumerate(info['model_items'])}
        pop=np.array([totals[mapping[int(i)]] if int(i) in mapping else 0 for i in meta['items']])
    return np.lexsort((np.array(meta['items']),-pop))


def features_for_indices(raw,bias,unorm,inorm,user_known,item_known,user_index,item_index,fallback):
    rank=[]
    for u in np.unique(user_index):
        at=np.flatnonzero(user_index==u)
        order=ordered(raw[u],fallback)
        inverse=np.empty(raw.shape[1],int);inverse[order]=np.arange(1,raw.shape[1]+1)
        rank.append((at,inverse[item_index[at]]))
    r=np.zeros(len(user_index))
    for at,value in rank:r[at]=value
    score=raw[user_index,item_index]
    known=user_known[user_index]&item_known[item_index]
    score=np.where(known,score,0.)
    dot=np.where(known,score-bias[item_index],0.)
    norm=unorm[user_index]*inorm[item_index]
    cosine=np.divide(dot,norm,out=np.zeros_like(dot,dtype=float),where=norm>0)
    X=np.column_stack([score,1/(10+r),bias[item_index],np.clip(cosine,-1,1),
        unorm[user_index],inorm[item_index],user_known[user_index],item_known[item_index]]).astype(np.float32)
    assert np.isfinite(X).all()
    return X


def extra(c,name,config,out):
    folder,info=fit_context(c,name,config,out)
    base,meta,z=load_dataset(name)
    target=folder/'pair_features.npy'
    if target.exists():return target
    with np.load(folder/'scores.npz') as q:
        raw=q['score'];bias=q['item_bias'];unorm=q['user_norm'];inorm=q['item_norm'];uk=q['user_seen'];ik=q['item_seen']
    fallback=cutoff_fallback(name,folder,meta)
    X=features_for_indices(raw,bias,unorm,inorm,uk,ik,z['user_index'],z['item_index'],fallback)
    np.save(target,X)
    return target


def expansion_gate(out,config):
    gate={'minimum_new_relevant_top50_pairs_each_window':10,'top_n':50,'results':[]}
    for name in ['main','earlier']:
        _,meta,z=load_dataset(name);source=baseline(name)
        with np.load(Path('cache/new_cf')/config['positive_weight']/name/'scores.npz') as q:raw=q['score']
        with np.load(f'cache/improvements/{name}_v1_scores.npz') as q:pop=q['popularity'][0]
        fallback=np.lexsort((np.array(meta['items']),-pop));starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
        novel=0;total_extra=0
        for n,u in enumerate(meta['users']):
            order=ordered(raw[n],fallback)[:50];order=order[np.isfinite(raw[n,order])]
            existing=set(z['item_index'][starts[n]:starts[n+1]])
            new=set(order)-existing;total_extra+=len(new)
            novel+=len({meta['items'][i] for i in new}&set(map(int,source[u]['truth'].split())))
        gate['results'].append({'window':name,'additional_pairs':total_extra,'new_relevant_pairs':novel})
    gate['expansion_justified']=all(r['new_relevant_pairs']>=10 for r in gate['results'])
    save(out/'expansion_gate.json',gate)
    return gate


def main():
    out=Path('artifacts/stage8');frozen=freeze(out)
    information=json.loads((out/'information_gate.json').read_text())
    if not information['run_reranker']:
        save(out/'decision.json',{'selected':'retain_v5','reason':'CF information gate failed','kaggle_requests':0});return
    config=CONFIGS[information['selected_weighting']]
    p=out/'integration_plan.json'
    plan={'configuration':config,'features':FEATURES,'user_bias':'not identifiable; omitted',
        'base_features':59,'same_pools':True,'training_folds':ROLLING['training_folds'],
        'ranker_parameters':PARAMETERS|{'lambdarank_truncation_level':5},
        'maximum_variants':2,'expansion_rule':'One optional V5-pool union BPR Top50 variant only if >=10 new relevant pairs on BOTH holdouts.',
        'model_selection_uses_kaggle_scores':False}
    if p.exists():assert json.loads(p.read_text())==plan
    else:save(p,plan)
    gate=expansion_gate(out,config)
    c=connect(Path('cache'));xs=[];ys=[];groups=[];records=[]
    audit_training_cutoffs(plan['training_folds'],EARLIER)
    with measured() as resources:
        for fold in plan['training_folds']:
            X,meta,z,names=matrix(fold['name'])
            f=extra(c,fold['name'],config,out)
            X=np.column_stack([X,np.load(f)])
            truth,_=truth_for(c,fold['name'],fold['cutoff'],fold['target_end'],meta['users'])
            y=labels_for_pool(meta,z,truth)
            audit=validate_fold_groups(z['user_index'],z['item_index'],np.array(meta['pool_lengths']),y)
            records.append(fold|audit)
            xs.append(X);ys.append(y);groups.extend(meta['pool_lengths'])
        X=np.concatenate(xs);y=np.concatenate(ys);del xs,ys
        with measured() as fit:model=fit_ranker(X,y,groups,parameters=plan['ranker_parameters'])
        model.save_model('cache/new_cf/same_pool_ranker_early.txt')
        del X,y
        import gc;gc.collect()
        results=[]
        seg=pd.read_csv('artifacts/stage4/private/v2_v3_user_deltas.csv',dtype={'user_id':str})
        for name in ['main','earlier']:
            X,meta,z,names=matrix(name)
            with measured() as inference:
                X=np.column_stack([X,np.load(extra(c,name,config,out))]);score=model.predict(X)
                pred=prediction_lists(meta,z,score)
            np.save(Path('cache/new_cf')/(name+'_same_pool_ranker_scores.npy'),score)
            source=baseline(name);truth={u:set(map(int,source[u]['truth'].split())) for u in meta['users']}
            warm={u for n,u in enumerate(meta['users']) if not z['cold'][n]}
            delta=np.array([ap_at_5(truth[u],pred[u])-float(source[u]['ap5']) for u in meta['users']])
            rng=np.random.default_rng(42);boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)])
            s=seg[seg.window==name].set_index('user_id');subgroups={}
            for group,field,key in [('light','history_group','light_under10'),('heavy','history_group','heavy10plus'),('repeat_heavy','repeat_group','repeat_heavy')]:
                index=[n for n,u in enumerate(meta['users']) if s.loc[u,field]==key]
                subgroups[group]=float(delta[index].mean())
            row={'variant':'same_pool_features','window':name,**metrics(truth,pred,meta['users'],warm),
                'gain':float(delta.mean()),'paired_95_ci':np.quantile(boot,[.025,.975]).tolist(),
                'improved':int((delta>0).sum()),'regressed':int((delta<0).sum()),'subgroups':subgroups,
                'fit_resources':fit,'inference_resources':inference,'training_folds':records,'input_features':59+len(FEATURES)}
            results.append(row);save(out/'integration_results.json',results)
            save_predictions(out/'private'/f'{name}_same_pool_features.csv',pred,truth,meta['users'],warm)
            print(name,'same_pool_features',row['map5'],row['gain'],row['paired_95_ci'],subgroups,flush=True)
        c.close();check_evidence(frozen)
    save(out/'execution_integration.json',{'resources':resources,'config':config,'new_cf_feature_names':FEATURES,'kaggle_requests':0})
    by={r['window']:r for r in results}
    eligible=all(by[w]['gain']>=.001 and by[w]['paired_95_ci'][0]>0 and by[w]['subgroups']['light']>=-.003
        and by[w]['subgroups']['repeat_heavy']>=-.003 for w in ['main','earlier'])
    save(out/'same_pool_decision.json',{'selected':'same_pool_features' if eligible else 'retain_v5',
        'expansion_gate':gate,'kaggle_requests':0})


if __name__=='__main__':main()
