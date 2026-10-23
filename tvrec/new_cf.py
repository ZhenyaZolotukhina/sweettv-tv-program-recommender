"""Cutoff-correct BPR feasibility/retrieval study; reranking only if informative.

python -m tvrec.new_cf standalone
"""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .advanced import metrics,save_predictions
from .bpr_model import ImplicitBPR
from .core import ap_at_5
from .final_refit import digest,check_evidence
from .focused import load_dataset
from .improve import connect
from .labels import event_frame,training_features
from .models import matrix_from_features,measured
from .pipeline import LOW,SPLIT,TEST,save
from .rerank_data import EARLIER


CONFIGS={name:{'factors':32,'epochs':25,'batch_size':2048,'learning_rate':.05,
    'regularization':.002,'seed':42,'positive_weight':name} for name in ['binary','log_frequency']}
PLAN={'new_cf':'BPR fallback after one failed LightFM attempt; never described as WARP',
    'configurations':CONFIGS,'input':'unchanged >=50% uninterrupted duration, distinct qualifying-airing frequency',
    'user_sampling':'uniform users with at least one positive and at least one eligible unobserved item',
    'item_universe':'only historical items with qualified interactions, never future-only schedule IDs',
    'cold_policy':'same frozen V5 global-popularity fallback; no CF personalization without qualified history',
    'biases':'learned item bias; user bias not identifiable under pairwise loss',
    'integration_gate':'Each window must yield >=20 relevant user-item pairs outside frozen existing pool AND standalone MAP no worse than ALS by >0.05',
    'cf_selection':'largest minimum novel-pair gain across both windows, then mean standalone MAP; no Kaggle scores',
    'maximum_downstream_variants':2,'downstream':'same-pool CF features first; optional union+newCF Top50 only if useful novelty',
    'selection_rule':'Both MAP gains>=0.001 and paired95 lower>0; light/repeat-heavy gain>=-0.003 both, cold identical; else retain V5'}


def freeze(out):
    f=out/'frozen_prior.json'
    if not f.exists():
        files={str(p):{'sha256':digest(p),'bytes':p.stat().st_size}
            for p in Path('artifacts').rglob('*') if p.is_file() and 'stage8' not in p.parts}
        for directory,pattern in [('repository_artifacts','*'),('cache/group_ltr','*.txt'),('cache/focused_v4','*.pkl'),('cache/rerank','*.pkl')]:
            for p in Path(directory).glob(pattern):
                if p.is_file():files[str(p)]={'sha256':digest(p),'bytes':p.stat().st_size}
        save(f,{'files':files,'created_at_utc':datetime.now(timezone.utc).isoformat()})
    frozen=json.loads(f.read_text())['files'];check_evidence(frozen)
    assert digest('artifacts/final/submission_hybrid_v5.csv')=='249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f'
    return frozen


def baseline(window):
    with Path(f'artifacts/stage6/private/{window}_lambda_top5.csv').open(newline='') as f:
        return {r['user_id']:r for r in csv.DictReader(f)}


def fit_context(c,name,config,out):
    base,meta,z=load_dataset(name);cutoff=meta['signature']['cutoff'];upper=meta['signature']['upper']
    folder=Path('cache/new_cf')/config['positive_weight']/name;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'manifest.json'
    signature={'configuration':config,'cutoff':cutoff,'upper':upper,'code':digest('tvrec/bpr_model.py'),
        'input_manifest':digest(base/'features_manifest.json')}
    if marker.exists() and json.loads(marker.read_text())['signature']==signature:
        return folder,json.loads(marker.read_text())
    with measured() as resources:
        frame=event_frame(c,LOW,cutoff)
        assert (frame.b<=cutoff).all()
        f=training_features(frame,minimum=.5,strength='frequency',half_life=0,cutoff=cutoff)
        users=sorted(set(f.user_id));items=sorted(set(int(i) for i in f.item))
        ui={u:n for n,u in enumerate(users)};ii={i:n for n,i in enumerate(items)}
        matrix=matrix_from_features(f,users,items)
        with measured() as fit:model=ImplicitBPR(**config).fit(matrix)
        with (folder/'model.pkl').open('wb') as g:pickle.dump(model,g,protocol=pickle.HIGHEST_PROTOCOL)
        query_user=np.array([ui.get(u,-1) for u in meta['users']])
        candidate_item=np.array([ii.get(int(i),-1) for i in meta['items']])
        score=np.full((len(query_user),len(candidate_item)),-np.inf,dtype=np.float32)
        user_ok=query_user>=0;item_ok=candidate_item>=0
        with measured() as inference:
            with threadpool_limits(limits=1):
                score[np.ix_(user_ok,item_ok)]=model.score(query_user[user_ok],candidate_item[item_ok]).astype(np.float32)
        bias=np.zeros(len(candidate_item));bias[item_ok]=model.item_bias[candidate_item[item_ok],0]
        unorm=np.zeros(len(query_user));unorm[user_ok]=np.linalg.norm(model.user_factors[query_user[user_ok]],axis=1)
        inorm=np.zeros(len(candidate_item));inorm[item_ok]=np.linalg.norm(model.item_factors[candidate_item[item_ok]],axis=1)
        np.savez_compressed(folder/'scores.npz',score=score,item_bias=bias,user_norm=unorm,item_norm=inorm,
                            user_seen=user_ok,item_seen=item_ok)
    record={'signature':signature,'fit_resources':fit,'inference_resources':inference,'resources':resources,
        'training_users':len(users),'training_items':len(items),'positive_pairs':matrix.nnz,
        'historical_last_second':int(frame.b.max()),'no_future_item_negative_sampling':True,
        'score_mapping_users':meta['users'],'score_mapping_items':meta['items'],
        'model_users':users,'model_items':items,'user_bias':'not identifiable, omitted'}
    save(marker,record)
    return folder,record


def ordered(score,fallback):
    order=fallback[np.argsort(-score[fallback],kind='stable')]
    supported=order[np.isfinite(score[order])];seen=set(supported.tolist())
    return np.r_[supported,np.array([i for i in fallback if i not in seen],dtype=int)]


def standalone(c,out):
    rows=[]
    for weighting,config in CONFIGS.items():
        for window in ['main','earlier']:
            folder,info=fit_context(c,window,config,out)
            _,meta,z=load_dataset(window)
            source=baseline(window);users=meta['users'];truth={u:set(map(int,source[u]['truth'].split())) for u in users}
            with np.load(folder/'scores.npz') as q:score=q['score']
            with np.load(f'cache/improvements/{window}_v1_scores.npz') as q:pop=q['popularity'][0]
            fallback=np.lexsort((np.array(meta['items']),-pop))
            starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
            pred={};hits=novel_als=novel_pool=0;overlap=[]
            for n,u in enumerate(users):
                order=ordered(score[n],fallback) if not z['cold'][n] else fallback
                pred[u]=[int(meta['items'][i]) for i in order[:5]]
                supported=order[:200]
                ids={meta['items'][i] for i in supported}
                als={meta['items'][i] for i in z['cf'][n,:200]}
                existing={meta['items'][i] for i in z['item_index'][starts[n]:starts[n+1]]}
                hits+=len(ids&truth[u]);novel_als+=len((ids-als)&truth[u]);novel_pool+=len((ids-existing)&truth[u])
                overlap.append(len(ids&als)/len(ids|als))
            warm={u for n,u in enumerate(users) if not z['cold'][n]}
            row={'model':'implicit_BPR','weighting':weighting,'window':window,**metrics(truth,pred,users,warm),
                'recall200_micro':hits/sum(len(t) for t in truth.values()),
                'new_relevant_vs_als200':novel_als,'new_relevant_outside_v5_union':novel_pool,
                'mean_top200_jaccard_als':float(np.mean(overlap)),
                'fit_resources':info['fit_resources'],'inference_resources':info['inference_resources']}
            rows.append(row);save(out/'standalone_results.json',rows)
            save_predictions(out/'private'/f'{window}_bpr_{weighting}.csv',pred,truth,users,warm)
            print(window,weighting,'MAP',row['map5'],'Recall200',row['recall200_micro'],'new_vs_ALS',novel_als,'new_vs_pool',novel_pool,flush=True)
    als=json.loads(Path('artifacts/stage2/leaderboard.json').read_text())
    earlier=json.loads(Path('artifacts/stage2/earlier_robustness.json').read_text())
    als_map={'main':next(r['map5'] for r in als if r['family']=='cf'),'earlier':earlier['individual']['cf']['map5']}
    by={(r['weighting'],r['window']):r for r in rows}
    eligible=[w for w in CONFIGS if all(by[w,name]['new_relevant_outside_v5_union']>=20
        and by[w,name]['map5']>=als_map[name]-.05 for name in ['main','earlier'])]
    chosen=max(eligible,key=lambda w:(min(by[w,n]['new_relevant_outside_v5_union'] for n in ['main','earlier']),
        np.mean([by[w,n]['map5'] for n in ['main','earlier']]))) if eligible else None
    save(out/'information_gate.json',{'selected_weighting':chosen,'eligible':eligible,'rule':PLAN['integration_gate'],
        'als_map':als_map,'run_reranker':chosen is not None,'kaggle_requests':0})
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['fit_resources','inference_resources']} for r in rows]).to_csv(out/'standalone_results.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['standalone']);args=parser.parse_args()
    out=Path('artifacts/stage8');out.mkdir(exist_ok=True);(out/'private').mkdir(exist_ok=True)
    frozen=freeze(out);p=out/'plan.json'
    if p.exists():assert json.loads(p.read_text())==PLAN
    else:save(p,PLAN)
    with measured() as resources:
        c=connect(Path('cache'));standalone(c,out);c.close();check_evidence(frozen)
    save(out/'execution_standalone.json',{'resources':resources,'source_sha256':{str(p):digest(p) for p in Path('tvrec').glob('*.py')},
        'executor':'AI assistant; no student verification claimed','kaggle_requests':0})


if __name__=='__main__':main()
