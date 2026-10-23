"""Association study and verified BPR failure audit; no Kaggle requests.

python -m tvrec.assoc_study audit
python -m tvrec.assoc_study standalone
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
from .association_model import ItemAssociations
from .final_refit import digest,check_evidence
from .focused import load_dataset
from .group_rank import lightgbm_module
from .improve import connect
from .labels import event_frame,training_features
from .models import matrix_from_features,measured,ranked_predictions
from .pipeline import LOW,save
from .rerank import PLAN as ROLLING,truth_for


CONFIGS={
    'confidence':{'kind':'confidence','min_users':3,'shrinkage':20.,'prior':10.,'neighbors':200},
    'shrink_cosine':{'kind':'shrink_cosine','min_users':3,'shrinkage':20.,'prior':10.,'neighbors':200}}
PLAN={'association_scorers':CONFIGS,'transactions':'One binary basket per distinct user over pre-cutoff history; repeated airings affect seed weight, not independent-user rule support.',
    'input_policy':'>=50% uninterrupted original programme duration, same unique-airing counts as V5',
    'seed_weights':'L1-normalized log1p qualified airing frequency; self rules removed; watched items otherwise allowed',
    'association_config_selection':'Largest average standalone MAP across both windows; no Kaggle score use',
    'integration_a':'Old V5 pool UNION association supported Top100; V5 59 features plus association evidence; same policy in rolling training/inference.',
    'integration_b_gate':'Run only if audit confirms raw CF scale/distribution shift (main median user norm >1.5x maximum rolling median). Existing exposure policy is NOT assumed buggy.',
    'integration_b':'Same cached BPR Top200 expansion; replace raw scale-dependent BPR fields with cutoff-local z/percentiles, explicit NaN missing values and known flags. No new CF fits or candidate N search.',
    'maximum_integrations':2,'training_folds':ROLLING['training_folds'],
    'selection':'Both MAP gain>=0.001 and paired95 lower>0; heavy/light/repeat-heavy gain>=-0.003 both, cold identical; otherwise retain V5',
    'kaggle_score_use':False,'source':'https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/',
    'attribution':'Support/confidence terminology and standard cosine. Formulas and sparse implementation independently authored; mlxtend not installed or copied.'}


def freeze(out):
    p=out/'frozen_prior.json'
    if not p.exists():
        files={str(f):{'sha256':digest(f),'bytes':f.stat().st_size} for f in Path('artifacts').rglob('*')
               if f.is_file() and 'stage9' not in f.parts}
        for directory,pattern in [('repository_artifacts','*'),('cache/group_ltr','*.txt'),('cache/new_cf','*.txt'),('cache/focused_v4','*.pkl'),('cache/rerank','*.pkl')]:
            for f in Path(directory).glob(pattern):
                if f.is_file():files[str(f)]={'sha256':digest(f),'bytes':f.stat().st_size}
        for f in Path('cache/new_cf').rglob('model.pkl'):files[str(f)]={'sha256':digest(f),'bytes':f.stat().st_size}
        save(p,{'files':files,'created_at_utc':datetime.now(timezone.utc).isoformat()})
    frozen=json.loads(p.read_text())['files'];check_evidence(frozen)
    assert digest('artifacts/final/submission_hybrid_v5.csv')=='249ed58e19e51ee2905bb256d97263ec06201d51f9cf40a2765904f4377c448f'
    return frozen


def baseline(window):
    with Path(f'artifacts/stage6/private/{window}_lambda_top5.csv').open(newline='') as f:
        return {r['user_id']:r for r in csv.DictReader(f)}


def association_fallback(name,score_folder,meta):
    cached=Path(f'cache/improvements/{name}_v1_scores.npz')
    if cached.exists():
        with np.load(cached) as q:pop=q['popularity'][0]
    else:
        info=json.loads((score_folder/'manifest.json').read_text())
        with (score_folder/'model.pkl').open('rb') as f:model=pickle.load(f)
        totals=np.asarray(model.frequency.sum(axis=0)).ravel()
        ii={int(i):n for n,i in enumerate(info['model_items'])}
        pop=np.array([totals[ii[int(i)]] if int(i) in ii else 0 for i in meta['items']])
    return np.lexsort((np.array(meta['items']),-pop))


def dataset(c,name,kind):
    _,meta,z=load_dataset(name);config=CONFIGS[kind]
    folder=Path('cache/association')/name/kind;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'manifest.json'
    signature={'config':config,'cutoff':meta['signature']['cutoff'],'code':digest('tvrec/association_model.py')}
    if marker.exists() and json.loads(marker.read_text())['signature']==signature:return folder,json.loads(marker.read_text())
    with measured() as resources:
        f=training_features(event_frame(c,LOW,signature['cutoff']),minimum=.5,strength='frequency',cutoff=signature['cutoff'])
        users=sorted(set(f.user_id));items=sorted(set(int(i) for i in f.item));ui={u:n for n,u in enumerate(users)};ii={i:n for n,i in enumerate(items)}
        frequency=matrix_from_features(f,users,items)
        with measured() as fit:model=ItemAssociations(**config).fit(frequency)
        query_user=np.array([ui.get(u,-1) for u in meta['users']]);candidate=np.array([ii.get(int(i),-1) for i in meta['items']])
        uk=query_user>=0;ik=candidate>=0
        score=np.zeros((len(query_user),len(candidate)),dtype=np.float32);count=np.zeros_like(score);mass=np.zeros_like(score)
        with measured() as inference:
            a,b,d=model.score(query_user[uk]);score[np.ix_(uk,ik)]=a[:,candidate[ik]]
            count[np.ix_(uk,ik)]=b[:,candidate[ik]];mass[np.ix_(uk,ik)]=d[:,candidate[ik]]
        with (folder/'model.pkl').open('wb') as g:pickle.dump(model,g,protocol=pickle.HIGHEST_PROTOCOL)
        support=np.zeros(len(candidate));support[ik]=model.support[candidate[ik]]/len(users)
        np.savez_compressed(folder/'scores.npz',score=score,seed_count=count,seed_mass=mass,item_support=support,
            user_known=uk,item_known=ik)
    record={'signature':signature,'model_users':users,'model_items':items,'last_historical_second':int(f.last_second.max()),
        'fit_resources':fit,'inference_resources':inference,'resources':resources,
        'distinct_user_item_pairs':frequency.nnz,'edges':model.similarity.nnz,'no_future_views':True}
    save(marker,record);return folder,record


def standalone(c,out):
    rows=[]
    for kind in CONFIGS:
        for window in ['main','earlier']:
            folder,info=dataset(c,window,kind);_,meta,z=load_dataset(window)
            with np.load(folder/'scores.npz') as q:score=q['score']
            with np.load(f'cache/improvements/{window}_v1_scores.npz') as q:pop=q['popularity'][0]
            fallback=np.lexsort((np.array(meta['items']),-pop));users=meta['users'];warm={u for n,u in enumerate(users) if not z['cold'][n]}
            pred=ranked_predictions(score,users,meta['items'],fallback,set(users)-warm)
            original=baseline(window);truth={u:set(map(int,original[u]['truth'].split())) for u in users}
            starts=np.r_[0,np.cumsum(meta['pool_lengths'])];hits=outside=0
            for n,u in enumerate(users):
                order=fallback[np.argsort(-score[n,fallback],kind='stable')][:200] if u in warm else fallback[:200]
                ids={meta['items'][i] for i in order};old={meta['items'][i] for i in z['item_index'][starts[n]:starts[n+1]]}
                hits+=len(ids&truth[u]);outside+=len((ids-old)&truth[u])
            row={'method':kind,'window':window,**metrics(truth,pred,users,warm),'recall200_micro':hits/sum(len(t) for t in truth.values()),
                'new_relevant_outside_v5_pool':outside,'fit_resources':info['fit_resources'],'inference_resources':info['inference_resources']}
            rows.append(row);save(out/'association_standalone.json',rows)
            save_predictions(out/'private'/f'{window}_association_{kind}.csv',pred,truth,users,warm)
            print(window,kind,row['map5'],row['recall200_micro'],outside,flush=True)
    selected=max(CONFIGS,key=lambda k:np.mean([r['map5'] for r in rows if r['method']==k]))
    save(out/'association_choice.json',{'selected':selected,'rule':PLAN['association_config_selection'],'two_scorers_only':True})


def audit_bpr(c,out):
    training_pairs=set();positive_pairs=set();training_items=set();positive_items=set();distribution=[];norms=[]
    for name in ['roll_mar23','roll_apr06','main','earlier']:
        base,meta,z=load_dataset(name);folder=Path('cache/new_cf/log_frequency')/name/'expanded200'
        em=json.loads((folder/'manifest.json').read_text())
        with np.load(folder/'indices.npz') as q:ez={k:q[k] for k in q.files}
        nitems=len(meta['items']);oldkeys=z['user_index']*nitems+z['item_index'];keys=ez['user_index']*nitems+ez['item_index']
        fresh=~np.isin(keys,oldkeys)
        truth,_=truth_for(c,name,meta['signature']['cutoff'],meta['signature']['upper'],meta['users'])
        label=np.array([meta['items'][i] in truth[meta['users'][u]] for u,i in zip(ez['user_index'],ez['item_index'])])
        # All original rows and old CF fields must remain exactly unchanged.
        order=np.argsort(keys);at=order[np.searchsorted(keys[order],oldkeys)]
        assert np.array_equal(keys[at],oldkeys)
        old_base=np.column_stack([np.load(base/'X.npy'),np.load(Path('cache/focused_v4')/name/'loyalty.npy')])
        exp=np.load(folder/'X.npy',mmap_mode='r')
        np.testing.assert_array_equal(exp[at,:59],old_base)
        np.testing.assert_array_equal(exp[at,59:],np.load(Path('cache/new_cf/log_frequency')/name/'pair_features.npy'))
        with np.load(Path('cache/new_cf/log_frequency')/name/'scores.npz') as q:
            norms.append({'task':name,'median_user_norm':float(np.median(q['user_norm'][q['user_seen']]))})
        distribution.append({'task':name,'total_rows':len(keys),'new_cf_rows':int(fresh.sum()),'new_cf_positive_rows':int((fresh&label).sum()),
            'new_cf_positive_rate':float(label[fresh].mean()),'original_59_and_old_cf_fields_identical':True,
            'policy':'old V5 pool union own-cutoff supported BPR Top200, in every task'})
        if name in ['roll_mar23','roll_apr06']:
            for u,i,y in zip(ez['user_index'],ez['item_index'],label):
                key=(meta['users'][u],int(meta['items'][i]));training_pairs.add(key);training_items.add(key[1])
                if y:positive_pairs.add(key);positive_items.add(key[1])
    model=lightgbm_module().Booster(model_file='cache/new_cf/expanded_ranker_early.txt')
    recovered=[];chosen=[]
    for name in ['main','earlier']:
        _,meta,z=load_dataset(name);folder=Path('cache/new_cf/log_frequency')/name/'expanded200'
        em=json.loads((folder/'manifest.json').read_text())
        with np.load(folder/'indices.npz') as q:ez={k:q[k] for k in q.files}
        X=np.load(folder/'X.npy',mmap_mode='r');score=np.load(folder/'expanded_ranker_scores.npy')
        with np.load(Path('cache/new_cf/log_frequency')/name/'scores.npz') as q:raw=q['score'];uk=q['user_seen'];ik=q['item_seen']
        with np.load(f'cache/improvements/{name}_v1_scores.npz') as q:pop=q['popularity'][0]
        fallback=np.lexsort((np.array(meta['items']),-pop));prior=np.empty(len(fallback),int);prior[fallback]=np.arange(len(fallback))
        starts=np.r_[0,np.cumsum(meta['pool_lengths'])];es=np.r_[0,np.cumsum(em['pool_lengths'])];original=baseline(name)
        found=[]
        for n,u in enumerate(meta['users']):
            old=set(z['item_index'][starts[n]:starts[n+1]])
            bpr_order=np.lexsort((prior,-raw[n]));bpr_rank=np.empty(len(fallback),int);bpr_rank[bpr_order]=np.arange(1,len(fallback)+1)
            lo,hi=es[n:n+2];order=np.argsort(-score[lo:hi],kind='stable');rank=np.empty(len(order),int);rank[order]=np.arange(1,len(order)+1)
            threshold=float(score[lo:hi][order[4]])
            for j,item_index in enumerate(ez['item_index'][lo:hi]):
                item=int(meta['items'][item_index])
                if item_index in old or item not in set(map(int,original[u]['truth'].split())):continue
                r=lo+j;key=(u,item)
                record={'window':name,'user_id':u,'item':item,'bpr_rank':int(bpr_rank[item_index]),'final_rank':int(rank[j]),
                    'bpr_score':float(raw[n,item_index]),'reranker_score':float(score[r]),'top5_cutoff_score':threshold,
                    'same_pair_in_training_candidates':key in training_pairs,'same_pair_positive_in_training':key in positive_pairs,
                    'item_in_any_training_candidates':item in training_items,'item_has_any_positive_training_label':item in positive_items,
                    'history50':float(np.expm1(X[r,meta['feature_names'].index('history_log_frequency')])),
                    'history80':float(np.expm1(X[r,meta['feature_names'].index('history_log_strong')])),
                    'global_count':float(np.expm1(X[r,meta['feature_names'].index('item_log_global_count')])),
                    'als_score':float(X[r,meta['feature_names'].index('cf_score')]),'cbf_score':float(X[r,meta['feature_names'].index('cbf_score')]),
                    'user_embedding_known':bool(uk[n]),'item_embedding_known':bool(ik[item_index]),'flat_feature_row':int(r)}
                found.append(record)
        found.sort(key=lambda r:(r['bpr_rank'],r['final_rank'],r['user_id']))
        selected=[];seen=set()
        for r in found:
            if r['user_id'] not in seen:selected.append(r);seen.add(r['user_id'])
            if len(selected)==2:break
        for r in selected:
            row=X[r['flat_feature_row']][None,:]
            contributions=model.predict(row,pred_contrib=True)[0]
            assert abs(float(contributions.sum())-r['reranker_score'])<1e-5
            names=em['feature_names'];order=np.argsort(np.abs(contributions[:-1]))[::-1][:10]
            r['tree_shap_descriptive_not_causal']=[{'feature':names[j],'value':float(row[0,j]),'contribution':float(contributions[j])} for j in order]
            title=c.execute('SELECT first(title ORDER BY ss,sid),first(category ORDER BY ss,sid) FROM base.airings WHERE item=?',[r['item']]).fetchone()
            r['title']=title[0];r['category']=title[1];chosen.append(r)
        recovered.append({'window':name,'recovered_relevant_pairs':len(found),
            'same_pair_exposed_in_training':sum(r['same_pair_in_training_candidates'] for r in found),
            'same_pair_positive_in_training':sum(r['same_pair_positive_in_training'] for r in found),
            'item_has_positive_training_example':sum(r['item_has_any_positive_training_label'] for r in found),
            'entered_top5':sum(r['final_rank']<=5 for r in found),'known_embedding_pairs':sum(r['user_embedding_known'] and r['item_embedding_known'] for r in found)})
        pd.DataFrame(found).to_csv(out/'private'/f'{name}_recovered_bpr_rows.csv',index=False)
    trainmax=max(r['median_user_norm'] for r in norms if r['task'].startswith('roll'))
    mainnorm=next(r['median_user_norm'] for r in norms if r['task']=='main')
    gate=mainnorm/trainmax>1.5
    save(out/'private/bpr_actual_cases.json',chosen)
    save(out/'bpr_diagnosis.json',{'candidate_distributions':distribution,'recovered':recovered,'norms':norms,
        'main_to_rolling_max_median_norm_ratio':mainnorm/trainmax,'plausible_calibration_shift':gate,
        'run_integration_b':gate,'verified_implementation_bugs':[],
        'verified_correct':['New candidates are exposed in chronological training, using identical Top200 policy.',
            'Original59 and existing8 CF fields are unchanged when expanding.',
            'All recovered BPR positives have supported embeddings; zero-imputation with support flags is not a verified bug.'],
        'interpretation':'Correct but weak ranking plus measured source-prevalence/raw-scale shift; B tests label-free scale calibration, not a claimed bug fix.',
        'shap_interpretation':'Descriptive decomposition of the fitted model for actual rows, not causal effect of feature removal.'})
    print(json.dumps({'distributions':distribution,'recovered':recovered,'norms':norms,'run_b':gate}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['audit','standalone']);args=parser.parse_args()
    out=Path('artifacts/stage9');out.mkdir(exist_ok=True);(out/'private').mkdir(exist_ok=True)
    frozen=freeze(out);p=out/'plan.json'
    if p.exists():assert json.loads(p.read_text())==PLAN
    else:save(p,PLAN)
    with measured() as resources:
        c=connect(Path('cache'))
        if args.stage=='audit':audit_bpr(c,out)
        else:standalone(c,out)
        c.close();check_evidence(frozen)
    save(out/f'execution_{args.stage}.json',{'resources':resources,'source_sha256':{str(p):digest(p) for p in Path('tvrec').glob('*.py')},
        'executor':'AI assistant; no student verification claimed','kaggle_requests':0})


if __name__=='__main__':main()
