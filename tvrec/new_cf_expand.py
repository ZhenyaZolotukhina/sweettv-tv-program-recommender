"""One disclosed candidate-generation follow-up, not another CF/weight search.

The original Top50 gate and failed same-pool result are preserved. Top200 is
tested ONCE because measured CF novelty was concentrated below rank50.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .advanced import metadata,metrics,save_predictions
from .core import ap_at_5
from .final_refit import digest,check_evidence
from .focused_features import loyalty_features
from .group_rank import PARAMETERS,fit_ranker,labels_for_pool,validate_fold_groups,prediction_lists
from .improve import connect
from .ltr_experiment import matrix
from .models import measured
from .new_cf import CONFIGS,freeze,fit_context,ordered,baseline
from .new_cf_rerank import FEATURES,features_for_indices,cutoff_fallback
from .pipeline import save
from .rerank import PLAN as ROLLING,audit_training_cutoffs,truth_for
from .rerank_data import EARLIER,context,build_features,order_row


def expanded_dataset(c,name,config,out):
    destination=Path('cache/new_cf')/config['positive_weight']/name/'expanded200'
    destination.mkdir(exist_ok=True)
    marker=destination/'manifest.json'
    if marker.exists():return destination,json.loads(marker.read_text())
    with measured() as resources:
        old_X,meta,z,names=matrix(name)
        cf_folder,_=fit_context(c,name,config,out)
        with np.load(cf_folder/'scores.npz') as q:
            raw=q['score'];bias=q['item_bias'];unorm=q['user_norm'];inorm=q['item_norm'];uk=q['user_seen'];ik=q['item_seen']
        fallback=cutoff_fallback(name,cf_folder,meta)
        ctx=context(c,Path('cache'),meta['users'],name,meta['signature']['cutoff'],meta['signature']['upper'])
        assert ctx['items']==meta['items']
        _,ctx['metadata']=metadata(c,ctx['cutoff'],ctx['upper'],ctx['items'])
        old_starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
        new_pools=[];final_pools=[];check_rows=[]
        for n,u in enumerate(meta['users']):
            original=z['item_index'][old_starts[n]:old_starts[n+1]]
            proposed=ordered(raw[n],fallback)[:200]
            proposed=proposed[np.isfinite(raw[n,proposed])] if not z['cold'][n] else np.array([],dtype=int)
            added=np.setdiff1d(proposed,original)
            # Three existing candidates per query are rebuilt as identity checks,
            # not new training rows. New candidate features use exact V5 formulas.
            new_pools.append(np.r_[original[:3],added])
            check_rows.extend((old_starts[n]+np.arange(3)).tolist())
            combined=np.r_[original,added]
            order=order_row(ctx['scores']['v2'][n],fallback)
            inverse=np.empty(len(order),int);inverse[order]=np.arange(len(order))
            final_pools.append(combined[np.argsort(inverse[combined],kind='stable')])
        vocab=json.loads(Path('artifacts/stage4/metadata_feature_vocabulary.json').read_text())
        extra53,feature_names,new_ui,new_ii=build_features(ctx,new_pools,vocab)
        assert feature_names==names[:53]
        pu=np.array(meta['users'],dtype=object)[new_ui];pi=np.array(meta['items'])[new_ii]
        channels=loyalty_features(ctx['occurrences'],ctx['future'].drop_duplicates(['channel','ss','duration','item']),pu,pi)
        extra59=np.column_stack([extra53,np.column_stack([channels[n] for n in names[53:]])]).astype(np.float32)
        new_starts=np.r_[0,np.cumsum([len(p) for p in new_pools])]
        check=np.concatenate([new_starts[n]+np.arange(3) for n in range(len(meta['users']))])
        np.testing.assert_allclose(extra59[check],old_X[check_rows],rtol=1e-6,atol=1e-5)
        total=sum(len(p) for p in final_pools)
        X=np.empty((total,59),dtype=np.float32);ui=[];ii=[];cursor=0
        for n,pool in enumerate(final_pools):
            old_ids=z['item_index'][old_starts[n]:old_starts[n+1]]
            existing={int(item):old_starts[n]+j for j,item in enumerate(old_ids)}
            fresh={int(item):new_starts[n]+j for j,item in enumerate(new_pools[n])}
            for item in pool:
                X[cursor]=old_X[existing[int(item)]] if int(item) in existing else extra59[fresh[int(item)]]
                cursor+=1
            ui.extend([n]*len(pool));ii.extend(pool.tolist())
        ui=np.array(ui,dtype=int);ii=np.array(ii,dtype=int)
        cf=features_for_indices(raw,bias,unorm,inorm,uk,ik,ui,ii,fallback)
        X=np.column_stack([X,cf]);np.save(destination/'X.npy',X)
        np.savez_compressed(destination/'indices.npz',user_index=ui,item_index=ii,cold=z['cold'],fallback=fallback)
    result={'users':meta['users'],'items':meta['items'],'pool_lengths':[len(p) for p in final_pools],
        'feature_names':names+FEATURES,'cutoff':ctx['cutoff'],'upper':ctx['upper'],
        'existing_features_rebuilt_sample_match':True,'only_new_pairs_rebuilt':True,'resources':resources,
        'expanded_rows':len(ui),'old_rows':len(z['user_index']),'top_n':200}
    save(marker,result)
    return destination,result


def main():
    out=Path('artifacts/stage8');frozen=freeze(out)
    information=json.loads((out/'information_gate.json').read_text())
    assert information['run_reranker'];config=CONFIGS[information['selected_weighting']]
    plan={'reason':'Top50 adds only3/7 positives, but the measured Top200 adds220/134 outside the old pool. Test the tail once rather than claiming head-only feature integration exploited it.',
        'top_n':200,'downstream_variants_total':2,'cf_configurations_total':2,
        'cf_parameters_changed':False,'new_features_beyond_variant_A':False,
        'original_top50_gate_preserved':True,'adaptive_followup_not_an_unbiased_test':True}
    save(out/'expansion200_plan.json',plan)
    if (out/'decision.json').exists():print('Existing bounded decision reused.');return
    c=connect(Path('cache'));xs=[];ys=[];groups=[];records=[]
    audit_training_cutoffs(ROLLING['training_folds'],EARLIER)
    with measured() as resources:
        for fold in ROLLING['training_folds']:
            folder,meta=expanded_dataset(c,fold['name'],config,out)
            with np.load(folder/'indices.npz') as q:z={k:q[k] for k in q.files}
            truth,_=truth_for(c,fold['name'],fold['cutoff'],fold['target_end'],meta['users'])
            y=labels_for_pool(meta,z,truth)
            audit=validate_fold_groups(z['user_index'],z['item_index'],np.array(meta['pool_lengths']),y)
            records.append(fold|audit);xs.append(np.load(folder/'X.npy'));ys.append(y);groups.extend(meta['pool_lengths'])
        X=np.concatenate(xs);y=np.concatenate(ys);del xs,ys
        with measured() as fit:model=fit_ranker(X,y,groups,parameters=PARAMETERS|{'lambdarank_truncation_level':5})
        model.save_model('cache/new_cf/expanded_ranker_early.txt')
        del X,y
        import gc;gc.collect()
        results=[];seg=pd.read_csv('artifacts/stage4/private/v2_v3_user_deltas.csv',dtype={'user_id':str})
        for name in ['main','earlier']:
            folder,meta=expanded_dataset(c,name,config,out)
            with np.load(folder/'indices.npz') as q:z={k:q[k] for k in q.files}
            with measured() as inference:score=model.predict(np.load(folder/'X.npy',mmap_mode='r'));pred=prediction_lists(meta,z,score)
            np.save(folder/'expanded_ranker_scores.npy',score)
            source=baseline(name);users=meta['users'];truth={u:set(map(int,source[u]['truth'].split())) for u in users}
            warm={u for n,u in enumerate(users) if not z['cold'][n]}
            delta=np.array([ap_at_5(truth[u],pred[u])-float(source[u]['ap5']) for u in users])
            rng=np.random.default_rng(42);boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)])
            group=seg[seg.window==name].set_index('user_id');sub={}
            for label,field,key in [('light','history_group','light_under10'),('heavy','history_group','heavy10plus'),('repeat_heavy','repeat_group','repeat_heavy')]:
                index=[n for n,u in enumerate(users) if group.loc[u,field]==key];sub[label]=float(delta[index].mean())
            starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
            hits=sum(len({meta['items'][i] for i in z['item_index'][starts[n]:starts[n+1]]}&truth[u]) for n,u in enumerate(users))
            row={'variant':'expanded200','window':name,**metrics(truth,pred,users,warm),'gain':float(delta.mean()),
                'paired_95_ci':np.quantile(boot,[.025,.975]).tolist(),'improved':int((delta>0).sum()),'regressed':int((delta<0).sum()),
                'subgroups':sub,'pool_micro_recall':hits/sum(len(t) for t in truth.values()),'mean_pool_size':float(np.mean(meta['pool_lengths'])),
                'fit_resources':fit,'inference_resources':inference,'training_folds':records}
            results.append(row);save(out/'expansion200_results.json',results)
            save_predictions(out/'private'/f'{name}_expanded200.csv',pred,truth,users,warm)
            print(name,'expanded200',row['map5'],row['gain'],row['paired_95_ci'],row['pool_micro_recall'],flush=True)
        c.close();check_evidence(frozen)
    all_rows=json.loads((out/'integration_results.json').read_text())+results
    by={(r['variant'],r['window']):r for r in all_rows}
    eligible=[v for v in ['same_pool_features','expanded200'] if all(by[v,w]['gain']>=.001 and by[v,w]['paired_95_ci'][0]>0
        and by[v,w]['subgroups']['light']>=-.003 and by[v,w]['subgroups']['repeat_heavy']>=-.003 for w in ['main','earlier'])]
    chosen=max(eligible,key=lambda v:sum(by[v,w]['gain'] for w in ['main','earlier'])) if eligible else 'retain_v5'
    save(out/'decision.json',{'selected':chosen,'eligible':eligible,'results':all_rows,'downstream_variants':2,
        'cf_configurations':2,'original_top50_plan_and_gate_preserved':True,'adaptive_tail_followup_disclosed':True,'kaggle_requests':0})
    save(out/'execution_expansion.json',{'resources':resources,'new_cf_fits':0,'kaggle_requests':0})
    print('Decision',chosen)


if __name__=='__main__':main()
