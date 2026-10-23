"""Exactly two scoped integrations: AR expansion and justified BPR calibration."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .advanced import metrics,save_predictions
from .association_data import ar_dataset,calibrated_dataset
from .assoc_study import PLAN,freeze,baseline
from .core import ap_at_5
from .final_refit import check_evidence
from .group_rank import PARAMETERS,fit_ranker,labels_for_pool,validate_fold_groups,prediction_lists
from .improve import connect
from .models import measured
from .pipeline import save
from .rerank import audit_training_cutoffs,truth_for
from .rerank_data import EARLIER


def main():
    out=Path('artifacts/stage9');frozen=freeze(out)
    choice=json.loads((out/'association_choice.json').read_text())['selected']
    diagnosis=json.loads((out/'bpr_diagnosis.json').read_text())
    variants=['association']+(['bpr_calibrated'] if diagnosis['run_integration_b'] else [])
    save(out/'integration_plan.json',{'variants':variants,'ar_kind':choice,'association_top_n':100,
        'bpr_variant_justification':'Measured raw-scale shift; original generator exposure and flags were correct, so B changes only calibration/missing handling, not a nonexistent bug.',
        'bpr_candidate_policy':'unchanged previous expanded Top200','parameters':PARAMETERS|{'lambdarank_truncation_level':5},
        'training_folds':PLAN['training_folds'],'no_eval_labels_in_training':True,'kaggle_requests':0})
    audit_training_cutoffs(PLAN['training_folds'],EARLIER)
    c=connect(Path('cache'));rows=[]
    with measured() as resources:
        for variant in variants:
            xs=[];ys=[];groups=[];records=[]
            for fold in PLAN['training_folds']:
                folder,meta=ar_dataset(c,fold['name'],choice) if variant=='association' else calibrated_dataset(fold['name'])
                assert meta['cutoff']==fold['cutoff'] and meta['upper']==fold['target_end']
                with np.load(folder/'indices.npz') as q:z={k:q[k] for k in q.files}
                truth,_=truth_for(c,fold['name'],fold['cutoff'],fold['target_end'],meta['users'])
                y=labels_for_pool(meta,z,truth)
                group=validate_fold_groups(z['user_index'],z['item_index'],np.array(meta['pool_lengths']),y)
                records.append(fold|group);xs.append(np.load(folder/'X.npy'));ys.append(y);groups.extend(meta['pool_lengths'])
            X=np.concatenate(xs);y=np.concatenate(ys);del xs,ys
            with measured() as fit:model=fit_ranker(X,y,groups,parameters={'lambdarank_truncation_level':5})
            model_folder=Path('cache/association_integration');model_folder.mkdir(exist_ok=True)
            model.save_model(str(model_folder/(variant+'_early.txt')))
            del X,y
            import gc;gc.collect()
            segment=pd.read_csv('artifacts/stage4/private/v2_v3_user_deltas.csv',dtype={'user_id':str})
            for window in ['main','earlier']:
                folder,meta=ar_dataset(c,window,choice) if variant=='association' else calibrated_dataset(window)
                assert max(f['target_end'] for f in records)<=meta['cutoff']
                with np.load(folder/'indices.npz') as q:z={k:q[k] for k in q.files}
                with measured() as inference:score=model.predict(np.load(folder/'X.npy',mmap_mode='r'));pred=prediction_lists(meta,z,score)
                np.save(folder/(variant+'_scores.npy'),score)
                original=baseline(window);users=meta['users'];truth={u:set(map(int,original[u]['truth'].split())) for u in users}
                warm={u for n,u in enumerate(users) if not z['cold'][n]}
                delta=np.array([ap_at_5(truth[u],pred[u])-float(original[u]['ap5']) for u in users])
                rng=np.random.default_rng(42);boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)])
                s=segment[segment.window==window].set_index('user_id');subgroups={}
                for label,field,key in [('light','history_group','light_under10'),('heavy','history_group','heavy10plus'),('repeat_heavy','repeat_group','repeat_heavy')]:
                    index=[n for n,u in enumerate(users) if s.loc[u,field]==key];subgroups[label]=float(delta[index].mean())
                offsets=np.r_[0,np.cumsum(meta['pool_lengths'])]
                retrieved=sum(len({meta['items'][i] for i in z['item_index'][offsets[n]:offsets[n+1]]}&truth[u]) for n,u in enumerate(users))
                row={'variant':variant,'window':window,**metrics(truth,pred,users,warm),
                    'gain':float(delta.mean()),'paired_95_ci':np.quantile(boot,[.025,.975]).tolist(),
                    'improved':int((delta>0).sum()),'regressed':int((delta<0).sum()),'subgroups':subgroups,
                    'pool_micro_recall':retrieved/sum(len(t) for t in truth.values()),'mean_pool_size':float(np.mean(meta['pool_lengths'])),
                    'fit_resources':fit,'inference_resources':inference,'feature_resources':meta['resources'],'training_folds':records}
                rows.append(row);save(out/'integration_results.json',rows)
                save_predictions(out/'private'/f'{window}_{variant}.csv',pred,truth,users,warm)
                print(window,variant,row['map5'],row['gain'],row['paired_95_ci'],subgroups,flush=True)
            del model;gc.collect()
        c.close();check_evidence(frozen)
    by={(r['variant'],r['window']):r for r in rows}
    eligible=[v for v in variants if all(by[v,w]['gain']>=.001 and by[v,w]['paired_95_ci'][0]>0
        and all(by[v,w]['subgroups'][g]>=-.003 for g in ['heavy','light','repeat_heavy']) for w in ['main','earlier'])]
    chosen=max(eligible,key=lambda v:sum(by[v,w]['gain'] for w in ['main','earlier'])) if eligible else 'retain_v5'
    save(out/'decision.json',{'selected':chosen,'eligible':eligible,'ar_kind':choice,'results':rows,
        'rule':PLAN['selection'],'experiments_run':len(variants),'target_changed':False,'kaggle_requests':0})
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['fit_resources','inference_resources','feature_resources','training_folds']}
                  for r in rows]).to_csv(out/'comparison.csv',index=False)
    save(out/'execution_integration.json',{'resources':resources,'new_cf_or_base_latent_fits':0,'kaggle_requests':0})
    print('Decision',chosen)


if __name__=='__main__':main()
