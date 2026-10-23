"""One bounded loss-focus follow-up: truncation5 rather than8, no other search.

The original one-configuration records remain unchanged. Total real-data models
are capped at two; the one previously tested fusion is not re-tuned/repeated.
"""
import json
from pathlib import Path

import numpy as np

from .final_refit import digest,check_evidence
from .group_rank import fit_ranker,labels_for_pool,validate_fold_groups,PARAMETERS,ROUNDS
from .improve import connect
from .ltr_experiment import PLAN,freeze,matrix,evaluate
from .models import measured
from .pipeline import save
from .rerank import audit_training_cutoffs,truth_for
from .rerank_data import EARLIER


def main():
    out=Path('artifacts/stage6');frozen=freeze(out)
    plan={'reason':'A improved both point estimates but main CI crossed zero; test whether focusing lambdas at the actual Top5 improves MAP rather than surrogate@8.',
        'only_change':{'lambdarank_truncation_level':5},'maximum_total_model_configurations':2,
        'additional_fusions':0,'selection_rule_unchanged':PLAN['selection'],
        'adaptive_followup_not_an_independent_test':True}
    pp=out/'top5_followup_plan.json'
    if pp.exists():assert json.loads(pp.read_text())==plan
    else:save(pp,plan)
    if (out/'decision.json').exists():
        print('Existing bounded follow-up reused; no repeated fit.');return
    audit_training_cutoffs(PLAN['training_folds'],EARLIER)
    c=connect(Path('cache'));xs=[];ys=[];groups=[]
    with measured() as resources:
        for fold in PLAN['training_folds']:
            X,meta,z,names=matrix(fold['name'])
            truth,_=truth_for(c,fold['name'],fold['cutoff'],fold['target_end'],meta['users'])
            y=labels_for_pool(meta,z,truth)
            validate_fold_groups(z['user_index'],z['item_index'],np.array(meta['pool_lengths']),y)
            xs.append(X);ys.append(y);groups.extend(meta['pool_lengths'])
        X=np.concatenate(xs);y=np.concatenate(ys);del xs,ys
        with measured() as fit:model=fit_ranker(X,y,groups,parameters=plan['only_change'])
        model.save_model('cache/group_ltr/lambda_top5_early.txt')
        del X,y
        import gc;gc.collect()
        rows=[]
        for window in ['main','earlier']:
            X,meta,z,names=matrix(window)
            with measured() as inference:score=model.predict(X)
            np.save(Path('cache/group_ltr')/(window+'_lambda_top5_scores.npy'),score)
            row,_=evaluate(out,window,'lambda_top5',meta,z,score)
            row['inference_resources']=inference;rows.append(row)
            del X;gc.collect()
        c.close();check_evidence(frozen)
    save(out/'top5_followup_results.json',{'results':rows,'resources':resources,'fit_resources':fit,
        'parameters':PARAMETERS|plan['only_change'],'rounds':ROUNDS,
        'last_training_label_end':EARLIER,'new_holdout_labels_in_fit':False})
    original=json.loads((out/'results.json').read_text())
    combined=original+rows
    by={(r['candidate'],r['window']):r for r in combined}
    eligible=[name for name in sorted({r['candidate'] for r in combined})
        if all(by[name,w]['gain']>=.001 and by[name,w]['paired_95_ci'][0]>0
               and by[name,w]['light_history_gain']>=-.003 for w in ['main','earlier'])]
    chosen=max(eligible,key=lambda k:sum(by[k,w]['gain'] for w in ['main','earlier'])) if eligible else 'retain_v4'
    save(out/'decision.json',{'selected':chosen,'eligible':eligible,'rule':PLAN['selection'],
        'configurations_trained':2,'one_fusion_only':True,'adaptive_followup_disclosed':True,
        'results':combined,'features_pools_and_targets_changed':False,'kaggle_requests':0})
    print('Final decision:',chosen)


if __name__=='__main__':main()
