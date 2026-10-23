"""Refit only a robustly selected group ranker; never evaluates refit on train."""
import json
from pathlib import Path
from datetime import datetime,timezone

import numpy as np

from .core import write_submission
from .final_refit import digest,check_evidence
from .group_rank import (PARAMETERS,ROUNDS,lightgbm_module,fit_ranker,
    labels_for_pool,validate_fold_groups,prediction_lists)
from .improve import connect
from .ltr_experiment import PLAN,matrix,freeze
from .models import measured
from .pipeline import SPLIT,TEST,END,save
from .rerank import audit_training_cutoffs,truth_for
from .rerank_data import EARLIER


def main():
    out=Path('artifacts/stage6');frozen=freeze(out)
    decision=json.loads((out/'decision.json').read_text());selected=decision['selected']
    if selected=='retain_v4':raise RuntimeError('No robust V5; retain V4 without a cosmetic CSV')
    assert selected in decision['eligible']
    parameters=PARAMETERS.copy()
    if selected=='lambda_top5':parameters['lambdarank_truncation_level']=5
    csv=Path('artifacts/final/submission_hybrid_v5.csv');manifest=csv.with_suffix('.manifest.json')
    if manifest.exists():
        assert digest(csv)==json.loads(manifest.read_text())['submission_sha256']
        print('Existing V5 reused unchanged; no repeat fit/submission.');return
    if csv.exists():raise RuntimeError('Refuse to overwrite an existing V5')
    c=connect(Path('cache'))
    folds=PLAN['training_folds']+[
        {'name':'earlier','cutoff':EARLIER,'target_end':SPLIT},
        {'name':'main','cutoff':SPLIT,'target_end':TEST}]
    audit_training_cutoffs(folds,TEST)
    records=[];xs=[];ys=[];groups=[]
    with measured() as resources:
        for fold in folds:
            X,meta,z,names=matrix(fold['name'])
            if xs:assert names==feature_names
            feature_names=names
            truth,_=truth_for(c,fold['name'],fold['cutoff'],fold['target_end'],meta['users'])
            y=labels_for_pool(meta,z,truth)
            audit=validate_fold_groups(z['user_index'],z['item_index'],np.array(meta['pool_lengths']),y)
            records.append(fold|audit)
            xs.append(X);ys.append(y);groups.extend(meta['pool_lengths'])
        X=np.concatenate(xs);y=np.concatenate(ys);del xs,ys
        with measured() as fit_resources:model=fit_ranker(X,y,groups,parameters=parameters)
        model_file=Path('cache/group_ltr')/(selected+'_final.txt')
        model.save_model(str(model_file))
        del X,y
        import gc;gc.collect()
        X,meta,z,names=matrix('final')
        assert names==feature_names and meta['signature']['cutoff']==TEST and meta['signature']['upper']==END
        with measured() as scoring_resources:
            score=model.predict(X)
            scores_file=Path('cache/group_ltr/final_selected_scores.npy');np.save(scores_file,score)
            old=None
            if selected=='lambda_v4_rrf':old=np.load('cache/focused_v4/final/loyalty_final_probabilities.npy')
            pred=prediction_lists(meta,z,score,old,selected=='lambda_v4_rrf')
            write_submission(csv,meta['users'],pred,set(meta['items']))
        assert c.execute('SELECT max(ve) FROM base.views').fetchone()[0]<TEST
        check_evidence(frozen)
    early_file=Path('cache/group_ltr')/('lambda_top5_early.txt' if selected=='lambda_top5' else 'lambda_early.txt')
    data={'created_at_utc':datetime.now(timezone.utc).isoformat(),'executor':'AI assistant; no student verification claimed',
        'submission_path':str(csv),'submission_sha256':digest(csv),'rows':len(meta['users']),'candidates':len(meta['items']),
        'selected':selected,'parameters':parameters,'rounds':ROUNDS,'lightgbm':lightgbm_module().__version__,
        'feature_names':feature_names,'features_and_pools_identical_to_v4':True,
        'ranking_group':'user x historical forecast task, contiguous rows, separate query per fold',
        'group_records':records,'training_queries':len(groups),'training_rows':sum(r['rows'] for r in records),
        'training_positives':sum(r['positives'] for r in records),'last_final_label_end':TEST,
        'reported_holdout_model':{'path':str(early_file),'sha256':digest(early_file),'last_label_end':EARLIER,
            'main_or_earlier_holdout_labels_used':False},
        'final_model':{'path':str(model_file),'sha256':digest(model_file),'not_evaluated_on_training_targets_as_validation':True},
        'reported_validation':decision,'fit_resources':fit_resources,'scoring_resources':scoring_resources,'resources':resources,
        'final_scores_path':str(scores_file),'final_scores_sha256':digest(scores_file),
        'final_base_features_sha256':digest('cache/rerank/final/X.npy'),
        'final_loyalty_features_sha256':digest('cache/focused_v4/final/loyalty.npy'),
        'final_indices_sha256':digest('cache/rerank/final/indices.npz'),
        'cold_policy':'unchanged V4 popularity fallback','cold_users':int(z['cold'].sum()),'repeats_allowed':True,
        'frozen_prior_manifest':str(out/'frozen_prior.json'),'prior_evidence_unchanged':True,'independent_validation':None,
        'v1_v4_sha256':{f'v{n}':digest(f'artifacts/final/submission_hybrid_v{n}.csv') for n in [1,2,3,4]},
        'source_sha256':{str(p):digest(p) for p in Path('tvrec').glob('*.py')},
        'optional_requirements_sha256':digest('requirements-ltr.txt'),
        'kaggle_requests':0,'submitted':False,
        'limitations':['Two ranking configurations and one fusion tested on reused holdouts; paired CIs do not correct adaptive selection.',
            'LambdaRank optimizes an NDCG-based surrogate, not exact MAP; reported comparison uses exact MAP5.',
            'Ranking library/regularization and weighting differ from the classifier; gain is not a perfect objective-only causal experiment.',
            'Original target ties, uninterrupted/device interpretation and timezone remain uncertain.']}
    save(manifest,data)
    print(json.dumps({'path':str(csv),'sha256':data['submission_sha256'],'queries':len(groups),
        'training_rows':data['training_rows'],'seconds':resources['seconds'],
        'fit_seconds':fit_resources['seconds'],'peak_rss_bytes':resources['peak_rss_bytes']}),flush=True)
    c.close()


if __name__=='__main__':main()
