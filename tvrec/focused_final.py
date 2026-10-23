"""Conditionally refit only the selected final focused block; never submits."""
import json
from pathlib import Path
import pickle
from datetime import datetime,timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .core import write_submission
from .final_refit import digest,check_evidence
from .focused import PLAN,load_dataset,extra_dataset
from .improve import connect
from .models import measured
from .pipeline import SPLIT,TEST,END,save
from .rerank import audit_training_cutoffs,truth_for,query_predictions
from .rerank_data import EARLIER


def main():
    out=Path('artifacts/stage5')
    frozen=json.loads((out/'frozen_prior.json').read_text())['files'];check_evidence(frozen)
    selected=json.loads((out/'selection.json').read_text())
    if selected['selected']=='retain_v3':raise RuntimeError('No robust improvement; no cosmetic V4')
    block=selected['selected']
    assert block in PLAN['experiments'] and selected['experiments_run']<=3
    assert all(selected[w]['paired_95_ci'][0]>0 and selected[w]['light_history_gain']>=-.003 for w in ['main','earlier'])
    csv=Path('artifacts/final/submission_hybrid_v4.csv');manifest_path=csv.with_suffix('.manifest.json')
    if manifest_path.exists():
        assert digest(csv)==json.loads(manifest_path.read_text())['submission_sha256']
        print('Existing V4 reused; no repeated fit/submission.');return
    if csv.exists():raise RuntimeError('Refuse to overwrite V4')
    c=connect(Path('cache'))
    folds=PLAN['training_folds']+[
        {'name':'earlier','cutoff':EARLIER,'target_end':SPLIT},
        {'name':'main','cutoff':SPLIT,'target_end':TEST}]
    audit_training_cutoffs(folds,TEST)
    xs=[];ys=[];weights=[];records=[]
    model_path=Path('cache/focused_v4')/(block+'_final.pkl')
    with measured() as resources:
        for fold in folds:
            base,meta,z=load_dataset(fold['name']);extra,info=extra_dataset(c,fold['name'],out)
            assert info['cutoff']==fold['cutoff']
            truth,_=truth_for(c,fold['name'],fold['cutoff'],fold['target_end'],meta['users'])
            y=np.array([int(meta['items'][i] in truth[meta['users'][u]]) for u,i in zip(z['user_index'],z['item_index'])],dtype=np.uint8)
            X=np.column_stack([np.load(base/'X.npy'),np.load(extra/(block+'.npy'))])
            names=meta['feature_names']+info['columns'][block]
            if xs:assert names==feature_names
            feature_names=names
            w=1/np.array(meta['pool_lengths'])[z['user_index']]*np.where(y,10.,1.);w/=w.mean()
            xs.append(X);ys.append(y);weights.append(w)
            records.append(fold|{'rows':len(y),'positives':int(y.sum()),'base_manifest_sha256':digest(base/'features_manifest.json'),
                                'extra_features_sha256':digest(extra/(block+'.npy'))})
        X=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(weights);del xs,ys,weights
        with measured() as fit_resources:
            with threadpool_limits(limits=2):model=HistGradientBoostingClassifier(**PLAN['model_parameters']).fit(X,y,sample_weight=w)
        with model_path.open('wb') as f:pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)
        del X,y,w
        import gc;gc.collect()
        with measured() as feature_resources:
            base,meta,z=load_dataset('final')
            assert meta['signature']['cutoff']==TEST and meta['signature']['upper']==END
            extra,info=extra_dataset(c,'final',out,blocks=[block])
            X=np.column_stack([np.load(base/'X.npy'),np.load(extra/(block+'.npy'))])
            assert meta['feature_names']+info['columns'][block]==feature_names
        with measured() as scoring_resources:
            with threadpool_limits(limits=2):p=model.predict_proba(X)[:,1]
            np.save(extra/(block+'_final_probabilities.npy'),p)
            predictions=query_predictions(meta,z,p,'v2_supervised_rrf_equal')
            write_submission(csv,meta['users'],predictions,set(meta['items']))
        assert c.execute('SELECT max(ve) FROM base.views').fetchone()[0]<TEST
        check_evidence(frozen)
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'executor':'AI assistant; not student verification',
        'submission_path':str(csv),'submission_sha256':digest(csv),'rows':len(meta['users']),'candidates':len(meta['items']),
        'selected_block':block,'model_parameters':PLAN['model_parameters'],'feature_names':feature_names,
        'added_features':info['columns'][block],'prediction_policy':'Unchanged V3 equal V2/learned RRF, k10, same pool/ties/cold fallback',
        'offline_results':selected,'final_training_folds':records,'last_training_label_end':TEST,
        'reported_validation_last_training_label_end':EARLIER,'reported_validation_model':{
            'path':str(Path('cache/focused_v4')/(block+'_early.pkl')),
            'sha256':digest(Path('cache/focused_v4')/(block+'_early.pkl'))},
        'final_model':{'path':str(model_path),'sha256':digest(model_path),'not_retested_on_training_targets_as_validation':True},
        'final_base_feature_hash':digest(base/'X.npy'),'final_extra_feature_hash':digest(extra/(block+'.npy')),
        'final_probability_hash':digest(extra/(block+'_final_probabilities.npy')),
        'resources':resources,'fit_resources':fit_resources,'feature_resources':feature_resources,'scoring_resources':scoring_resources,
        'source_sha256':{str(p):digest(p) for p in Path('tvrec').glob('*.py')},
        'v1_sha256':digest('artifacts/final/submission_hybrid_v1.csv'),'v2_sha256':digest('artifacts/final/submission_hybrid_v2.csv'),
        'v3_sha256':digest('artifacts/final/submission_hybrid_v3.csv'),'frozen_prior_manifest':str(out/'frozen_prior.json'),
        'prior_evidence_unchanged':True,'new_als_or_cbf_fits':0,'cold_users':int(z['cold'].sum()),'repeats_allowed':True,
        'independent_validation':None,'kaggle_requests':0,'submitted':False,
        'limitations':['Three feature blocks selected on two reused holdouts; bootstrap intervals do not correct adaptive selection.',
            'Target frequency-tie/session/device and timezone ambiguities remain; primary target was not changed.',
            'Known future channel schedule mix is allowed metadata, not future viewing probability.']}
    save(manifest_path,manifest)
    print(json.dumps({'path':str(csv),'sha256':manifest['submission_sha256'],'rows':len(meta['users']),
        'seconds':resources['seconds'],'fit_seconds':fit_resources['seconds'],'peak_rss_bytes':resources['peak_rss_bytes']}),flush=True)
    c.close()


if __name__=='__main__':main()
