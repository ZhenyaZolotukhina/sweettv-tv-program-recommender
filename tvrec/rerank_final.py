"""Conditional V3 final refit; no claimed evaluation on final training labels.

python -m tvrec.rerank_final
Reported holdout scores remain those of the separate early-only model.
"""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .core import write_submission
from .final_refit import digest,check_evidence
from .improve import connect
from .models import measured
from .pipeline import SPLIT,TEST,END,save
from .rerank import PLAN,audit_training_cutoffs,truth_for,query_predictions,reference_predictions
from .rerank_data import EARLIER,cached_feature_dataset


def main():
    out=Path('artifacts/stage4');cache=Path('cache')
    frozen=json.loads((out/'frozen_prior.json').read_text())['files'];check_evidence(frozen)
    selected=json.loads((out/'selection.json').read_text())
    if selected['selected']=='retain_v2':
        raise RuntimeError('No robust V3 selected; no cosmetic file will be created')
    style=selected['selected']
    assert style in PLAN['prediction_variants']
    path=Path('artifacts/final/submission_hybrid_v3.csv')
    manifest_path=path.with_suffix('.manifest.json')
    if manifest_path.exists():
        assert digest(path)==json.loads(manifest_path.read_text())['submission_sha256']
        print('Existing frozen V3 reused; no fitting or submission repeated.')
        return
    if path.exists():
        raise RuntimeError('Refuse to overwrite existing V3')
    c=connect(cache)
    users=[u for u, in c.execute('SELECT user_id FROM base.template ORDER BY source_row').fetchall()]
    vocab=json.loads((out/'metadata_feature_vocabulary.json').read_text())
    folds=PLAN['training_folds']+[
        {'name':'earlier','cutoff':EARLIER,'target_end':SPLIT},
        {'name':'main','cutoff':SPLIT,'target_end':TEST}]
    audit_training_cutoffs(folds,TEST)
    xs=[];ys=[];weights=[];records=[]
    final_model_path=cache/'rerank/model_final_allowed_history.pkl'
    with measured() as resources:
        for fold in folds:
            folder,meta=cached_feature_dataset(c,cache,users,fold['name'],fold['cutoff'],fold['target_end'],vocab)
            with np.load(folder/'indices.npz') as loaded:
                z={k:loaded[k] for k in loaded.files}
            truth,_=truth_for(c,fold['name'],fold['cutoff'],fold['target_end'],users)
            y=np.array([int(meta['items'][i] in truth[users[u]]) for u,i in zip(z['user_index'],z['item_index'])],dtype=np.uint8)
            X=np.load(folder/'X.npy')
            if xs:
                assert meta['feature_names']==feature_names
            feature_names=meta['feature_names']
            w=1/np.array(meta['pool_lengths'])[z['user_index']]
            w=w*np.where(y,10.,1.);w=w/w.mean()
            xs.append(X);ys.append(y);weights.append(w)
            records.append(fold|{'rows':len(y),'positive_rows':int(y.sum()),
                'feature_manifest_sha256':digest(folder/'features_manifest.json')})
        X=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(weights)
        del xs,ys,weights
        with measured() as fit_resources:
            with threadpool_limits(limits=2):
                model=HistGradientBoostingClassifier(**PLAN['model']).fit(X,y,sample_weight=w)
        with final_model_path.open('wb') as f:
            pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)
        del X,y,w
        import gc;gc.collect()
        with measured() as feature_resources:
            folder,meta=cached_feature_dataset(c,cache,users,'final',TEST,END,vocab)
            assert meta['feature_names']==feature_names
            with np.load(folder/'indices.npz') as loaded:
                z={k:loaded[k] for k in loaded.files}
            baseline=reference_predictions(meta,z)
            with Path('artifacts/final/submission_hybrid_v2.csv').open(newline='') as f:
                original={r['user_id']:list(map(int,r['tv_show_id'].split())) for r in csv.DictReader(f)}
            assert baseline==original,'Final base features no longer reproduce frozen V2'
        with measured() as scoring_resources:
            with threadpool_limits(limits=2):
                probabilities=model.predict_proba(np.load(folder/'X.npy',mmap_mode='r'))[:,1]
            np.save(folder/'final_model_probabilities.npy',probabilities)
            predictions=query_predictions(meta,z,probabilities,style)
            write_submission(path,users,predictions,set(meta['items']))
        assert c.execute('SELECT max(ve) FROM base.views').fetchone()[0]<TEST
        check_evidence(frozen)
    rows=sum(r['rows'] for r in records)
    positives=sum(r['positive_rows'] for r in records)
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'executor':'AI assistant; no student verification claimed',
        'submission_path':str(path),'submission_sha256':digest(path),'rows':len(users),'candidates':len(meta['items']),
        'strategy':style,'model_parameters':PLAN['model'],'feature_names':feature_names,
        'retrieval':PLAN['candidates'],'fusion':'equal reciprocal rank of frozen V2 and learned scores, k=10, within retrieval pool',
        'cold_policy':PLAN['cold_policy'],'cold_users':int(z['cold'].sum()),'repeats_allowed':True,
        'final_supervised_folds':records,'training_rows':rows,'training_positive_rows':positives,
        'final_training_label_end':TEST,'test_features_cutoff':TEST,'target_end':END,
        'reported_holdout_model':{'path':str(cache/'rerank/model_early_only.pkl'),
            'sha256':digest(cache/'rerank/model_early_only.pkl'),'last_training_label_end':EARLIER,
            'main_or_earlier_holdout_labels_used':False},
        'final_model':{'path':str(final_model_path),'sha256':digest(final_model_path),
            'historical_main_and_earlier_labels_now_allowed_for_final_refit':True,
            'not_evaluated_on_its_training_labels_as_independent_validation':True},
        'metadata_vocabulary':vocab,'main_earlier_reported_scores':selected,
        'resources':resources,'fit_resources':fit_resources,'feature_resources':feature_resources,
        'scoring_resources':scoring_resources,'final_feature_manifest':str(folder/'features_manifest.json'),
        'final_features_sha256':digest(folder/'X.npy'),'final_indices_sha256':digest(folder/'indices.npz'),
        'independent_validation':None,'v1_sha256':digest('artifacts/final/submission_hybrid_v1.csv'),
        'v2_sha256':digest('artifacts/final/submission_hybrid_v2.csv'),'prior_evidence_unchanged':True,
        'frozen_prior_manifest':str(out/'frozen_prior.json'),'kaggle_requests':0,'submitted':False,
        'source_sha256':{str(p):digest(p) for p in Path('tvrec').glob('*.py')},
        'limitations':['Early supervised targets span only 7-14 days, while holdouts span 5-10 weeks; final training includes longer historical targets.',
            'Both heldouts were used to choose pure versus blended output, so bootstrap intervals do not correct selection optimism.',
            'Existing continuity/device, target frequency tie and timezone assumptions remain.']}
    save(manifest_path,manifest)
    print(json.dumps({'path':str(path),'sha256':manifest['submission_sha256'],'training_rows':rows,
        'positives':positives,'seconds':resources['seconds'],'fit_seconds':fit_resources['seconds'],
        'peak_rss_bytes':resources['peak_rss_bytes']}),flush=True)
    c.close()


if __name__=='__main__':
    main()
