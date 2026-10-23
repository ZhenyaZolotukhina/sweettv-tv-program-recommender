"""Freeze the measured Stage-2 winner and refit exactly once on weeks 11--30.

python -m tvrec.final_refit
This module never invokes earlier experiments and never sends a Kaggle request.
"""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from .advanced import metadata
from .core import ap_at_5, match_disjoint, write_submission
from .labels import continuous_sessions, event_frame, training_features
from .models import (ContentProfiles, ImplicitALS, matrix_from_features, measured,
    popularity_scores, score_features, ranked_predictions, reciprocal_ranks)
from .pipeline import LOW, SPLIT, TEST, END, save


def digest(path):
    sha=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            sha.update(chunk)
    return sha.hexdigest()


def verify_selected(strategy,board,fusions):
    expected={'type':'rrf','k':10,'top':200,
              'weights':{'repeat':1.,'cf':.1,'cbf':.1,'popularity':.1}}
    assert strategy['selected_family']=='hybrid'
    assert strategy['selected_config']==strategy['hybrid_config']==expected
    assert abs(strategy['primary_map5']-.37363117283950614)<1e-14
    best=max(board,key=lambda row:row['map5'])
    assert best['family']=='hybrid' and best['config']==expected
    assert best['map5']==strategy['primary_map5']==max(r['map5'] for r in fusions)
    assert {r['family'] for r in board}>={'cf','cbf','repeat','popularity'}
    assert all(r['denominator']==1260 for r in board+fusions)
    for name in ['repeat','popularity']:
        assert strategy['individual_configs'][name]=={'minimum':.5,'strength':'frequency','half_life':0}
    assert strategy['individual_configs']['cf']=={'minimum':.5,'factors':32,'alpha':10.,
        'regularization':1.,'iterations':12,'seed':42}
    assert strategy['individual_configs']['cbf']=={'minimum':.5,'rich':True,'half_life':0}
    assert strategy['next_final_refit_window']==[LOW,TEST]
    return {'verified':True,'local_validation_map5':best['map5'],
            'winner_name':best['name'],'independent_methods_preserved':['cf','cbf']}


def snapshot_evidence(artifacts):
    """Content-address all earlier results without changing them."""
    return {str(p):{'bytes':p.stat().st_size,'sha256':digest(p)}
            for p in sorted(artifacts.rglob('*')) if p.is_file()
            and 'final' not in p.relative_to(artifacts).parts}


def check_evidence(snapshot):
    changed=[name for name,info in snapshot.items()
             if not Path(name).is_file() or digest(name)!=info['sha256']]
    if changed:
        raise RuntimeError(f'Frozen evidence changed: {changed}')
    return True


def credential_probe():
    """Only booleans, never credential contents, paths, cookies or account IDs."""
    config=Path(os.environ.get('KAGGLE_CONFIG_DIR',str(Path.home()/'.kaggle'))).expanduser()
    checks={'api_token_environment_present':bool(os.environ.get('KAGGLE_API_TOKEN')),
        'legacy_environment_pair_present':bool(os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')),
        'configured_kaggle_json_present':(config/'kaggle.json').is_file(),
        'configured_access_token_present':(config/'access_token').is_file(),
        'kaggle_cli_present':shutil.which('kaggle') is not None,
        'kaggle_package_present':importlib.util.find_spec('kaggle') is not None}
    available=any(checks[k] for k in ['api_token_environment_present','legacy_environment_pair_present',
        'configured_kaggle_json_present','configured_access_token_present'])
    return {'checked_at_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,
        'credential_material_present':available,'authenticated_account_verified':False,
        'rules_accepted':None,'submission_access_verified':False,'submission_attempts':0,
        'status':'MANUAL_UPLOAD_REQUIRED' if not available else 'CREDENTIAL_PRESENT_ACCOUNT_NOT_YET_VERIFIED',
        'reason':'No Kaggle API token, legacy credential pair, configured kaggle.json or access_token is available; Kaggle CLI/package also absent.'
                 if not available else 'Credential presence alone is not proof of authentication or accepted rules.',
        'submission_id':None,'public_score':None,'private_score':None,'screenshot_path':None}


def counts(frame,features):
    return {'labelled_continuous_runs':len(frame),'qualifying_distinct_airing_interactions':int(features.frequency.sum()),
        'usable_user_programme_pairs':len(features),'usable_users':int(features.user_id.nunique()),
        'usable_programmes':int(features.item.nunique()),
        'first_run_start':int(frame.a.min()),'last_run_end':int(frame.b.max())}


def assert_history_only(c,frame,features,cutoff=TEST):
    assert (frame.a>=LOW).all() and (frame.b<=cutoff).all() and (frame.b>frame.a).all()
    assert (frame.item>0).all() and (frame.duration>0).all()
    assert (features.last_second<=cutoff).all()
    bounds=c.execute('SELECT min(vs),max(ve),count(*) FILTER(WHERE vs>=? OR ve>?) FROM base.views',[cutoff,cutoff]).fetchone()
    assert bounds[2]==0,'Viewing log contains hidden-period observations'
    hidden=c.execute('''SELECT count(*) FROM continuous_events e JOIN base.airings a USING(sid)
        WHERE a.ss>=? OR a.period=31 OR e.b>?''',[cutoff,cutoff]).fetchone()[0]
    assert hidden==0,'Hidden-period broadcast/interaction entered cached training events'
    return {'passed':True,'raw_view_min_start':bounds[0],'raw_view_max_stop':bounds[1],
        'hidden_period_view_rows':bounds[2],'hidden_period_training_broadcast_or_event_rows':hidden,
        'all_features_clipped_before_prediction_cutoff':True,
        'test_schedule_use':'availability and transformed permitted metadata only; never observed user viewing labels'}


def report_bundle(stage2,out):
    target=out/'private/report_evidence'
    target.mkdir(parents=True,exist_ok=True)
    names=['representative_users.json','cbf_profile_terms.json','coverage_witness.json',
           'source_line_checks.json','actual_manual_ap5.json']
    copied={}
    for name in names:
        source=stage2/'private'/name
        destination=target/name
        if not destination.exists():
            shutil.copyfile(source,destination)
        assert digest(source)==digest(destination)
        copied[name]={'source':str(source),'copy':str(destination),'sha256':digest(destination)}
    examples=json.loads((target/'representative_users.json').read_text())
    for example in examples:
        truth=set(example['truth'])
        details={}
        for method,pred in example['predictions'].items():
            hits=[rank for rank,item in enumerate(pred,1) if item in truth]
            terms=[Fraction(n,rank) for n,rank in enumerate(hits,1)]
            denominator=min(5,len(truth))
            exact=sum(terms,Fraction(0))/denominator if denominator else Fraction(0)
            assert abs(float(exact)-example['ap5'][method])<1e-12
            details[method]={'hit_ranks':hits,'precision_terms':[str(t) for t in terms],
                'denominator':denominator,'exact_ap5_fraction':str(exact),'ap5':float(exact),
                'arithmetic':'(' + (' + '.join(f'{n}/{r}' for n,r in enumerate(hits,1)) or '0') + f') / {denominator}' if denominator else 'empty truth -> 0'}
        example['manual_ap5_details']=details
    save(target/'two_users_exact_ap5.json',{'executor':'AI assistant; not student verification','examples':examples})
    return {'byte_exact_stage2_copies':copied,'two_user_exact_calculations':str(target/'two_users_exact_ap5.json'),
        'comparison':'artifacts/stage2/leaderboard.json','label_repair':'artifacts/stage2/old_vs_corrected.csv'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('sweettv-tv-program-recommender'))
    parser.add_argument('--cache',type=Path,default=Path('cache'))
    parser.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/final'))
    parser.add_argument('--archive',type=Path,default=Path('repository_artifacts/submission_hybrid_v1.csv'))
    args=parser.parse_args()
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    stage2=args.artifacts/'stage2'
    strategy=json.loads((stage2/'selected_strategy.json').read_text())
    winner_check=verify_selected(strategy,json.loads((stage2/'leaderboard.json').read_text()),
                                 json.loads((stage2/'fusion_results.json').read_text()))
    path=out/'submission_hybrid_v1.csv'
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text())
        assert digest(path)==existing['submission']['sha256']
        check_evidence(json.loads((out/'frozen_evidence_manifest.json').read_text())['files'])
        print('Existing frozen hybrid v1 reused; no refit or submission repeated.')
        return
    if path.exists() or args.archive.exists():
        raise RuntimeError('An existing candidate/archive must not be silently overwritten')
    frozen=out/'frozen_evidence_manifest.json'
    if not frozen.exists():
        save(frozen,{'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'files':snapshot_evidence(args.artifacts)})
    previous=json.loads(frozen.read_text())['files']
    check_evidence(previous)
    shutil.copyfile(stage2/'selected_strategy.json',out/'frozen_selected_strategy.json')
    reference=json.loads((stage2/'execution_run.json').read_text())['source_sha256']
    algorithm_files=['tvrec/labels.py','tvrec/models.py','tvrec/advanced.py','tvrec/core.py','tvrec/pipeline.py']
    assert all(digest(name)==reference[name] for name in algorithm_files),'Stage-2 model/matching source changed'
    original=json.loads((args.cache/'manifest.json').read_text())
    for name,stat in original['stats'].items():
        actual=(args.data/name).stat()
        assert actual.st_size==stat['bytes'] and actual.st_mtime_ns==stat['mtime_ns'],f'Input changed: {name}'
    session_manifest=json.loads((args.cache/'stage2_sessions_manifest.json').read_text())
    signature=hashlib.sha256((inspect.getsource(continuous_sessions)+inspect.getsource(match_disjoint)).encode()).hexdigest()
    assert session_manifest=={'algorithm_sha256':signature,'original_manifest':original}
    with (args.data/'submission.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['user_id','tv_show_id']
        users=[row['user_id'] for row in reader]
    assert len(users)==len(set(users))==1260
    c=duckdb.connect(str(args.cache/'stage2.duckdb'),read_only=True)
    c.execute(f"ATTACH '{str((args.cache/'data.duckdb').resolve())}' AS base (READ_ONLY)")
    c.execute("SET threads=4; SET memory_limit='1GB'")
    cache_out=args.cache/'final/hybrid_v1'
    cache_out.mkdir(parents=True,exist_ok=True)
    timings={}
    with measured() as total:
        with measured() as timings['events_and_features']:
            training=event_frame(c,LOW,TEST)
            f=training_features(training,minimum=.5,strength='frequency',half_life=0,cutoff=TEST)
            initial=training.loc[training.a<SPLIT].copy()
            initial['b']=np.minimum(initial.b,SPLIT)
            initial_f=training_features(initial,minimum=.5,strength='frequency',half_life=0,cutoff=SPLIT)
            initial_counts=counts(initial,initial_f)
            final_counts=counts(training,f)
            benchmark=json.loads((stage2/'cf_fresh_fit_benchmark.json').read_text())
            assert len(initial_f)==benchmark['matrix_nonzero'],'Initial interaction definition differs from Stage 2'
            leakage=assert_history_only(c,training,f)
            warm=set(training.loc[(training.b-training.a)>=60,'user_id'])
            cold=set(users)-warm
            items=sorted(r[0] for r in c.execute('''SELECT DISTINCT item FROM base.airings
                WHERE period=31 AND item>0 AND duration>0 AND ss<? AND se>?''',[END,TEST]).fetchall())
            assert len(items)>=5
            all_users=sorted(set(training.user_id)|set(users))
            all_items=sorted(set(int(i) for i in training.item)|set(items))
            ui={u:i for i,u in enumerate(all_users)}
            ii={i:n for n,i in enumerate(all_items)}
            user_indices=np.array([ui[u] for u in users])
            item_indices=np.array([ii[i] for i in items])
            save(cache_out/'mappings.json',{'model_users':all_users,'model_items':all_items,
                'template_users':users,'test_candidates':items})
            del initial,initial_f
            gc.collect()
        with measured() as timings['repeat_popularity']:
            scores={'repeat':score_features(f,users,items),'popularity':popularity_scores(f,users,items)}
            fallback=np.lexsort((np.array(items),-scores['popularity'][0]))
        with measured() as timings['als_refit_and_score']:
            matrix=matrix_from_features(f,all_users,all_items)
            cfg=strategy['individual_configs']['cf']
            als=ImplicitALS(**{k:v for k,v in cfg.items() if k!='minimum'}).fit(matrix)
            with threadpool_limits(limits=1):
                scores['cf']=np.maximum((als.user_factors[user_indices]@als.item_factors[item_indices].T).astype(np.float32),0)
            np.savez_compressed(cache_out/'als_factors.npz',user_factors=als.user_factors,item_factors=als.item_factors)
            sparse.save_npz(cache_out/'training_matrix.npz',matrix)
            matrix_shape=list(matrix.shape)
            matrix_nonzero=matrix.nnz
        with measured() as timings['cbf_refit_and_score']:
            meta_train,meta_all=metadata(c,TEST,END,all_items)
            assert all(LOW<=max(row['ss'],LOW)<TEST for row in meta_train)
            cbf=ContentProfiles(rich=True,max_features=40000).fit_items(meta_train,meta_all)
            weights=matrix_from_features(f,users,all_items)
            with threadpool_limits(limits=1):
                scores['cbf'],profiles=cbf.score(weights,item_indices)
            with (cache_out/'tfidf_vectorizer.pkl').open('wb') as g:
                pickle.dump(cbf.vectorizer,g,protocol=pickle.HIGHEST_PROTOCOL)
            sparse.save_npz(cache_out/'item_vectors.npz',cbf.item_vectors)
            vocabulary_size=len(cbf.vectorizer.vocabulary_)
            metadata_items=len(meta_train)
        with measured() as timings['rrf_and_csv']:
            cfg=strategy['selected_config']
            ranks={name:reciprocal_ranks(scores[name],fallback,top=cfg['top'],k=cfg['k'])
                   for name in ['repeat','cf','cbf','popularity']}
            fused=sum(weight*ranks[name] for name,weight in cfg['weights'].items())
            predictions=ranked_predictions(fused,users,items,fallback,cold)
            # One final candidate only; component outputs stay in ignored cache.
            write_submission(path,users,predictions,set(items))
            np.savez_compressed(cache_out/'component_scores.npz',**scores,hybrid=fused,fallback=fallback)
            args.archive.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,args.archive)
            assert digest(path)==digest(args.archive)
        reports=report_bundle(stage2,out)
        check_evidence(previous)
    candidate_seen=set(int(i) for i in f.item)&set(items)
    initial_benchmark_users=benchmark['matrix_shape'][0]
    auth=credential_probe()
    save(out/'kaggle_access.json',auth)
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'executor':'AI assistant; no student-performed verification claimed',
        'input_files':original['files'],'input_integrity':'Prior SHA256s reused after exact size/mtime checks; raw template separately read and parsed',
        'split_boundaries':{'initial_train':'[2020-03-09,2020-05-18)',
            'local_validation':'[2020-05-18,2020-07-27)','final_refit':'[2020-03-09,2020-07-27)',
            'hidden_evaluation':'[2020-07-27,2020-10-19)','naive_original_clock':True},
        'strategy':strategy,'winner_verification':winner_check,'strategy_sha256':digest(stage2/'selected_strategy.json'),
        'initial_training_counts':initial_counts,'final_refit_counts':final_counts,
        'count_increases':{name:final_counts[name]-initial_counts[name] for name in
            ['qualifying_distinct_airing_interactions','usable_user_programme_pairs','usable_users','usable_programmes']},
        'leakage_checks':leakage,'matrix_shape':matrix_shape,'matrix_nonzero':matrix_nonzero,
        'mapping_rebuilt':True,'initial_mapping_users':initial_benchmark_users,
        'metadata':{'vocabulary_refitted':True,'fit_period':'weeks 11-30 only','training_metadata_items':metadata_items,
            'vocabulary_size':vocabulary_size,'max_features':40000,'initial_vocabulary_size':20258,
            'test_metadata_transformed_with_frozen_vocabulary':True},
        'availability':{'candidates':len(items),'period':31,'window':[TEST,END],
            'positive_labelled_valid_airings_only':True,'seen_in_qualifying_refit':len(candidate_seen),
            'repeats_allowed':True,'cold_template_users':len(cold),'warm_template_users':len(users)-len(cold),
            'cold_definition':'no >=60s unambiguous historical run; all components use the same popularity fallback'},
        'determinism':{'seed':42,'blas_threads':1,'duckdb_threads':4,
            'tie_break':'positive model scores; training-popularity order, then numeric item ID',
            'rrf':'weighted sum of 1/(10+rank) for each component positive-score Top-200'},
        'timings':timings,'total_refit_resources':total,
        'submission':{'path':str(path),'sha256':digest(path),'rows':len(users),
            'columns':['user_id','tv_show_id'],'pre_save_and_roundtrip_validation':True,
            'independent_validation':None,'archive_path':str(args.archive),'archive_sha256':digest(args.archive),
            'not_preliminary_baseline':True},
        'model_artifacts':{str(p):digest(p) for p in sorted(cache_out.glob('*')) if p.is_file()},
        'earlier_results_unchanged':True,'frozen_evidence_manifest':str(frozen),'report_evidence':reports,
        'regression_tests':None,'kaggle':auth,
        'environment':{'python':sys.version,'platform':platform.platform(),'duckdb':duckdb.__version__,
            'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'source_sha256':{str(p):digest(p) for p in sorted(Path('tvrec').glob('*.py'))},
        'limitations':['Official >=80% Top-5 target is confirmed; uninterrupted/concurrent-device and frequency-tie semantics remain reconstructed assumptions.',
            'Original timestamps lack timezone metadata; known clock-change discrepancy remains documented.',
            'Local validation is not a Kaggle score; same-holdout model selection can be optimistic.',
            'Kaggle authentication/account acceptance cannot be verified without account access.']}
    save(manifest_path,manifest)
    c.close()
    print(json.dumps({'initial':initial_counts,'final':final_counts,'candidates':len(items),
        'cold_template_users':len(cold),'vocabulary_size':vocabulary_size,
        'refit_seconds':total['seconds'],'peak_rss_bytes':total['peak_rss_bytes'],
        'submission_path':str(path),'sha256':manifest['submission']['sha256'],'kaggle_status':auth['status']}),flush=True)


if __name__=='__main__':
    main()
