"""Independent witness/CSV checks and one isolated winning-ALS fit benchmark.

python -m tvrec.verify_stage2
python -m tvrec.verify_stage2 --benchmark-cf
"""
import argparse
import csv
import json
import hashlib
import platform
import sys
from pathlib import Path

import duckdb
import numpy as np

from .core import ap_at_5
from .labels import event_frame, training_features
from .models import ImplicitALS, ContentProfiles, matrix_from_features, measured
from .pipeline import LOW, SPLIT, TEST, save


def brute_continuous_coverage(views,airings,channel,sid,lower,upper):
    """Independent endpoint enumeration for ONE witness; no production join code.

    views: (channel,start,stop); airings: (sid,start,end). Returns longest unique-
    channel run owned by the requested airing under the documented conflict rule.
    """
    endpoints={lower,upper}
    for _,start,stop in list(views)+list(airings):
        if lower<start<upper:
            endpoints.add(start)
        if lower<stop<upper:
            endpoints.add(stop)
    points=sorted(endpoints)
    longest=0
    current=0
    for start,stop in zip(points,points[1:]):
        active={ch for ch,a,b in views if a<=start and b>=stop}
        owners=[(a,-i,i) for i,a,b in airings if a<=start and b>=stop]
        winner=max(owners)[2] if owners else None
        if active=={channel} and winner==sid:
            current+=stop-start
            longest=max(longest,current)
        else:
            current=0
    return longest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('cache'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/stage2'))
    parser.add_argument('--benchmark-cf',action='store_true')
    args=parser.parse_args()
    out=args.out
    c=duckdb.connect(str(args.cache/'stage2.duckdb'),read_only=True)
    c.execute(f"ATTACH '{str((args.cache/'data.duckdb').resolve())}' AS base (READ_ONLY)")
    c.execute('SET threads=4')
    witness=json.loads((out/'private/coverage_witness.json').read_text())
    airing=c.execute('SELECT ss,se,duration,channel,item FROM base.airings WHERE sid=?',[witness['sid']]).fetchone()
    ss,se,duration,channel,item=airing
    views=c.execute('SELECT channel,vs,ve FROM base.views WHERE user_id=? AND vs<? AND ve>?',
                    [witness['user_id'],se,ss]).fetchall()
    airings=c.execute('SELECT sid,ss,se FROM base.airings WHERE channel=? AND ss<? AND se>? AND se>ss',
                      [channel,se,ss]).fetchall()
    seconds=brute_continuous_coverage(views,airings,channel,witness['sid'],max(SPLIT,ss),min(TEST,se))
    assert seconds==witness['overlap_seconds']
    assert 5*seconds>4*duration
    users=[r[0] for r in c.execute('SELECT user_id FROM base.template ORDER BY source_row').fetchall()]
    candidates={r[0] for r in c.execute('SELECT DISTINCT item FROM base.airings WHERE period=21 AND item>0 AND duration>0 AND ss<? AND se>?',[TEST,SPLIT]).fetchall()}
    truth=json.loads((out/'private/truth_versions.json').read_text())['official_ge80_top5']
    truth={u:set(v) for u,v in truth.items()}
    all_checks=[]
    for path in sorted((out/'private').glob('main_*.csv')):
        with path.open(newline='') as f:
            rows=list(csv.DictReader(f))
        assert [r['user_id'] for r in rows]==users,path
        aps=[]
        for r in rows:
            u=r['user_id']
            pred=[int(i) for i in r['predictions'].split()]
            assert len(pred)==len(set(pred))==5,path
            assert set(pred)<=candidates,path
            assert set(map(int,r['truth'].split()))==truth[u],path
            ap=ap_at_5(truth[u],pred)
            assert abs(ap-float(r['ap5']))<1e-12,path
            assert [i for i,p in enumerate(pred,1) if p in truth[u]]==list(map(int,r['hit_ranks'].split())),path
            aps.append(ap)
        all_checks.append({'file':str(path),'rows':len(rows),'map5':float(np.mean(aps))})
    board=json.loads((out/'leaderboard.json').read_text())
    lookup={Path(r['file']).stem:r for r in all_checks}
    for result in board:
        filename=result['name'] if result['family']=='hybrid' else 'main_'+result['name']
        assert abs(lookup[filename]['map5']-result['map5'])<1e-12
    result={'independent_coverage_verified':True,'overlap_seconds':seconds,'programme_duration':duration,
        'fraction':seconds/duration,'strict_80':True,'integer_arithmetic':f'{5*seconds} > {4*duration}',
        'method_prediction_files_checked':len(all_checks),'rows_per_method':1260,
        'same_candidates_users_truth_each_method':True,'all_per_user_aps_and_hit_ranks_recomputed':True,
        'files':all_checks,'no_raw_csv_rescan':True}
    save(out/'independent_verification.json',result)
    if args.benchmark_cf:
        cfg=json.loads((out/'selected_strategy.json').read_text())['individual_configs']['cf']
        with measured() as timing:
            train=event_frame(c,LOW,SPLIT)
            training=training_features(train,minimum=cfg['minimum'],strength='frequency',cutoff=SPLIT)
            model_users=sorted(set(train.user_id)|set(users))
            model_items=sorted(set(int(i) for i in train.item)|candidates)
            matrix=matrix_from_features(training,model_users,model_items)
            model=ImplicitALS(**{k:v for k,v in cfg.items() if k!='minimum'}).fit(matrix)
        save(out/'cf_fresh_fit_benchmark.json',{'config':cfg,**timing,'matrix_shape':list(matrix.shape),
            'matrix_nonzero':matrix.nnz,'why_run':'Earlier main run reused fitted factors but had not persisted original fit RSS. One isolated winning-config refit measures actual fit resources; no hyperparameter search.',
            'includes':'cached training event read, feature aggregation, sparse matrix and fresh ALS fit; no validation-view fitting',
            'deterministic_factor_checksum':float(np.linalg.norm(model.user_factors))})
    examples=json.loads((out/'private/representative_users.json').read_text())
    summary=[]
    for n,ex in enumerate(examples):
        metadata={int(r['item']):r for r in ex['metadata']}
        summary.append({'alias':'U-heavy' if n==0 else 'U-light','selection_reason':ex['original_selection_reason'],
            'corrected_truth_count':len(ex['truth']),'training_runs':len(ex['training_continuous_runs']),
            'validation_runs':len(ex['validation_continuous_runs']),'warm':ex['warm'],'ap5':ex['ap5'],
            'recommendations':{name:[metadata[i] for i in rec] for name,rec in ex['predictions'].items()},
            'private_user_id_location':'private/representative_users.json'})
    save(out/'representative_user_summary.json',summary)
    # Reconstruct only two CBF profiles to preserve actual feature evidence.
    # This is verification, not additional tuning or another full dense scoring run.
    from .advanced import metadata
    chosen=json.loads((out/'selected_strategy.json').read_text())
    cfg=chosen['individual_configs']['cbf']
    train=event_frame(c,LOW,SPLIT)
    model_items=sorted(set(int(i) for i in train.item)|candidates)
    meta_train,meta_all=metadata(c,SPLIT,TEST,model_items)
    profile_users=[ex['user_id'] for ex in examples]
    f=training_features(train[train.user_id.isin(profile_users)],minimum=cfg['minimum'],
        half_life=cfg['half_life'],strength='frequency',cutoff=SPLIT)
    cbf=ContentProfiles(rich=cfg['rich']).fit_items(meta_train,meta_all)
    weights=matrix_from_features(f,profile_users,model_items)
    item_lookup={i:n for n,i in enumerate(model_items)}
    _,profiles=cbf.score(weights,[item_lookup[i] for i in sorted(candidates)])
    terms=cbf.vectorizer.get_feature_names_out()
    details=[]
    for n,user in enumerate(profile_users):
        profile=profiles.getrow(n)
        order=np.argsort(-profile.data,kind='stable')[:20]
        ids=set(int(i) for i in f.loc[f.user_id==user,'item'])
        details.append({'user_id':user,'alias':summary[n]['alias'],
            'weighted_profile_terms':[{'term':str(terms[profile.indices[j]]),'weight':float(profile.data[j])} for j in order],
            'profile_item_metadata':[meta_all[item_lookup[i]] for i in sorted(ids)]})
    save(out/'private/cbf_profile_terms.json',details)
    save(out/'verification_environment.json',{'python':sys.version,'platform':platform.platform(),
        'duckdb':duckdb.__version__,'numpy':np.__version__,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('tvrec').glob('*.py')},
        'note':'Checks only; primary fit/evaluation source hashes are preserved in execution_run.json'})
    print({k:v for k,v in result.items() if k!='files'})
    c.close()


if __name__=='__main__':
    main()
