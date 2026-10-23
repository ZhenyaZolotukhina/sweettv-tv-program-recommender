"""Independent checks of the three V6 experiments and the retain-V5 decision."""
import csv
import json
from pathlib import Path

import duckdb
import numpy as np

from .core import ap_at_5
from .final_refit import digest,check_evidence
from .group_rank import lightgbm_module
from .pipeline import save
from .verify_final import validate_file_records


def main():
    out=Path('artifacts/stage7');frozen=json.loads((out/'frozen_prior.json').read_text())['files'];check_evidence(frozen)
    decision=json.loads((out/'decision.json').read_text());assert decision['experiments']==3
    assert decision['selected']=='retain_v5' and not decision['eligible']
    assert not Path('artifacts/final/submission_hybrid_v6.csv').exists()
    with Path('sweettv-tv-program-recommender/submission.csv').open(newline='') as f:
        reader=csv.reader(f);assert next(reader)==['user_id','tv_show_id'];users=[r[0] for r in reader]
    c=duckdb.connect('cache/data.duckdb',read_only=True);checked=0
    for window,lower,upper in [('main',1589760000,1595808000),('earlier',1586736000,1589760000)]:
        with Path(f'artifacts/stage6/private/{window}_lambda_top5.csv').open(newline='') as f:
            original={r['user_id']:r for r in csv.DictReader(f)}
        sql='SELECT DISTINCT item FROM airings WHERE item>0 AND duration>0 AND ss<? AND se>?'
        if window=='main':sql+=' AND period=21'
        candidates={i for i, in c.execute(sql,[upper,lower]).fetchall()}
        base=Path('cache/rerank')/window;extra=Path('cache/habit_v6')/window
        meta=json.loads((base/'features_manifest.json').read_text());info=json.loads((extra/'extras.json').read_text())
        assert meta['users']==users and set(meta['items'])==candidates
        assert info['cutoff']==meta['signature']['cutoff']==lower and info['last_historical_second']<=lower
        assert not info['future_views_used'] and info['timing_exposure_clipped_to_windows']
        with np.load(base/'indices.npz') as z:idx=z['item_index'];cold=z['cold'];fallback=z['fallback']
        starts=np.r_[0,np.cumsum(meta['pool_lengths'])]
        for block in ['active_stability','channel_calendar','matched_drift']:
            result=next(r for r in decision['results'] if r['experiment']==block and r['window']==window)
            assert all(f['target_end']<=1586736000 for f in result['training_folds'])
            model=lightgbm_module().Booster(model_file=f'cache/habit_v6/{block}_early.txt')
            assert model.params['objective']=='lambdarank' and model.current_iteration()==80
            # cache/rerank holds the original 53 columns; V5 also includes the
            # six frozen channel-loyalty columns. Keep those controls intact.
            X=np.column_stack([np.load(base/'X.npy'),
                np.load(Path('cache/focused_v4')/window/'loyalty.npy'),np.load(extra/(block+'.npy'))])
            assert X.shape[1]==59+len(info['columns'][block])==model.num_feature()
            scores=model.predict(X)
            np.testing.assert_array_equal(scores,np.load(extra/(block+'_early_scores.npy')))
            with (out/'private'/f'{window}_{block}.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            validate_file_records(users,['user_id','tv_show_id'],[[r['user_id'],r['predictions']] for r in rows],candidates)
            aps=[]
            for n,(u,r) in enumerate(zip(users,rows)):
                old=original[u];truth=set(map(int,r['truth'].split()))
                assert truth==set(map(int,old['truth'].split())) and r['group']==old['group']
                lo,hi=starts[n:n+2];pos=np.arange(hi-lo)
                order=np.lexsort((pos,-scores[lo:hi]))
                chosen=idx[lo:hi][order[:5]] if not cold[n] else fallback[:5]
                pred=[meta['items'][i] for i in chosen]
                assert pred==list(map(int,r['predictions'].split()))
                value=ap_at_5(truth,pred);assert abs(value-float(r['ap5']))<1e-12
                assert [rank for rank,item in enumerate(pred,1) if item in truth]==list(map(int,r['hit_ranks'].split()))
                if cold[n]:assert r['predictions']==old['predictions']
                aps.append(value)
            assert abs(np.mean(aps)-result['map5'])<1e-12
            checked+=1
    regression=json.loads((out/'instructor_ap_regressions.json').read_text())
    assert all(r['passed'] for r in regression['cases'])
    check_evidence(frozen);c.close()
    result={'valid':True,'candidate_method_files_checked':checked,'all_users_each':len(users),
        'same_frozen_truth_pools_and_warm_cold_groups':True,'all_scores_and_top5_reconstructed':True,
        'no_holdout_targets_in_training':True,'exactly_three_experiments':True,
        'v1_v5_and_prior_evidence_unchanged':True,'no_v6_csv_created':True,
        'instructor_ap_cases_passed':True,'retained_v5_sha256':digest('artifacts/final/submission_hybrid_v5.csv'),
        'kaggle_requests':0}
    save(out/'independent_validation.json',result)
    print(result)


if __name__=='__main__':main()
