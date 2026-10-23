"""Refit the locally justified channel addition; reuse identical final v1 models.

python -m tvrec.channel_v2 prepare
No network access, Kaggle calls or new ALS/CBF training occurs here.
"""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from .core import write_submission
from .final_refit import digest,check_evidence,assert_history_only
from .improve import connect
from .labels import event_frame
from .models import measured,ranked_predictions,reciprocal_ranks
from .pipeline import LOW,TEST,END,save
from .rank_features import qualified_occurrences,channel_scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare'])
    parser.add_argument('--cache',type=Path,default=Path('cache'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/stage3'))
    args=parser.parse_args()
    out=args.out
    frozen=json.loads((out/'frozen_prior.json').read_text())['files']
    check_evidence(frozen)
    selected=json.loads((out/'robust_selection.json').read_text())
    if selected['selected']=='v1':
        raise RuntimeError('No robust improvement; refuse cosmetic v2')
    weight=selected['channel_weight']
    assert all(selected[w]['mean_delta']>=.001 and selected[w]['bootstrap_95_percentile_interval'][0]>0
               for w in ['main','earlier'])
    path=Path('artifacts/final/submission_hybrid_v2.csv')
    manifest_path=Path('artifacts/final/submission_hybrid_v2.manifest.json')
    if manifest_path.exists():
        assert digest(path)==json.loads(manifest_path.read_text())['submission_sha256']
        print('Existing v2 reused unchanged; no refit or submission.')
        return
    if path.exists():
        raise RuntimeError('Refuse to overwrite an existing candidate')
    v1=json.loads(Path('artifacts/final/manifest.json').read_text())
    assert digest(v1['submission']['path'])==v1['submission']['sha256']=='d4026ac73d6f940010d7be3f8241731bd5975054fc3127eebd0cc1cfaebfed72'
    model_dir=args.cache/'final/hybrid_v1'
    for name in ['component_scores.npz','mappings.json']:
        p=model_dir/name
        assert digest(p)==v1['model_artifacts'][str(p)]
    with Path('sweettv-tv-program-recommender/submission.csv').open(newline='') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['user_id','tv_show_id']
        users=[r['user_id'] for r in reader]
    mappings=json.loads((model_dir/'mappings.json').read_text())
    assert users==mappings['template_users'] and len(users)==1260
    items=mappings['test_candidates']
    cache_out=args.cache/'improvements/final_v2'
    cache_out.mkdir(parents=True,exist_ok=True)
    with measured() as total:
        c=connect(args.cache)
        actual={i for i, in c.execute('SELECT DISTINCT item FROM base.airings WHERE period=31 AND item>0 AND duration>0 AND ss<? AND se>?',[END,TEST]).fetchall()}
        assert actual==set(items)
        with np.load(model_dir/'component_scores.npz') as z:
            baseline=z['hybrid']
            fallback=z['fallback']
        with measured() as feature_resources:
            training=event_frame(c,LOW,TEST)
            occurrences=qualified_occurrences(training,TEST)
            latest=occurrences.groupby(['user_id','item'],as_index=False).agg(last_second=('b','max'))
            leakage=assert_history_only(c,training,latest)
            warm=set(training.loc[training.b-training.a>=60,'user_id'])
            channel=channel_scores(occurrences,users,items)
            occurrences.groupby(['user_id','channel']).size().rename('count').reset_index().to_csv(cache_out/'user_channel_counts.csv',index=False)
            occurrences.groupby(['channel','item']).size().rename('count').reset_index().to_csv(cache_out/'channel_item_counts.csv',index=False)
            occurrences.groupby('channel').size().rename('count').reset_index().to_csv(cache_out/'channel_totals.csv',index=False)
            channel_rank=reciprocal_ranks(channel,fallback,top=200,k=10)
        with measured() as ranking_resources:
            fused=baseline+weight*channel_rank
            predictions=ranked_predictions(fused,users,items,fallback,set(users)-warm)
            write_submission(path,users,predictions,set(items))
        np.savez_compressed(cache_out/'scores.npz',channel=channel,channel_rank=channel_rank,hybrid=fused)
        check_evidence(frozen)
        c.close()
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'executor':'AI assistant, not student verification',
        'submission_path':str(path),'submission_sha256':digest(path),'rows':1260,'candidates':len(items),
        'v1_csv_sha256':v1['submission']['sha256'],'base_strategy':'unchanged frozen v1 RRF',
        'addition':{'component':'training-only channel-conditioned programme popularity','weight':weight,
            'formula':'P(channel|user) based on normalized sqrt qualifying-airing counts, times training P(programme|channel)',
            'rrf_top':200,'rrf_k':10,'input_coverage':.5,'one_count_per_distinct_user_airing':True},
        'main_validation':selected['main'],'earlier_validation':selected['earlier'],
        'final_refit_window':[LOW,TEST],'target_window':[TEST,END],'leakage_checks':leakage,
        'channel_fit_qualified_airings':len(occurrences),'channel_fit_users':int(occurrences.user_id.nunique()),
        'channel_fit_programmes':int(occurrences.item.nunique()),'cold_users':len(set(users)-warm),
        'same_v1_cold_fallback':True,'repeats_allowed':True,
        'existing_final_als_and_cbf_reused':True,'reason':'Same input history, mappings, >=50% definition and unchanged fitted model settings; only the new channel component requires refitting.',
        'new_als_or_cbf_fits':0,'feature_resources':feature_resources,'ranking_resources':ranking_resources,
        'resources':total,'input_hashes':v1['input_files'],
        'model_artifact_hashes':{str(p):digest(p) for p in cache_out.glob('*') if p.is_file()},
        'frozen_prior_manifest':'artifacts/stage3/frozen_prior.json','prior_results_and_v1_unchanged':True,
        'independent_validation':None,'kaggle_requests':0,'submitted':False,
        'selection_uses_kaggle_scores':False,'risks':['Adaptive two-window selection; bootstrap intervals do not correct selection optimism.',
            'Channel popularity can displace niche repeat preferences; per-user regressions are preserved.',
            'Original uninterrupted/device, target tie and timezone assumptions remain unchanged.']}
    save(manifest_path,manifest)
    print(json.dumps({'path':str(path),'sha256':manifest['submission_sha256'],
        'seconds':total['seconds'],'feature_seconds':feature_resources['seconds'],
        'peak_rss_bytes':total['peak_rss_bytes'],'cold_users':manifest['cold_users']}),flush=True)


if __name__=='__main__':
    main()
