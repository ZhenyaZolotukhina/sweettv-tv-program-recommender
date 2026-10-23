"""Stage two: repaired labels, individual ALS/CBF, compact fusion and evidence.

python -m tvrec.advanced audit
python -m tvrec.advanced run
Only uses cache/data.duckdb. Never invokes the old full-data pipeline or refits
on validation views for a primary-window model. Never creates a test submission.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import duckdb
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from .core import ap_at_5, map_at_5
from .labels import cached_sessions, event_frame, target_from_events, training_features
from .models import (ImplicitALS, ContentProfiles, matrix_from_features, measured,
    popularity_scores, score_features, ranked_predictions, normalized_scores, reciprocal_ranks)
from .pipeline import LOW, SPLIT, TEST, candidate_set, features, rank_baseline, records, save


def connect(cache):
    if not (cache/'data.duckdb').exists():
        raise RuntimeError('Original cached DB required; this stage never rebuilds it')
    c=duckdb.connect(str(cache/'stage2.duckdb'))
    c.execute("SET threads=4; SET memory_limit='2GB'")
    c.execute(f"ATTACH '{str((cache/'data.duckdb').resolve()).replace(chr(39),chr(39)*2)}' AS base (READ_ONLY)")
    # Compatibility views for reproducing the three old selected baselines.
    for table in ['interactions','airings','template']:
        c.execute(f'CREATE OR REPLACE VIEW {table} AS SELECT * FROM base.{table}')
    return c


def truth_statistics(truth,pairs,users,candidates):
    counts=np.array([len(truth.get(u,set())) for u in users])
    u_set=set(users)
    p=pairs[pairs.user_id.isin(u_set)]
    all_items=set(int(i) for i in p.item)
    return {'users_in_denominator':len(users),'zero_positive_users':int((counts==0).sum()),
        'users_with_five':int((counts==5).sum()),'final_truth_pairs':int(counts.sum()),
        'qualifying_distinct_user_programme_pairs_before_top5':len(p),
        'truth_count_quantiles':np.quantile(counts,[0,.25,.5,.75,1]).tolist(),
        'candidate_count':len(candidates),'qualifying_items':len(all_items),
        'candidate_item_coverage':len(all_items&set(candidates))/len(all_items) if all_items else None,
        'truth_pairs_outside_candidates':sum(len(set(truth.get(u,()))-set(candidates)) for u in users)}


def audit_labels(c,cache,out,legacy_out):
    out.mkdir(parents=True,exist_ok=True)
    private=out/'private'
    private.mkdir(exist_ok=True)
    cached_sessions(c,cache,out)
    users=[r[0] for r in c.execute('SELECT user_id FROM base.template ORDER BY source_row').fetchall()]
    candidates=candidate_set(c,21)
    val=event_frame(c,SPLIT,TEST)
    truth,pairs,selected=target_from_events(val)
    strict,strict_pairs,_=target_from_events(val,strict=True)
    raw=c.execute('''SELECT m.vid session_id,v.user_id,m.sid,v.channel,
        m.first_second a,m.last_second b,a.duration,a.item,v.ve-v.vs session_seconds
        FROM base.matches m JOIN base.views v USING(vid) JOIN base.airings a USING(sid)
        WHERE m.phase='validation' AND a.item>0 AND a.duration>0 AND m.seconds*5>=a.duration*4''').fetchdf()
    raw_truth,raw_pairs,_=target_from_events(raw)
    inclusive_all,_,_=target_from_events(val,top5=False)
    without_long,long_pairs,_=target_from_events(val,long_limit=86400)
    alternate_tie,tie_pairs,_=target_from_events(val,tie='recency')
    cumulative=val.copy()
    cumulative['seconds']=cumulative.b-cumulative.a
    cumulative=cumulative.groupby(['user_id','sid','item','duration'],as_index=False).agg(
        b=('seconds','sum'),session_seconds=('session_seconds','max'))
    cumulative['a']=0
    cumulative_truth,_,_=target_from_events(cumulative)
    legacy_truth=defaultdict(set)
    for u,i in c.execute("SELECT DISTINCT user_id,item FROM base.interactions WHERE phase='validation' AND seconds>=60").fetchall():
        if i in candidates:
            legacy_truth[u].add(int(i))
    old_eligible=[u for u in users if legacy_truth[u]]
    policies={'official_ge80_top5':truth,'strict_gt80_top5':strict,
              'raw_single_record_ge80_top5':raw_truth,'ge80_all_items_no_top5':inclusive_all,
              'exclude_sessions_over_24h_ge80_top5':without_long,'frequency_ties_recency_ge80_top5':alternate_tie,
              'cumulative_switching_allowed_sensitivity_only':cumulative_truth}
    policies['legacy_ge60_all_items']=dict(legacy_truth)
    pd.DataFrame([{'user_id':u,'legacy_count':len(legacy_truth.get(u,())),
        'qualifying_ge80_pairs':int((pairs.user_id==u).sum()),
        **{p:len(t.get(u,())) for p,t in policies.items()}} for u in users]).to_csv(private/'truth_counts_per_user.csv',index=False)
    save(private/'truth_versions.json',{p:{u:sorted(t.get(u,())) for u in users} for p,t in policies.items()})
    selected[selected.user_id.isin(users)].to_csv(private/'official_target_frequencies.csv',index=False)
    old=json.loads((legacy_out/'validation_results.json').read_text())
    oldref=old['best_by_method']['popularity']
    ref_frame=features(c,oldref['variant'],oldref['threshold'],oldref['half_life_days'])
    totals=ref_frame[ref_frame.item.isin(candidates)].groupby('item').score.sum().to_dict()
    fallback=sorted(candidates,key=lambda i:(-totals.get(i,0),i))
    comparison=[]
    for method,config in old['best_by_method'].items():
        f=features(c,config['variant'],config['threshold'],config['half_life_days'])
        pred=rank_baseline(f,users,candidates,method,config['alpha'],config['min_history'],config['signal'],fallback)
        reproduced=map_at_5(legacy_truth,pred,old_eligible)
        assert abs(reproduced-config['map5'])<1e-12,(method,reproduced,config['map5'])
        row={'method':method,'original_published_local_map5':config['map5'],
             'old_reproduced_same_1246_users':reproduced,
             'old_on_all_1260_users':map_at_5(legacy_truth,pred,users),
             'corrected_on_same_1246_users':map_at_5(truth,pred,old_eligible)}
        row.update({p:map_at_5(t,pred,users) for p,t in policies.items()})
        comparison.append(row)
        save(private/f'legacy_predictions_{method}.json',pred)
    pd.DataFrame(comparison).to_csv(out/'old_vs_corrected.csv',index=False)
    audit={'source_evidence':'artifacts/sources/relevance_sources.json',
        'primary':'Official >=80% coverage, one continuous unambiguous channel session, then five most frequent distinct positive IDs per user',
        'strict_gt80_implemented_as_sensitivity':True,
        'frequency_unit':'one qualifying distinct airing (sid) per user; never sum visits to reach coverage',
        'session_assumptions':'Deduplicate identical channel/start/stop intervals; union touching/overlapping same-channel intervals only with no conflicting channel active. Drop concurrent different-channel time as ambiguous. Gaps and channel changes split sessions. Schedule conflicts retain stage-one disjoint rule.',
        'tie_rule':'frequency descending, programme ID ascending; recency tie sensitivity saved',
        'denominator':'All 1260 original template users, fixed for all models including empty-truth AP=0; nonempty and >=5-only alternatives reported',
        'model_inputs':'coverage/seconds thresholds and decay are independently selected from TRAIN ONLY interactions, not constrained to target 80%',
        'official':truth_statistics(truth,pairs,users,candidates),
        'strict':truth_statistics(strict,strict_pairs,users,candidates),
        'raw_single_record':truth_statistics(raw_truth,raw_pairs,users,candidates),
        'old_eligible_users':len(old_eligible),'old_truth_pairs':sum(len(legacy_truth[u]) for u in users),
        'old_median_truth_items':float(np.median([len(legacy_truth[u]) for u in old_eligible])),
        'inclusive_vs_strict_changed_users':sum(truth.get(u,set())!=strict.get(u,set()) for u in users),
        'session_policy_changed_users':sum(truth.get(u,set())!=raw_truth.get(u,set()) for u in users),
        'frequency_tie_policy_changed_users':sum(truth.get(u,set())!=alternate_tie.get(u,set()) for u in users),
        'long_session_policy_changed_users':sum(truth.get(u,set())!=without_long.get(u,set()) for u in users),
        'cumulative_switching_changed_users':sum(truth.get(u,set())!=cumulative_truth.get(u,set()) for u in users),
        'legacy_comparison':comparison,'limitations':[
            'Official page does not explain uninterrupted visits/concurrent devices or frequency ties; conservative continuity assumption is participant-informed.',
            'True Kaggle users have strictly five positives; historical local users can have fewer. Fixed-all-user and full-five-user metrics are both saved.',
            'No timezone column; original wall-clock convention retained. A clock-change hour can affect coverage.',
            'Top-5 truth ordering is used to choose membership; AP@5 uses binary membership, not a graded target-rank metric.'
        ]}
    save(out/'label_audit.json',audit)
    # Exact >80% witness, with source rows and original schedule row reference.
    witness=val[(5*(val.b-val.a)>4*val.duration)&(val.b-val.a<val.duration)&val.user_id.isin(users)].sort_values(['a','sid']).iloc[0]
    w={k:(int(v) if isinstance(v,(np.integer,int)) else v) for k,v in witness.to_dict().items()}
    w['overlap_seconds']=w['b']-w['a']
    w['fraction']=w['overlap_seconds']/w['duration']
    w['strict_arithmetic']=f"5 * {w['overlap_seconds']} > 4 * {w['duration']}"
    w['schedule']=records(c,f"SELECT sid,period,source_row,channel,ss,se,duration,item,title,category FROM base.airings WHERE sid={w['sid']}")[0]
    q=c.execute('SELECT vid,channel,vs,ve,duraton FROM base.views WHERE user_id=? AND vs<? AND ve>? ORDER BY vs,vid',
                [w['user_id'],int(w['b']),int(w['a'])])
    w['overlapping_original_view_rows']=[dict(zip([x[0] for x in q.description],r)) for r in q.fetchall()]
    for key in ['a','b']:
        w[key+'_timestamp']=str(np.datetime64(w[key],'s'))
    for key in ['ss','se']:
        w['schedule'][key+'_timestamp']=str(np.datetime64(w['schedule'][key],'s'))
    w['csv_line_rule']='source row ID + 1 (header is line 1); view vid refers to dataset11-30.csv'
    save(private/'coverage_witness.json',w)
    print(json.dumps({'official':audit['official'],'strict':audit['strict'],'old_vs_corrected':comparison}),flush=True)
    return users,sorted(candidates),policies


def metadata(c,cutoff,upper,items):
    columns=['tv_show_title','tv_show_category','tv_show_genre_1','tv_show_genre_2',
             'tv_show_genre_3','year_of_production','director','actors']
    text=','.join('s.'+x for x in columns)
    frame=c.execute(f'''SELECT a.item,a.ss,{text} FROM base.airings a JOIN base.schedule s USING(sid)
        WHERE a.item>0 AND a.duration>0 AND a.ss<? AND a.se>?
        QUALIFY row_number() OVER(PARTITION BY a.item ORDER BY a.ss,a.sid)=1 ORDER BY a.item''',[upper,LOW]).fetchdf()
    lookup={int(r['item']):r for r in frame.replace({np.nan:None}).to_dict('records')}
    training=[r for r in lookup.values() if r['ss']<cutoff]
    return training,[lookup.get(int(i),{}) for i in items]


def metrics(truth,pred,users,warm):
    cold=[u for u in users if u not in warm]
    nonempty=[u for u in users if truth.get(u)]
    five=[u for u in users if len(truth.get(u,()))==5]
    return {'map5':map_at_5(truth,pred,users),
        'warm_map5':map_at_5(truth,pred,[u for u in users if u in warm]),
        'cold_map5':map_at_5(truth,pred,cold),'warm_users':sum(u in warm for u in users),'cold_users':len(cold),
        'nonempty_map5':map_at_5(truth,pred,nonempty),'nonempty_users':len(nonempty),
        'full_five_map5':map_at_5(truth,pred,five),'full_five_users':len(five),'denominator':len(users)}


def save_predictions(path,pred,truth,users,warm):
    rows=[]
    for user in users:
        hits=[i for i,item in enumerate(pred[user],1) if item in truth.get(user,set())]
        rows.append({'user_id':user,'predictions':' '.join(map(str,pred[user])),
            'truth':' '.join(map(str,sorted(truth.get(user,())))),
            'hit_ranks':' '.join(map(str,hits)),'ap5':ap_at_5(truth.get(user,()),pred[user]),
            'group':'warm' if user in warm else 'cold'})
    pd.DataFrame(rows).to_csv(path,index=False)


def model_suite(c,cache,out,users,items,truth,lower=LOW,cutoff=SPLIT,upper=TEST,fixed=None,prefix='main'):
    """Small fixed grid; optional fixed configs for untuned earlier robustness."""
    training=event_frame(c,lower,cutoff)
    warm=set(training.loc[(training.b-training.a)>=60,'user_id'])
    cold=set(users)-warm
    all_users=sorted(set(training.user_id)|set(users))
    all_items=sorted(set(int(i) for i in training.item)|set(items))
    user_indices=np.array([all_users.index(u) for u in users])
    item_lookup={i:n for n,i in enumerate(all_items)}
    item_indices=np.array([item_lookup[i] for i in items])
    private=out/'private'
    board=[]
    best={}
    scores_best={}
    config_best={}
    predictions_best={}
    fallback=np.arange(len(items))
    def record(family,config,scores,timing,name):
        pred=ranked_predictions(scores,users,items,fallback,cold)
        result={'name':name,'family':family,'config':config,**metrics(truth,pred,users,warm),**timing}
        board.append(result)
        save(out/f'{prefix}_individual_results.json',board)
        save_predictions(private/f'{prefix}_{name}.csv',pred,truth,users,warm)
        if family not in best or result['map5']>best[family]['map5']:
            best[family]=result
            scores_best[family]=scores
            config_best[family]=config
            predictions_best[family]=pred
        print(f"{prefix} {name}: {result['map5']:.6f}, {timing['seconds']:.2f}s",flush=True)
    # Three popularity alternatives; no re-running the old 240-setting sweep.
    pops=([fixed['popularity']] if fixed else [
        {'minimum':.5,'strength':'frequency','half_life':0},
        {'minimum':.5,'strength':'frequency','half_life':35},
        {'seconds_cut':60,'strength':'seconds','half_life':14}])
    for n,config in enumerate(pops):
        with measured() as timing:
            f=training_features(training,cutoff=cutoff,**config)
            scores=popularity_scores(f,users,items)
        # A scores all users identically, including cold users.
        candidate_fallback=np.lexsort((np.array(items),-scores[0]))
        old_fallback=fallback
        fallback=candidate_fallback
        record('popularity',config,scores,timing,f'popularity_{n}')
        fallback=old_fallback
    fallback=np.lexsort((np.array(items),-scores_best['popularity'][0]))
    # Recreate A predictions under the fixed final fallback, cold included.
    predictions_best['popularity']=ranked_predictions(scores_best['popularity'],users,items,fallback)
    repeats=([fixed['repeat']] if fixed else [
        {'minimum':m,'strength':s,'half_life':h}
        for m,s,h in [(.05,'frequency',0),(.5,'frequency',0),(.5,'frequency',35),
                      (.8,'frequency',0),(.5,'fraction',0),(.5,'fraction',35)]])
    for n,config in enumerate(repeats):
        with measured() as timing:
            f=training_features(training,cutoff=cutoff,**config)
            scores=score_features(f,users,items)
        record('repeat',config,scores,timing,f'repeat_{n}')
    # Simple baseline C, normalized score blend, evaluated independently of ALS/CBF.
    simple=(fixed.get('repeat_popularity',{'weight':.1}),) if fixed else ({'weight':.1},{'weight':.5})
    for n,config in enumerate(simple):
        with measured() as timing:
            scores=normalized_scores(scores_best['repeat'])+config['weight']*normalized_scores(scores_best['popularity'])
        record('repeat_popularity',config,scores,timing,f'repeat_popularity_{n}')
    als_configs=([fixed['cf']] if fixed else [
        {'minimum':.5,'factors':32,'alpha':10.,'regularization':1.,'iterations':12,'seed':42},
        {'minimum':.5,'factors':64,'alpha':20.,'regularization':1.,'iterations':12,'seed':42},
        {'minimum':.05,'factors':32,'alpha':10.,'regularization':1.,'iterations':12,'seed':42}])
    for n,config in enumerate(als_configs):
        key=hashlib.sha256(json.dumps({'config':config,'window':[lower,cutoff,upper],
            'model_source':hashlib.sha256(Path(__file__).with_name('models.py').read_bytes()).hexdigest(),
            'sessions':json.loads((cache/'stage2_sessions_manifest.json').read_text())},sort_keys=True).encode()).hexdigest()[:20]
        cached=cache/f'als_{key}.npz'
        with measured() as timing:
            if cached.exists():
                data=np.load(cached)
                scores=data['scores']
                timing['model_cache_hit']=True
            else:
                f=training_features(training,minimum=config['minimum'],strength='frequency',cutoff=cutoff)
                matrix=matrix_from_features(f,all_users,all_items)
                model=ImplicitALS(**{k:v for k,v in config.items() if k!='minimum'}).fit(matrix)
                with threadpool_limits(limits=1):
                    scores=(model.user_factors[user_indices]@model.item_factors[item_indices].T).astype(np.float32)
                scores=np.maximum(scores,0)
                np.savez_compressed(cached,scores=scores,user_factors=model.user_factors,item_factors=model.item_factors)
                timing['model_cache_hit']=False
                timing['matrix_shape']=list(matrix.shape)
                timing['matrix_nonzero']=matrix.nnz
        record('cf',config,scores,timing,f'cf_{n}')
    meta_train,meta_all=metadata(c,cutoff,upper,all_items)
    cbf_configs=([fixed['cbf']] if fixed else [
        {'minimum':.5,'rich':True,'half_life':0},
        {'minimum':.5,'rich':False,'half_life':0},
        {'minimum':.05,'rich':True,'half_life':35}])
    for n,config in enumerate(cbf_configs):
        with measured() as timing:
            f=training_features(training,minimum=config['minimum'],strength='frequency',half_life=config['half_life'],cutoff=cutoff)
            weights=matrix_from_features(f,users,all_items)
            model=ContentProfiles(rich=config['rich']).fit_items(meta_train,meta_all)
            with threadpool_limits(limits=1):
                scores,profiles=model.score(weights,item_indices)
            timing['vocabulary_size']=len(model.vectorizer.vocabulary_)
            timing['training_metadata_items']=len(meta_train)
            observed_items=set(f.item)
            timing['candidate_items_without_training_interaction']=sum(i not in observed_items for i in items)
        record('cbf',config,scores,timing,f'cbf_{n}')
    save(out/f'{prefix}_individual_results.json',board)
    save(out/f'{prefix}_best_individual_configs.json',config_best)
    return best,scores_best,config_best,predictions_best,fallback,warm,board


def fusion_grid(scores,configs,fallback,users,items,truth,warm,private,prefix='main',fixed=None):
    """All components were evaluated alone first. No raw-score scale mixing."""
    rows=[]
    norms={k:normalized_scores(v) for k,v in scores.items() if k in ['repeat','cf','cbf','popularity']}
    ranks=None
    if fixed:
        settings=[fixed]
    else:
        settings=[]
        for cf,cbf,pop in [(0,0,.1),(.05,0,.1),(.15,0,.1),(.3,0,.1),
                           (0,.05,.1),(0,.15,.1),(.05,.05,.1),(.15,.05,.1),
                           (.05,.15,.1),(.15,.15,.1),(.3,.15,.1),(.15,.05,0)]:
            settings.append({'type':'max_normalized','weights':{'repeat':1.,'cf':cf,'cbf':cbf,'popularity':pop}})
        for k in [0,10]:
            for cf,cbf in [(.1,0),(.3,0),(0,.1),(.1,.1)]:
                settings.append({'type':'rrf','k':k,'top':200,
                    'weights':{'repeat':1.,'cf':cf,'cbf':cbf,'popularity':.1}})
        for threshold in [3,10,20]:
            for other in ['cf','cbf']:
                settings.append({'type':'sparse_switch','threshold':threshold,'other':other,'popularity_weight':.1})
    best=None
    predictions=None
    best_score=None
    cold=set(users)-warm
    for n,config in enumerate(settings):
        with measured() as timing:
            if config['type']=='max_normalized':
                fused=sum(weight*norms[name] for name,weight in config['weights'].items() if weight)
            elif config['type']=='rrf':
                if ranks is None:
                    ranks={}
                key=config['k']
                if key not in ranks:
                    ranks[key]={name:reciprocal_ranks(scores[name],fallback,top=config['top'],k=key)
                                for name in ['repeat','cf','cbf','popularity']}
                fused=sum(weight*ranks[key][name] for name,weight in config['weights'].items() if weight)
            else:
                count=(scores['repeat']>0).sum(axis=1)
                use=count<config['threshold']
                fused=norms['repeat'].copy()
                fused[use]=norms[config['other']][use]
                fused+=config['popularity_weight']*norms['popularity']
            pred=ranked_predictions(fused,users,items,fallback,cold)
        row={'name':f'{prefix}_fusion_{n}','family':'hybrid','config':config,
             **metrics(truth,pred,users,warm),**timing}
        rows.append(row)
        # Preserve every configuration's predictions and per-user contributions.
        save_predictions(private/f'{prefix}_fusion_{n}.csv',pred,truth,users,warm)
        if best is None or row['map5']>best['map5']:
            best=row
            predictions=pred
            best_score=fused
        print(f"{row['name']}: {row['map5']:.6f}",flush=True)
    return best,predictions,rows,best_score


def evidence(c,out,legacy_out,users,items,truths,predictions,configs,warm,main_scores,fallback):
    private=out/'private'
    earlier_examples=json.loads((legacy_out/'private/user_examples.json').read_text())
    selected_users=[x['user_id'] for x in earlier_examples]
    all_preds={k:v for k,v in predictions.items()}
    examples=[]
    training=event_frame(c,LOW,SPLIT)
    validation=event_frame(c,SPLIT,TEST)
    for old in earlier_examples:
        user=old['user_id']
        lists={k:pred[user] for k,pred in all_preds.items()}
        ids=sorted(set(i for p in lists.values() for i in p))
        title_rows=records(c,'SELECT item,first(title ORDER BY ss,sid) title,first(category ORDER BY ss,sid) category FROM base.airings WHERE item IN ('+','.join(map(str,ids))+') GROUP BY item')
        train_user=training[training.user_id==user].copy()
        val_user=validation[validation.user_id==user].copy()
        train_user['overlap_seconds']=train_user.b-train_user.a
        train_user['fraction']=train_user.overlap_seconds/train_user.duration
        val_user['overlap_seconds']=val_user.b-val_user.a
        val_user['fraction']=val_user.overlap_seconds/val_user.duration
        profile=training_features(train_user,cutoff=SPLIT,minimum=configs['cbf']['minimum'],
                                  half_life=configs['cbf']['half_life'],strength='frequency')
        examples.append({'user_id':user,'original_selection_reason':old['selection_reason'],
            'same_user_as_stage_one':True,'warm':user in warm,
            'training_continuous_runs':train_user.to_dict('records'),
            'validation_continuous_runs':val_user.to_dict('records'),
            'cbf_profile_inputs':profile.to_dict('records'),
            'predictions':lists,'metadata':title_rows,
            'truth':sorted(truths['official_ge80_top5'].get(user,())),
            'strict_truth':sorted(truths['strict_gt80_top5'].get(user,())),
            'ap5':{m:ap_at_5(truths['official_ge80_top5'].get(user,()),p) for m,p in lists.items()}})
    save(private/'representative_users.json',examples)
    # One actual hand-computable result, not just an illustrative synthetic AP.
    method='hybrid' if 'hybrid' in all_preds else 'repeat'
    eligible=[u for u in users if truths['official_ge80_top5'].get(u)]
    manual_user=next((u for u in selected_users if 0<ap_at_5(truths['official_ge80_top5'].get(u,()),all_preds[method][u])<1),None)
    if manual_user is None:
        manual_user=next(u for u in eligible if 0<ap_at_5(truths['official_ge80_top5'][u],all_preds[method][u])<1)
    truth=truths['official_ge80_top5'][manual_user]
    pred=all_preds[method][manual_user]
    ranks=[i for i,item in enumerate(pred,1) if item in truth]
    arithmetic=' + '.join(f'{hit}/{rank}' for hit,rank in enumerate(ranks,1))
    save(private/'actual_manual_ap5.json',{'user_id':manual_user,'method':method,'truth':sorted(truth),
        'predictions':pred,'hit_ranks':ranks,'precision_terms':arithmetic,
        'calculation':f'({arithmetic}) / {min(5,len(truth))}',
        'ap5':ap_at_5(truth,pred)})
    sensitivity=[]
    for method,pred in all_preds.items():
        for policy,truth in truths.items():
            sensitivity.append({'method':method,'label_policy':policy,**metrics(truth,pred,users,warm)})
    pd.DataFrame(sensitivity).to_csv(out/'method_label_sensitivity.csv',index=False)


def full_experiment(c,cache,out,legacy_out,users,items,truths):
    best,scores,configs,predictions,fallback,warm,board=model_suite(c,cache,out,users,items,truths['official_ge80_top5'])
    hybrid,hybrid_pred,fusions,_=fusion_grid(scores,configs,fallback,users,items,truths['official_ge80_top5'],warm,out/'private')
    save(out/'fusion_results.json',fusions)
    predictions['hybrid']=hybrid_pred
    individual_winner=max(best.values(),key=lambda r:r['map5'])
    winner=hybrid if hybrid['map5']>individual_winner['map5'] else individual_winner
    compact=list(best.values())+[hybrid]
    save(out/'leaderboard.json',sorted(compact,key=lambda r:-r['map5']))
    pd.DataFrame([{k:v for k,v in r.items() if k!='config'}|{'config':json.dumps(r['config'],sort_keys=True)} for r in compact]).sort_values('map5',ascending=False).to_csv(out/'leaderboard.csv',index=False)
    # Identical per-user denominator/candidates and paired gain, with seed-fixed CI.
    delta=np.array([ap_at_5(truths['official_ge80_top5'].get(u,()),hybrid_pred[u])-
                    ap_at_5(truths['official_ge80_top5'].get(u,()),predictions['repeat'][u]) for u in users])
    rng=np.random.default_rng(42)
    bootstrap=np.array([rng.choice(delta,size=len(delta),replace=True).mean() for _ in range(1000)])
    gain={'same_users':len(users),'same_candidates':len(items),'versus':'best repeat alone',
        'hybrid_minus_repeat_mean_ap5':float(delta.mean()),'improved_users':int((delta>0).sum()),
        'worsened_users':int((delta<0).sum()),'bootstrap_95_percentile_interval':np.quantile(bootstrap,[.025,.975]).tolist(),
        'warning':'Selection on same holdout; interval does not correct model-selection optimism.'}
    save(out/'paired_gain.json',gain)
    evidence(c,out,legacy_out,users,items,truths,predictions,configs,warm,scores,fallback)
    # Freeze the main-window choices first, then assess one earlier untuned split.
    earlier_cut=int(np.datetime64('2020-04-13','s').astype('int64'))
    earlier_val=event_frame(c,earlier_cut,SPLIT)
    earlier_truth,earlier_pairs,_=target_from_events(earlier_val)
    earlier_items=sorted(r[0] for r in c.execute('SELECT DISTINCT item FROM base.airings WHERE item>0 AND duration>0 AND ss<? AND se>?',[SPLIT,earlier_cut]).fetchall())
    eb,es,ec,ep,ef,ew,earlier_board=model_suite(c,cache,out,users,earlier_items,earlier_truth,
        lower=LOW,cutoff=earlier_cut,upper=SPLIT,fixed=configs,prefix='earlier')
    eh,ehp,_,_=fusion_grid(es,ec,ef,users,earlier_items,earlier_truth,ew,out/'private',prefix='earlier',fixed=hybrid['config'])
    save(out/'earlier_robustness.json',{'train':'[2020-03-09,2020-04-13)',
        'validation':'[2020-04-13,2020-05-18)','configurations_frozen_from_main':True,
        'no_earlier_window_tuning':True,'labels':truth_statistics(earlier_truth,earlier_pairs,users,earlier_items),
        'individual':eb,'main_selected_hybrid':eh,
        'interpretation':'Earlier labels precede the main holdout, but configurations were selected on the later main holdout; this is a retrospective stability check, not a fresh untouched test.'})
    final={'stage':'configuration_only; do not refit or submit yet',
        'selected_family':winner['family'],'selected_config':winner['config'],
        'primary_map5':winner['map5'],'individual_configs':configs,
        'hybrid_config':hybrid['config'],'hybrid_map5':hybrid['map5'],
        'target':'official >=80% uninterrupted run; distinct airing frequency; Top-5; frequency ties item-ID ascending',
        'strict_80_sensitivity_saved':True,'main_train':[LOW,SPLIT],'main_validation':[SPLIT,TEST],
        'next_final_refit_window':[LOW,TEST],'candidate_policy':'positive IDs airing during target calendar window; repeats allowed',
        'common_fallback':'selected popularity scores using selected input settings; never infer cold preferences from validation views',
        'normalize':'positive score divided by user maximum for max_normalized; RRF uses only ranks if selected',
        'seed':42,'threads':1,'two_required_methods_implemented':['implicit ALS','metadata TF-IDF user-profile CBF'],
        'source_evidence':'artifacts/sources/relevance_sources.json','limitations':['session/device and frequency-tie semantics unresolved in official description','same-holdout selection optimism'],
        'kaggle_submission_made':False,'final_refit_performed_this_stage':False}
    save(out/'selected_strategy.json',final)
    print('Selected:',json.dumps(final),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','run'])
    parser.add_argument('--cache',type=Path,default=Path('cache'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/stage2'))
    parser.add_argument('--legacy-out',type=Path,default=Path('artifacts'))
    args=parser.parse_args()
    started=time.perf_counter()
    c=connect(args.cache)
    users,items,truths=audit_labels(c,args.cache,args.out,args.legacy_out)
    if args.stage=='run':
        full_experiment(c,args.cache,args.out,args.legacy_out,users,items,truths)
    save(args.out/f'execution_{args.stage}.json',{'seconds':time.perf_counter()-started,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'platform':platform.platform(),
        'scipy':scipy.__version__,'sklearn':sklearn.__version__,'seed':42,'executor':'AI assistant',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('tvrec').glob('*.py')}})
    c.close()


if __name__=='__main__':
    main()
