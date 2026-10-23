"""Post-selection segment diagnostics and real-user examples; no model fitting."""
import csv
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .final_refit import digest,check_evidence
from .improve import connect
from .models import ranked_predictions
from .pipeline import LOW,SPLIT,save
from .rank_features import qualified_occurrences


def read_predictions(path,field='predictions'):
    with Path(path).open(newline='') as f:
        return {r['user_id']:r|{'ids':list(map(int,r[field].split()))} for r in csv.DictReader(f)}


def arithmetic(truth,pred):
    ranks=[n for n,i in enumerate(pred,1) if i in truth]
    denominator=min(5,len(truth))
    exact=sum((Fraction(h,r) for h,r in enumerate(ranks,1)),Fraction(0))/denominator if denominator else Fraction(0)
    return {'hit_ranks':ranks,'fraction':str(exact),'ap5':float(exact),
        'arithmetic':f"({' + '.join(f'{h}/{r}' for h,r in enumerate(ranks,1)) or '0'})/{denominator}" if denominator else 'empty truth -> 0'}


def main():
    out=Path('artifacts/stage4')
    selected=json.loads((out/'selection.json').read_text())
    style=selected['selected']
    if style=='retain_v2':
        style='v2_supervised_rrf_equal'  # diagnostic only, explicitly not selected
    frozen=json.loads((out/'frozen_prior.json').read_text())['files'];check_evidence(frozen)
    losses=pd.read_csv(out/'private/v2_losses.csv',dtype={'user_id':str})
    segment_rows=[];all_pairs=[];ranking=[]
    for window in ['main','earlier']:
        base=read_predictions(f'artifacts/stage3/private/{window}_v1_plus_channel_0.5.csv')
        new=read_predictions(out/'private'/f'{window}_{style}.csv')
        folder=Path('cache/rerank')/window
        fm=json.loads((folder/'features_manifest.json').read_text())
        X=np.load(folder/'X.npy',mmap_mode='r')
        with np.load(folder/'indices.npz') as z:
            cold=z['cold']
        offsets=np.r_[0,np.cumsum(fm['pool_lengths'])]
        airings=np.expm1(X[offsets[:-1],fm['feature_names'].index('user_log_airings')])
        repeat=X[offsets[:-1],fm['feature_names'].index('user_repeat_fraction')]
        corrected=0
        rows=[]
        for n,u in enumerate(fm['users']):
            old_ap=float(base[u]['ap5']);new_ap=float(new[u]['ap5'])
            source=losses[(losses.window==window)&(losses.user_id==u)].iloc[0]
            row={'window':window,'user_id':u,'v2_ap5':old_ap,'v3_ap5':new_ap,'delta':new_ap-old_ap,
                'history_group':source.history_group,
                'repeat_group':'no_qualified_history' if airings[n]==0 else 'repeat_heavy' if repeat[n]>=.75 else 'exploratory_history',
                'channel_group':'cold' if cold[n] else source.channel_group,
                'target_history_share_posthoc_only':source.target_history_share}
            rows.append(row);all_pairs.append(row)
            # Membership gain vs pure within-list ordering gain.
            corrected+=len((set(new[u]['ids'])-set(base[u]['ids']))&set(map(int,base[u]['truth'].split())))
        frame=pd.DataFrame(rows)
        for feature in ['history_group','repeat_group','channel_group']:
            for group,g in frame.groupby(feature):
                segment_rows.append({'window':window,'segmentation':feature,'group':group,'users':len(g),
                    'v2_map5':float(g.v2_ap5.mean()),'v3_map5':float(g.v3_ap5.mean()),
                    'gain':float(g.delta.mean()),'improved':int((g.delta>0).sum()),'regressed':int((g.delta<0).sum())})
        total_truth=sum(len(r['truth'].split()) for r in base.values())
        for name,pred in [('v2',base),('v3',new)]:
            hits=sum(len(set(r['ids'])&set(map(int,r['truth'].split()))) for r in pred.values())
            ranking.append({'window':window,'method':name,'top5_micro_recall':hits/total_truth,'hits':hits,'truth_pairs':total_truth})
    pd.DataFrame(segment_rows).to_csv(out/'v2_v3_segments.csv',index=False)
    pd.DataFrame(all_pairs).to_csv(out/'private/v2_v3_user_deltas.csv',index=False)
    save(out/'reranking_quality.json',ranking)
    c=connect(Path('cache'))
    base=read_predictions('artifacts/stage3/private/main_v1_plus_channel_0.5.csv')
    new=read_predictions(out/'private'/f'main_{style}.csv')
    main=[r for r in all_pairs if r['window']=='main']
    success=max(main,key=lambda r:(r['delta'],r['user_id']))['user_id']
    failure=min(main,key=lambda r:(r['delta'],r['user_id']))['user_id']
    originals=json.loads(Path('artifacts/stage2/private/representative_users.json').read_text())
    example_ids=[(r['user_id'],'same Stage-2 representative') for r in originals]
    example_ids+=[(success,'largest observed heldout correction'),(failure,'largest observed heldout regression')]
    with np.load('cache/improvements/main_v1_scores.npz') as z:
        components={name:z[name] for name in ['popularity','repeat','cf','cbf']}
    fm=json.loads(Path('cache/rerank/main/features_manifest.json').read_text())
    users=fm['users'];items=fm['items'];user_index={u:n for n,u in enumerate(users)}
    fallback=np.lexsort((np.array(items),-components['popularity'][0]))
    with np.load('cache/rerank/main/indices.npz') as z:
        cold={u for n,u in enumerate(users) if z['cold'][n]}
    examples=[]
    for u,reason in example_ids:
        query=c.execute('''SELECT e.session_id,e.user_id,e.sid,e.channel,greatest(e.a,?) a,
            least(e.b,?) b,a.duration,a.item FROM continuous_events e JOIN base.airings a USING(sid)
            WHERE e.user_id=? AND e.a<? AND e.b>? AND a.item>0 AND a.duration>0''',[LOW,SPLIT,u,SPLIT,LOW])
        history=query.fetchdf()
        occ=qualified_occurrences(history,SPLIT)
        hist=occ.groupby('item',as_index=False).agg(frequency=('sid','size'),last_second=('b','max')).sort_values(['frequency','item'],ascending=[False,True])
        predictions={name:ranked_predictions(score[[user_index[u]]],[u],items,fallback,cold)[u]
                     for name,score in components.items()}
        predictions.update(v2=base[u]['ids'],v3=new[u]['ids'])
        truth=set(map(int,base[u]['truth'].split()))
        ids=set(hist.item)|truth|set(i for row in predictions.values() for i in row)
        q=c.execute('''SELECT item,first(title ORDER BY ss,sid) title,first(category ORDER BY ss,sid) category
            FROM base.airings WHERE item IN ('''+','.join(map(str,sorted(ids)))+') GROUP BY item')
        metadata={int(i):{'item':int(i),'title':title,'category':category} for i,title,category in q.fetchall()}
        ap={name:arithmetic(truth,pred) for name,pred in predictions.items()}
        assert abs(ap['v2']['ap5']-float(base[u]['ap5']))<1e-12
        assert abs(ap['v3']['ap5']-float(new[u]['ap5']))<1e-12
        examples.append({'user_id':u,'reason':reason,'training_window':[LOW,SPLIT],
            'historical_continuous_runs':history.to_dict('records'),
            'historical_programmes':[metadata[int(r.item)]|{'frequency':int(r.frequency),'last_second':int(r.last_second)} for r in hist.itertuples()],
            'truth':[metadata[i] for i in sorted(truth)],
            'method_top5':{name:[metadata[i] for i in pred] for name,pred in predictions.items()},
            'exact_ap5':ap,'v3_model_is_early_only':True,
            'posthoc_example_selection_never_used_as_feature':True})
    save(out/'private/user_examples.json',{'executor':'AI assistant, no student verification claimed','examples':examples})
    save(out/'user_example_summary.json',[{'alias':f'User-{n+1}','reason':r['reason'],
        'v2_ap5':r['exact_ap5']['v2']['ap5'],'v3_ap5':r['exact_ap5']['v3']['ap5'],
        'v2_top5':r['method_top5']['v2'],'v3_top5':r['method_top5']['v3']}
        for n,r in enumerate(examples)])
    check_evidence(frozen);c.close()
    print({'segments':segment_rows,'reranking_quality':ranking,
        'examples':[(r['reason'],r['exact_ap5']['v2']['ap5'],r['exact_ap5']['v3']['ap5']) for r in examples]})


if __name__=='__main__':
    main()
