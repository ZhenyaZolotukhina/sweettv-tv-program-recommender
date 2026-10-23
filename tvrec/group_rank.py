"""Validated user/task groups and a documented LightGBM LambdaRank model."""
from pathlib import Path
import sys

import numpy as np


PARAMETERS={
    'objective':'lambdarank','metric':'None','label_gain':[0,1],
    'lambdarank_truncation_level':8,'lambdarank_norm':True,
    'learning_rate':.06,'num_leaves':15,'min_data_in_leaf':50,
    'max_bin':63,'lambda_l1':0.,'lambda_l2':1.,
    'num_threads':2,'deterministic':True,'force_col_wise':True,
    'seed':42,'data_random_seed':42,'feature_fraction_seed':42,'bagging_seed':42,
    'feature_fraction':1.,'bagging_fraction':1.,'bagging_freq':0,'verbosity':-1}
ROUNDS=80


def lightgbm_module():
    # Installed in a gitignored target to leave the frozen environment/configs
    # untouched. A standard requirements-ltr installation also works.
    local=Path('cache/ltr_dependencies')
    if local.is_dir() and str(local.resolve()) not in sys.path:
        sys.path.insert(0,str(local.resolve()))
    import lightgbm
    if lightgbm.__version__!='4.6.0':
        raise RuntimeError('Reproduction requires lightgbm==4.6.0')
    return lightgbm


def labels_for_pool(meta,indices,truth):
    if any(len(v)>5 for v in truth.values()):
        raise ValueError('Ranking relevance must be frequency-selected Top5, not all completed programmes')
    return np.array([int(meta['items'][i] in truth[meta['users'][u]])
        for u,i in zip(indices['user_index'],indices['item_index'])],dtype=np.int32)


def validate_fold_groups(user_index,item_index,group_sizes,labels):
    sizes=np.asarray(group_sizes)
    users=np.asarray(user_index);items=np.asarray(item_index);y=np.asarray(labels)
    if sizes.ndim!=1 or not np.issubdtype(sizes.dtype,np.integer) or np.any(sizes<=0):
        raise ValueError('Positive integer group sizes required')
    if sizes.sum()!=len(users) or len(users)!=len(items) or len(items)!=len(y):
        raise ValueError('Group sizes/feature rows/labels do not align')
    if not np.array_equal(users,np.repeat(np.arange(len(sizes)),sizes)):
        raise ValueError('Candidate rows cross user query boundaries')
    if not np.isin(y,[0,1]).all():
        raise ValueError('Binary Top5 membership labels required')
    offsets=np.r_[0,np.cumsum(sizes)]
    positives=[]
    for lo,hi in zip(offsets,offsets[1:]):
        if len(set(items[lo:hi].tolist()))!=hi-lo:
            raise ValueError('Duplicate programme within ranking query')
        n=int(y[lo:hi].sum())
        if n>5:raise ValueError('More than five relevant programmes in a query')
        positives.append(n)
    return {'queries':len(sizes),'rows':len(y),'positives':int(y.sum()),
        'zero_positive_queries':positives.count(0),'group_sizes_min':int(sizes.min()),'group_sizes_max':int(sizes.max())}


def fit_ranker(X,y,groups,parameters=None,rounds=ROUNDS):
    lgb=lightgbm_module()
    if len(X)!=len(y) or sum(groups)!=len(X):raise ValueError('Ranking rows/groups mismatch')
    p=PARAMETERS.copy()
    if parameters:p.update(parameters)
    train=lgb.Dataset(np.asarray(X,dtype=np.float32),label=y,group=list(map(int,groups)),
        feature_name=[f'f{n}' for n in range(X.shape[1])],free_raw_data=True)
    # No eval_set, no main/earlier targets, no early stopping or round selection.
    return lgb.train(p,train,num_boost_round=rounds)


def query_order(scores):
    return np.argsort(-np.asarray(scores),kind='stable')


def prediction_lists(meta,z,rank_scores,baseline_classifier=None,fusion=False):
    """A: LambdaRank order; B: one equal RRF of LambdaRank and frozen V4."""
    starts=np.r_[0,np.cumsum(meta['pool_lengths'])];result={}
    for n,u in enumerate(meta['users']):
        lo,hi=starts[n:n+2];ids=z['item_index'][lo:hi]
        if z['cold'][n]:
            result[u]=[int(meta['items'][i]) for i in z['fallback'][:5]];continue
        order=query_order(rank_scores[lo:hi])
        if fusion:
            if baseline_classifier is None:raise ValueError('V4 scores required for B')
            old_order=query_order(baseline_classifier[lo:hi])
            old_rank=np.empty(len(order),int);old_rank[old_order]=np.arange(1,len(order)+1)
            # V4 already fuses V2 and its classifier. Preserve it exactly.
            v4=1/(10+np.arange(1,len(order)+1))+1/(10+old_rank)
            v4_order=query_order(v4);v4_rank=np.empty(len(order),int);v4_rank[v4_order]=np.arange(1,len(order)+1)
            rank=np.empty(len(order),int);rank[order]=np.arange(1,len(order)+1)
            order=query_order(1/(10+v4_rank)+1/(10+rank))
        result[u]=[int(meta['items'][i]) for i in ids[order[:5]]]
    return result
