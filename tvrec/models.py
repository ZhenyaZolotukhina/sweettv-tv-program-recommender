"""Independent implicit MF and metadata-profile CBF, plus calibrated rankings.

ALS implements the Hu/Koren/Volinsky implicit weighted-square objective with
SciPy SPD solves. TF-IDF uses scikit-learn. No participant solution code copied.
"""
from contextlib import contextmanager
import threading
import time

import numpy as np
import psutil
from scipy import sparse
from scipy.linalg import solve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits


@contextmanager
def measured():
    """Actual wall time and sampled whole-process RSS, including shared data."""
    proc=psutil.Process()
    result={'rss_before_bytes':proc.memory_info().rss,'peak_rss_bytes':proc.memory_info().rss}
    stop=threading.Event()
    def sample():
        while not stop.wait(.02):
            result['peak_rss_bytes']=max(result['peak_rss_bytes'],proc.memory_info().rss)
    thread=threading.Thread(target=sample,daemon=True)
    thread.start()
    start=time.perf_counter()
    try:
        yield result
    finally:
        stop.set()
        thread.join()
        result['seconds']=time.perf_counter()-start
        result['peak_rss_bytes']=max(result['peak_rss_bytes'],proc.memory_info().rss)
        result['peak_increment_bytes']=max(0,result['peak_rss_bytes']-result['rss_before_bytes'])


def matrix_from_features(frame, users, items):
    ui={u:i for i,u in enumerate(users)}
    ii={int(x):i for i,x in enumerate(items)}
    f=frame[frame.user_id.isin(ui)&frame.item.isin(ii)]
    return sparse.csr_matrix((f.score.to_numpy(dtype=np.float64),
        ([ui[u] for u in f.user_id],[ii[int(i)] for i in f.item])),shape=(len(users),len(items)))


class ImplicitALS:
    """p_ui=1[r_ui>0], c_ui=1+alpha*log(1+r_ui); zero preference elsewhere.

    min sum_ui c_ui*(p_ui - x_u@y_i)^2 + reg*(||X||^2+||Y||^2).
    Sparse corrections avoid materializing the dense confidence matrix.
    """
    def __init__(self,factors=32,alpha=10.,regularization=1.,iterations=12,seed=42):
        self.factors=factors
        self.alpha=alpha
        self.regularization=regularization
        self.iterations=iterations
        self.seed=seed

    def fit(self,matrix):
        r=matrix.tocsr().astype(np.float64)
        rng=np.random.default_rng(self.seed)
        self.user_factors=rng.normal(0,.01,(r.shape[0],self.factors))
        self.item_factors=rng.normal(0,.01,(r.shape[1],self.factors))
        # Thread oversubscription severely hurts these small dense systems.
        with threadpool_limits(limits=1):
            for _ in range(self.iterations):
                self.user_factors=self._step(r,self.item_factors)
                self.item_factors=self._step(r.T.tocsr(),self.user_factors)
        return self

    def _step(self,rows,other):
        gram=other.T@other+self.regularization*np.eye(self.factors)
        out=np.zeros((rows.shape[0],self.factors))
        for u in range(rows.shape[0]):
            start,end=rows.indptr[u:u+2]
            if end==start:
                continue
            idx=rows.indices[start:end]
            extra=self.alpha*np.log1p(rows.data[start:end])
            y=other[idx]
            a=gram+(y.T*extra)@y
            b=y.T@(1+extra)
            out[u]=solve(a,b,assume_a='pos',check_finite=False)
        return out

    def objective(self,matrix):
        r=matrix.tocoo()
        scores=self.user_factors@self.item_factors.T
        value=float(np.square(scores).sum())
        observed=scores[r.row,r.col]
        extra=self.alpha*np.log1p(r.data)
        value+=float(((1+extra)*np.square(1-observed)-np.square(observed)).sum())
        return value+self.regularization*(np.square(self.user_factors).sum()+np.square(self.item_factors).sum())


def document(row,rich=True):
    import re
    fields=['tv_show_category','tv_show_genre_1','tv_show_genre_2','tv_show_genre_3']
    if rich:
        fields+=['tv_show_title','year_of_production','director','actors']
    tokens=[]
    for field in fields:
        value=row.get(field) or ''
        if not isinstance(value,str):
            continue
        words=re.findall(r'[^\W_]+',value.lower(),flags=re.UNICODE)
        prefix={'tv_show_category':'category','tv_show_title':'title'}.get(field,field)
        # Genre columns share a namespace rather than encode arbitrary slot number.
        if field.startswith('tv_show_genre'):
            prefix='genre'
        tokens.extend(prefix+'_'+word for word in words)
    return ' '.join(tokens)


class ContentProfiles:
    def __init__(self,rich=True,max_features=40000):
        self.rich=rich
        self.max_features=max_features

    def fit_items(self,training_metadata,all_metadata):
        self.vectorizer=TfidfVectorizer(token_pattern=r'(?u)\b\w+\b',min_df=1,
            max_features=self.max_features,sublinear_tf=True,dtype=np.float32)
        self.vectorizer.fit([document(row,self.rich) for row in training_metadata])
        self.item_vectors=self.vectorizer.transform([document(row,self.rich) for row in all_metadata])
        return self

    def score(self,user_item_weights,candidate_indices):
        weights=user_item_weights.copy()
        weights.data=np.log1p(weights.data)
        profiles=normalize(weights@self.item_vectors,norm='l2')
        result=(profiles@self.item_vectors[candidate_indices].T).toarray().astype(np.float32)
        return result,profiles


def score_features(frame,users,items):
    return matrix_from_features(frame,users,items).toarray().astype(np.float32)


def popularity_scores(frame,users,items):
    totals=frame.groupby('item').score.sum().to_dict()
    row=np.array([totals.get(int(i),0) for i in items],dtype=np.float32)
    return np.broadcast_to(row,(len(users),len(items))).copy()


def ranked_predictions(scores,users,items,fallback,cold=None,limit=5):
    """Positive model scores first; unsupported items/cold users use common A.

    Stable tie-break: fallback popularity order, then item ID in fallback itself.
    MF may have negative scores; they receive zero support rather than outrank
    unavailable latent items through arbitrary zeros.
    """
    items=np.asarray(items)
    fallback=np.asarray(fallback,dtype=int)
    out={}
    for u,user in enumerate(users):
        if cold is not None and user in cold:
            order=fallback
        else:
            order=fallback[np.argsort(-scores[u,fallback],kind='stable')]
            positive=order[scores[u,order]>0]
            supported=set(positive.tolist())
            order=np.concatenate([positive,np.array([i for i in fallback if i not in supported],dtype=int)])
        out[user]=[int(x) for x in items[order[:limit]]]
    return out


def normalized_scores(scores):
    scores=np.maximum(scores,0).astype(np.float32)
    maximum=scores.max(axis=1,keepdims=True)
    return np.divide(scores,maximum,out=np.zeros_like(scores),where=maximum>0)


def reciprocal_ranks(scores,fallback,top=200,k=10):
    out=np.zeros_like(scores,dtype=np.float32)
    for u in range(len(scores)):
        order=fallback[np.argsort(-scores[u,fallback],kind='stable')][:top]
        order=order[scores[u,order]>0]
        out[u,order]=1/(k+np.arange(1,len(order)+1))
    return out
