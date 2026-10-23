"""Scalable one-item association rules from distinct historical user baskets.

Confidence and shrinkage-cosine are empirical co-view measures, not causal
implications or calibrated future probabilities. No external solution code used.
"""
import numpy as np
from scipy import sparse


class ItemAssociations:
    def __init__(self,kind='confidence',min_users=3,shrinkage=20.,prior=10.,neighbors=200):
        self.kind=kind;self.min_users=min_users;self.shrinkage=shrinkage;self.prior=prior;self.neighbors=neighbors

    def fit(self,frequency_matrix):
        self.frequency=frequency_matrix.tocsr().astype(np.float32)
        self.frequency.sum_duplicates();self.frequency.sort_indices()
        if np.any(self.frequency.data<=0):raise ValueError('Positive qualifying counts required')
        binary=self.frequency.copy();binary.data.fill(1)
        self.binary=binary
        self.support=np.asarray(binary.sum(axis=0)).ravel()
        count=(binary.T@binary).tocsr();count.setdiag(0);count.eliminate_zeros()
        self.co=count
        if self.kind=='confidence':
            similarity=count.copy()
            row=np.repeat(np.arange(count.shape[0]),np.diff(count.indptr))
            similarity.data/=self.support[row]+self.prior
        elif self.kind=='shrink_cosine':
            similarity=count.copy()
            row=np.repeat(np.arange(count.shape[0]),np.diff(count.indptr));col=count.indices
            similarity.data/=np.sqrt(self.support[row]*self.support[col])
            similarity.data*=count.data/(count.data+self.shrinkage)
        else:raise ValueError('Unknown association score')
        similarity.data[count.data<self.min_users]=0;similarity.eliminate_zeros()
        # Deterministic row-wise neighbor pruning; IDs map to sorted item columns.
        rows=[];cols=[];values=[]
        for i in range(similarity.shape[0]):
            lo,hi=similarity.indptr[i:i+2];j=similarity.indices[lo:hi];v=similarity.data[lo:hi]
            order=np.lexsort((j,-v))[:self.neighbors]
            rows.extend([i]*len(order));cols.extend(j[order]);values.extend(v[order])
        self.similarity=sparse.csr_matrix((values,(rows,cols)),shape=similarity.shape,dtype=np.float32)
        return self

    def profiles(self):
        p=self.frequency.copy();p.data=np.log1p(p.data)
        total=np.asarray(p.sum(axis=1)).ravel()
        return sparse.diags(np.divide(1,total,out=np.zeros_like(total),where=total>0))@p

    def score(self,user_rows):
        p=self.profiles()[user_rows]
        score=(p@self.similarity).toarray().astype(np.float32)
        edges=self.similarity.copy();edges.data.fill(1)
        seed=self.binary[user_rows]
        count=(seed@edges).toarray().astype(np.float32)
        mass=(p@edges).toarray().astype(np.float32)
        return score,count,mass
