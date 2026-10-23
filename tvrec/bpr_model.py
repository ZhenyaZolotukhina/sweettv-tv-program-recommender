"""Small deterministic implicit BPR MF using existing NumPy/SciPy.

Pairwise objective: -log sigmoid(p_u.(q_i-q_j)+b_i-b_j) + sampled L2.
Unobserved j is a weak relative comparison, not a confirmed dislike. Positives
are qualification-defined historical interactions. No future items are sampled.
"""
import numpy as np
from scipy.special import expit


class ImplicitBPR:
    def __init__(self,factors=32,epochs=25,batch_size=2048,learning_rate=.05,
                 regularization=.002,seed=42,positive_weight='binary'):
        self.factors=factors;self.epochs=epochs;self.batch_size=batch_size
        self.learning_rate=learning_rate;self.regularization=regularization
        self.seed=seed;self.positive_weight=positive_weight

    def setup(self,matrix):
        self.matrix=matrix.tocsr().astype(np.float64)
        self.matrix.sum_duplicates();self.matrix.sort_indices()
        if np.any(self.matrix.data<=0):raise ValueError('BPR inputs must be positive implicit evidence')
        self.n_users,self.n_items=self.matrix.shape
        degree=np.diff(self.matrix.indptr)
        self.active_users=np.where((degree>0)&(degree<self.n_items))[0]
        self.keys=np.repeat(np.arange(self.n_users),degree)*self.n_items+self.matrix.indices
        weight=np.ones_like(self.matrix.data) if self.positive_weight=='binary' else np.log1p(self.matrix.data)
        if self.positive_weight not in ['binary','log_frequency']:raise ValueError('Unknown positive weighting')
        self.cdf=np.cumsum(weight)
        ends=np.zeros(self.n_users)
        ok=self.matrix.indptr[1:]>0
        ends[ok]=self.cdf[self.matrix.indptr[1:][ok]-1]
        starts=np.r_[0,ends[:-1]]
        self.start_mass=starts;self.mass=ends-starts
        self.rng=np.random.default_rng(self.seed)
        return self

    def observed(self,u,item):
        key=u*self.n_items+item
        if not len(self.keys):return np.zeros(len(key),dtype=bool)
        index=np.searchsorted(self.keys,key)
        return (index<len(self.keys))&(self.keys[np.minimum(index,len(self.keys)-1)]==key)

    def sample(self,n):
        if not len(self.active_users):return np.array([],int),np.array([],int),np.array([],int)
        u=self.rng.choice(self.active_users,n)
        target=self.start_mass[u]+self.rng.random(n)*self.mass[u]
        position=np.searchsorted(self.cdf,target,side='right')
        assert ((position>=self.matrix.indptr[u])&(position<self.matrix.indptr[u+1])).all()
        i=self.matrix.indices[position]
        j=self.rng.integers(0,self.n_items,n)
        bad=self.observed(u,j)
        while bad.any():
            j[bad]=self.rng.integers(0,self.n_items,int(bad.sum()))
            bad=self.observed(u,j)
        return u,i,j

    def _update(self,parameter,accumulator,index,gradient):
        unique,inverse,count=np.unique(index,return_inverse=True,return_counts=True)
        summed=np.zeros((len(unique),parameter.shape[1]),dtype=np.float64)
        np.add.at(summed,inverse,gradient)
        # Mean per touched entity avoids oversized steps for repeated rows in a
        # minibatch. Sparse AdaGrad, initial accumulator1, is an implementation
        # choice rather than a claim of reproducing LightFM's optimizer exactly.
        summed/=count[:,None]
        accumulator[unique]+=summed*summed
        parameter[unique]+=self.learning_rate*summed/np.sqrt(accumulator[unique])

    def fit(self,matrix):
        self.setup(matrix)
        self.user_factors=self.rng.normal(0,.01,(self.n_users,self.factors))
        self.item_factors=self.rng.normal(0,.01,(self.n_items,self.factors))
        self.item_bias=np.zeros((self.n_items,1))
        au=np.ones_like(self.user_factors);ai=np.ones_like(self.item_factors);ab=np.ones_like(self.item_bias)
        degree=np.diff(self.matrix.indptr)
        active=np.zeros(self.n_users,dtype=bool);active[self.active_users]=True
        self.user_factors[~active]=0  # no pairwise evidence, no invented personal vector
        self.item_factors[self.matrix.getnnz(axis=0)==0]=0
        for _ in range(self.epochs):
            remaining=self.matrix.nnz
            while remaining>0:
                u,i,j=self.sample(min(remaining,self.batch_size));remaining-=self.batch_size
                if not len(u):break
                p=self.user_factors[u].copy();qi=self.item_factors[i].copy();qj=self.item_factors[j].copy()
                bi=self.item_bias[i,0].copy();bj=self.item_bias[j,0].copy()
                g=expit(-(np.einsum('ij,ij->i',p,qi-qj)+bi-bj))
                gu=g[:,None]*(qi-qj)-self.regularization*p
                gi=g[:,None]*p-self.regularization*qi
                gj=-g[:,None]*p-self.regularization*qj
                self._update(self.user_factors,au,u,gu)
                self._update(self.item_factors,ai,np.r_[i,j],np.vstack([gi,gj]))
                gb=np.r_[g-self.regularization*bi,-g-self.regularization*bj][:,None]
                self._update(self.item_bias,ab,np.r_[i,j],gb)
        assert np.isfinite(self.user_factors).all() and np.isfinite(self.item_factors).all()
        return self

    def triplet_loss(self,u,i,j):
        x=np.einsum('ij,ij->i',self.user_factors[u],self.item_factors[i]-self.item_factors[j])
        x+=self.item_bias[i,0]-self.item_bias[j,0]
        return float(np.logaddexp(0,-x).mean())

    def score(self,user_index,item_index):
        return self.user_factors[user_index]@self.item_factors[item_index].T+self.item_bias[item_index,0][None,:]
