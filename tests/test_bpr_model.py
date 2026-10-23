import unittest
import tempfile
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from tvrec.bpr_model import ImplicitBPR


class BPRTests(unittest.TestCase):
    def test_sampling_excludes_known_positives_and_saturated_users(self):
        r=sparse.csr_matrix([[1.,1.,0.],[1.,1.,1.],[0.,0.,0.],[0.,0.,2.]])
        model=ImplicitBPR(seed=42).setup(r)
        u,i,j=model.sample(500)
        self.assertTrue(np.all(np.asarray(r[u,i]).ravel()>0))
        self.assertTrue(np.all(np.asarray(r[u,j]).ravel()==0))
        self.assertNotIn(1,set(u))
        self.assertNotIn(2,set(u))

    def test_frequency_sampler_is_distinct_but_does_not_sample_future_items(self):
        r=sparse.csr_matrix([[1.,30.,0.]])
        binary=ImplicitBPR(seed=42,positive_weight='binary').setup(r)
        frequency=ImplicitBPR(seed=42,positive_weight='log_frequency').setup(r)
        _,a,_=binary.sample(4000);_,b,j=frequency.sample(4000)
        self.assertGreater(np.mean(b==1),np.mean(a==1)+.25)
        self.assertTrue(np.all(j==2))

    def test_pairwise_loss_decreases_and_seed_is_reproducible(self):
        r=sparse.csr_matrix([[5.,4.,0.,0.],[4.,5.,0.,0.],[0.,0.,5.,4.],[0.,0.,4.,5.],[0.,0.,0.,0.]])
        short=ImplicitBPR(factors=8,epochs=1,batch_size=4,learning_rate=.1,seed=42).fit(r)
        long=ImplicitBPR(factors=8,epochs=80,batch_size=4,learning_rate=.1,seed=42).fit(r)
        same=ImplicitBPR(factors=8,epochs=80,batch_size=4,learning_rate=.1,seed=42).fit(r)
        u=np.array([0,0,1,1,2,2,3,3]);i=np.array([0,1,0,1,2,3,2,3]);j=np.array([2,3,2,3,0,1,0,1])
        self.assertLess(long.triplet_loss(u,i,j),short.triplet_loss(u,i,j))
        np.testing.assert_array_equal(long.user_factors,same.user_factors)
        self.assertTrue(np.all(long.user_factors[-1]==0))
        scores=long.score(np.array([0]),np.arange(4))[0]
        self.assertGreater(scores[:2].mean(),scores[2:].mean())

    def test_item_bias_and_dot_exactly_reconstruct_score(self):
        r=sparse.csr_matrix([[2.,0.,1.],[0.,2.,0.]])
        model=ImplicitBPR(factors=4,epochs=3,batch_size=2).fit(r)
        expected=model.user_factors@model.item_factors.T+model.item_bias[:,0]
        np.testing.assert_allclose(model.score(np.arange(2),np.arange(3)),expected)

    def test_new_cf_features_handle_negative_and_unseen_scores_honestly(self):
        from tvrec.new_cf_rerank import features_for_indices,FEATURES
        X=features_for_indices(np.array([[5.,-2.,-np.inf]]),np.array([.5,-.5,0.]),
            np.array([2.]),np.array([3.,2.,0.]),np.array([True]),np.array([True,True,False]),
            np.array([0,0,0]),np.array([0,1,2]),np.arange(3))
        np.testing.assert_allclose(X[:,0],[5,-2,0])
        np.testing.assert_allclose(X[:,3],[.75,-.375,0])
        self.assertEqual(X[-1,-1],0)
        self.assertTrue(np.isfinite(X).all())
        self.assertFalse(any('user_bias' in name for name in FEATURES))

    def test_early_fallback_uses_own_historical_frequency_not_later_statistics(self):
        from tvrec.new_cf_rerank import cutoff_fallback
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            (folder/'manifest.json').write_text(json.dumps({'model_items':[10,20]}))
            with (folder/'model.pkl').open('wb') as f:
                pickle.dump(SimpleNamespace(matrix=sparse.csr_matrix([[1.,3.],[2.,5.]])),f)
            result=cutoff_fallback('synthetic_early_never_later',folder,{'items':[10,20,30]})
            np.testing.assert_array_equal(result,[1,0,2])


if __name__=='__main__':unittest.main()
