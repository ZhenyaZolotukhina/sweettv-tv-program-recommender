import unittest

import numpy as np

from tvrec.core import ap_at_5
from tvrec.group_rank import (labels_for_pool,validate_fold_groups,fit_ranker,
                             prediction_lists)


class RankingGroupTests(unittest.TestCase):
    def test_group_sizes_boundaries_and_binary_labels(self):
        user=np.array([0,0,0,1,1]);items=np.array([10,20,30,10,20]);y=np.array([1,0,0,0,0])
        audit=validate_fold_groups(user,items,np.array([3,2]),y)
        self.assertEqual(audit['queries'],2)
        self.assertEqual(audit['zero_positive_queries'],1)
        for sizes in [np.array([2,3]),np.array([3,1]),np.array([3,0,2])]:
            with self.assertRaises(ValueError):validate_fold_groups(user,items,sizes,y)
        with self.assertRaises(ValueError):validate_fold_groups(user,items,np.array([3,2]),np.array([2,0,0,0,0]))
        with self.assertRaises(ValueError):validate_fold_groups(user,np.array([10,10,30,10,20]),np.array([3,2]),y)

    def test_top5_membership_not_all_completed_items(self):
        meta={'users':['u'],'items':[10,20,30,40,50,60]}
        z={'user_index':np.zeros(6,dtype=int),'item_index':np.arange(6)}
        y=labels_for_pool(meta,z,{'u':{20,30,40,50,60}})
        np.testing.assert_array_equal(y,[0,1,1,1,1,1])
        with self.assertRaises(ValueError):labels_for_pool(meta,z,{'u':set(meta['items'])})

    def test_same_user_in_two_folds_remains_two_queries(self):
        meta={'users':['same-user'],'items':[10,20,30]}
        z={'user_index':np.zeros(3,dtype=int),'item_index':np.arange(3)}
        a=labels_for_pool(meta,z,{'same-user':{10}});b=labels_for_pool(meta,z,{'same-user':{20}})
        one=validate_fold_groups(z['user_index'],z['item_index'],np.array([3]),a)
        two=validate_fold_groups(z['user_index'],z['item_index'],np.array([3]),b)
        self.assertEqual(one['queries']+two['queries'],2)
        self.assertEqual([len(a),len(b)],[3,3])

    def test_real_lambdarank_learns_order_and_is_seed_reproducible(self):
        quality=np.tile(np.arange(12),8).astype(np.float32)
        X=quality[:,None];y=(quality==11).astype(int);groups=[12]*8
        params={'min_data_in_leaf':1,'num_leaves':3,'lambda_l2':.1,'num_threads':1}
        a=fit_ranker(X,y,groups,params,rounds=15)
        b=fit_ranker(X,y,groups,params,rounds=15)
        self.assertEqual(a.params['objective'],'lambdarank')
        np.testing.assert_array_equal(a.predict(X),b.predict(X))
        score=a.predict(X)[:12]
        predicted=np.argsort(-score,kind='stable')[:5].tolist()
        self.assertEqual(ap_at_5({11},predicted),1.)

    def test_cold_fallback_preserved_and_fusion_requires_v4(self):
        meta={'users':['warm','cold'],'items':[10,20,30,40,50,60],'pool_lengths':[6,6]}
        z={'item_index':np.tile(np.arange(6),2),'cold':np.array([False,True]),'fallback':np.arange(5,-1,-1)}
        scores=np.array([6,5,4,3,2,1]*2,dtype=float)
        result=prediction_lists(meta,z,scores)
        self.assertEqual(result['warm'],[10,20,30,40,50])
        self.assertEqual(result['cold'],[60,50,40,30,20])
        with self.assertRaises(ValueError):prediction_lists(meta,z,scores,fusion=True)

    def test_negative_raw_ranking_scores_still_produce_five_valid_ids(self):
        user='18446744073709551615'
        meta={'users':[user],'items':[10,20,30,40,50,60],'pool_lengths':[6]}
        z={'item_index':np.arange(6),'cold':np.array([False]),'fallback':np.arange(6)}
        result=prediction_lists(meta,z,np.array([-6.,-5.,-4.,-3.,-2.,-1.]))
        self.assertEqual(list(result),[user])
        self.assertEqual(result[user],[60,50,40,30,20])


if __name__=='__main__':unittest.main()
