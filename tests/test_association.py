import unittest
import numpy as np
from scipy import sparse

from tvrec.association_model import ItemAssociations
from tvrec.association_data import calibrated_bpr_features


class AssociationModelTests(unittest.TestCase):
    def test_distinct_user_support_not_repeated_views(self):
        r=sparse.csr_matrix([[100.,1.,0.],[1.,20.,1.],[0.,1.,1.]])
        model=ItemAssociations('confidence',min_users=1,prior=0).fit(r)
        np.testing.assert_array_equal(model.support,[2,3,2])
        self.assertEqual(float(model.co[0,1]),2)
        self.assertEqual(float(model.similarity[0,1]),1)
        self.assertAlmostEqual(float(model.similarity[1,0]),2/3,places=6)
        binary=r.copy();binary.data.fill(1)
        other=ItemAssociations('confidence',min_users=1,prior=0).fit(binary)
        np.testing.assert_array_equal(model.similarity.toarray(),other.similarity.toarray())

    def test_shrinkage_and_minimum_support_remove_spurious_pairs(self):
        r=sparse.csr_matrix([[1.,1.,0.],[1.,1.,1.],[0.,1.,1.]])
        model=ItemAssociations('shrink_cosine',min_users=2,shrinkage=20).fit(r)
        self.assertEqual(float(model.similarity[0,2]),0)
        self.assertAlmostEqual(float(model.similarity[0,1]),2/np.sqrt(6)*2/22,places=6)
        self.assertTrue(np.all(model.similarity.diagonal()==0))

    def test_seen_items_are_not_filtered_and_rule_pruning_is_deterministic(self):
        r=sparse.csr_matrix([[1.,1.,0.],[1.,1.,1.],[0.,1.,1.]])
        model=ItemAssociations('confidence',min_users=1,prior=0,neighbors=1).fit(r)
        self.assertTrue(np.all(np.diff(model.similarity.indptr)<=1))
        # Equal confidence 1->0 and 1->2 breaks tie toward smaller item index.
        self.assertEqual(model.similarity.getrow(1).indices.tolist(),[0])
        score,count,mass=model.score(np.array([0]))
        self.assertGreater(score[0,0],0)  # already watched, supported by another seed
        self.assertGreater(count[0,0],0)
        self.assertLessEqual(mass.max(),1.000001)


class CalibrationTests(unittest.TestCase):
    def test_query_score_calibration_is_affine_invariant_and_missing_explicit(self):
        raw=np.array([[1.,2.,3.,-np.inf]])
        bias=np.array([.1,.2,.3,0.]);un=np.array([2.]);ino=np.array([2.,2.,2.,0.])
        uk=np.array([True]);ik=np.array([True,True,True,False]);u=np.zeros(4,int);i=np.arange(4)
        a=calibrated_bpr_features(raw,bias,un,ino,uk,ik,u,i,np.array([1.,2.,3.]),np.array([1.,2.,3.]),np.array([.1,.2,.3]))
        b=calibrated_bpr_features(raw*5+7,bias*5+7,un*np.sqrt(5),ino*np.sqrt(5),uk,ik,u,i,
            np.array([1.,2.,3.])*np.sqrt(5),np.array([1.,2.,3.])*np.sqrt(5),np.array([.1,.2,.3])*5+7)
        np.testing.assert_allclose(a[:3],b[:3],rtol=1e-6)
        self.assertTrue(np.isnan(a[3,0]))
        self.assertEqual(a[3,-1],0)

    def test_calibration_does_not_change_when_retrieved_pool_changes(self):
        args=(np.array([[1.,2.,3.]]),np.array([0.,0.,0.]),np.array([2.]),np.ones(3),np.array([True]),np.ones(3,bool))
        hist=(np.array([1.,2.]),np.ones(3),np.array([0.,0.,0.]))
        full=calibrated_bpr_features(*args,np.zeros(3,int),np.arange(3),*hist)
        part=calibrated_bpr_features(*args,np.zeros(2,int),np.array([2,0]),*hist)
        np.testing.assert_allclose(part,full[[2,0]],equal_nan=True)


if __name__=='__main__':unittest.main()
