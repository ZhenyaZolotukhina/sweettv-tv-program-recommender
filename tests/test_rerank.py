import copy
import unittest

import numpy as np
import pandas as pd

from tvrec.rerank import PLAN,audit_training_cutoffs,query_predictions
from tvrec.rerank_data import EARLIER,order_row,retrieve,build_features
from tvrec.pipeline import LOW,SPLIT,TEST


class TemporalSupervisionTests(unittest.TestCase):
    def test_training_labels_finish_before_both_holdouts(self):
        self.assertTrue(audit_training_cutoffs(PLAN['training_folds'],EARLIER))
        future=copy.deepcopy(PLAN['training_folds'])
        future[-1]['target_end']=EARLIER+1
        with self.assertRaises(ValueError):
            audit_training_cutoffs(future,EARLIER)

    def test_overlapping_target_folds_rejected(self):
        folds=copy.deepcopy(PLAN['training_folds'])
        folds[0]['target_end']=folds[1]['cutoff']+1
        with self.assertRaises(ValueError):
            audit_training_cutoffs(folds,EARLIER)

    def test_no_random_or_heldout_early_stopping(self):
        self.assertIs(PLAN['model']['early_stopping'],False)
        self.assertFalse(PLAN['fit_heldout_targets'])

    def test_final_refit_labels_are_not_legal_for_reported_holdouts(self):
        final=copy.deepcopy(PLAN['training_folds'])+[
            {'cutoff':EARLIER,'target_end':SPLIT},{'cutoff':SPLIT,'target_end':TEST}]
        self.assertTrue(audit_training_cutoffs(final,TEST))
        with self.assertRaises(ValueError):
            audit_training_cutoffs(final,EARLIER)
        with self.assertRaises(ValueError):
            audit_training_cutoffs(final,SPLIT)


class RetrievalAndRerankTests(unittest.TestCase):
    def test_feature_table_is_invariant_to_target_labels(self):
        occ=pd.DataFrame({'user_id':['u'],'item':[10],'sid':[1],'a':[LOW],
            'b':[LOW+90],'seconds':[90],'duration':[100],'fraction':[.9]})
        schedule=pd.DataFrame({'item':[10,20],'duration':[100,100],'channel':[1,2]})
        scores={name:np.array([[2.,1.]],dtype=np.float32) for name in ['repeat','cf','cbf','popularity','channel','v2']}
        ctx={'users':['u'],'items':[10,20],'cutoff':LOW+100,'upper':LOW+200,
            'occurrences':occ,'scores':scores,'fallback':np.array([0,1]),'past':schedule,
            'future':schedule,'metadata':[{'tv_show_category':'news'},{'tv_show_category':'series'}],
            'truth':{'u':{10}}}
        vocab={'categories':['news'],'genres':[]}
        before,names,_,_=build_features(ctx,[np.array([0,1])],vocab)
        ctx['truth']={'u':{20}}
        after,_,_,_=build_features(ctx,[np.array([0,1])],vocab)
        np.testing.assert_array_equal(before,after)
        self.assertNotIn('user_id',names)
        self.assertNotIn('item_id',names)
        self.assertNotIn('truth',names)

    def test_positive_order_and_deterministic_padding(self):
        np.testing.assert_array_equal(order_row(np.array([0.,2.,2.,0.]),np.array([2,0,1,3])),[2,1,0,3])

    def test_candidate_pool_keeps_v2_top200_and_expands_without_labels(self):
        fallback=np.arange(400)
        score=np.arange(400,0,-1,dtype=float)[None,:]
        scores={k:score.copy() for k in ['v2','repeat','cf','cbf','popularity','channel']}
        scores['cf'][0,350]=1000
        ctx={'users':['u'],'items':list(range(400)),'warm':{'u'},'scores':scores,'fallback':fallback}
        pools,_=retrieve(ctx)
        self.assertTrue(set(range(200))<=set(pools[0]))
        self.assertIn(350,pools[0])
        self.assertEqual(pools[0][:200].tolist(),list(range(200)))

    def test_reranker_cold_fallback_and_unique_rankings(self):
        meta={'pool_lengths':[6,6],'users':['warm','cold'],'items':[10,20,30,40,50,60]}
        z={'item_index':np.tile(np.arange(6),2),'cold':np.array([False,True]),'fallback':np.arange(5,-1,-1)}
        p=np.array([.1,.2,.3,.4,.5,.6]*2)
        pred=query_predictions(meta,z,p,'supervised_only')
        self.assertEqual(pred['warm'],[60,50,40,30,20])
        self.assertEqual(pred['cold'],[60,50,40,30,20])
        blend=query_predictions(meta,z,p,'v2_supervised_rrf_equal')
        self.assertEqual(len(set(blend['warm'])),5)
        self.assertEqual(blend['cold'],pred['cold'])


if __name__=='__main__':
    unittest.main()
