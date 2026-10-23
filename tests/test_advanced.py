import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from tvrec.labels import continuous_sessions, qualifies, target_from_events, training_features
from tvrec.models import (ImplicitALS, ContentProfiles, normalized_scores,
                          ranked_predictions, reciprocal_ranks)


def event(user,sid,item,a,b,duration=100,session=1):
    return {'user_id':user,'sid':sid,'item':item,'a':a,'b':b,
            'duration':duration,'session_id':session,'session_seconds':b-a}


class ContinuousLabelTests(unittest.TestCase):
    def test_production_event_query_clips_before_features(self):
        import duckdb
        from tvrec.labels import event_frame
        c=duckdb.connect()
        c.execute("ATTACH ':memory:' AS base")
        c.execute('CREATE TABLE base.airings(sid BIGINT,duration BIGINT,item BIGINT)')
        c.execute('INSERT INTO base.airings VALUES(1,1000,10)')
        c.execute('CREATE TABLE sessions(session_id BIGINT,a BIGINT,b BIGINT)')
        c.execute('INSERT INTO sessions VALUES(1,-10,1000),(2,100,200)')
        c.execute('CREATE TABLE continuous_events(session_id BIGINT,user_id VARCHAR,sid BIGINT,channel INTEGER,a BIGINT,b BIGINT)')
        c.execute("INSERT INTO continuous_events VALUES(1,'u',1,1,-10,1000),(2,'future',1,1,100,200)")
        frame=event_frame(c,0,100)
        self.assertEqual(frame.user_id.tolist(),['u'])
        self.assertEqual(frame.a.tolist(),[0])
        self.assertEqual(frame.b.tolist(),[100])
        f=training_features(frame,seconds_cut=60,strength='seconds',cutoff=100)
        self.assertEqual(f.score.tolist(),[100])
        self.assertEqual(f.last_second.tolist(),[100])
        c.close()

    def test_independent_endpoint_enumerator(self):
        from tvrec.verify_stage2 import brute_continuous_coverage
        self.assertEqual(brute_continuous_coverage([(1,0,100),(2,40,60)],[(9,0,100)],1,9,0,100),40)
        self.assertEqual(brute_continuous_coverage([(1,0,100)],[(9,0,100),(10,40,60)],1,9,0,100),40)
        self.assertEqual(brute_continuous_coverage([(1,0,100)],[(9,0,100)],1,9,10,100),90)

    def test_strict_boundary(self):
        self.assertTrue(qualifies(80,100))
        self.assertFalse(qualifies(80,100,strict=True))
        self.assertTrue(qualifies(81,100,strict=True))
        self.assertFalse(qualifies(100,0))

    def test_duplicate_fragmentation_and_switches(self):
        sessions,stats=continuous_sessions([(1,0,40),(1,0,40),(1,40,100)])
        self.assertEqual(sessions,[(1,0,100)])
        self.assertEqual(stats['duplicate_intervals'],1)
        self.assertEqual(continuous_sessions([(1,0,40),(2,40,60),(1,60,100)])[0],
                         [(1,0,40),(2,40,60),(1,60,100)])
        self.assertEqual(continuous_sessions([(1,0,40),(1,41,100)])[0],
                         [(1,0,40),(1,41,100)])

    def test_concurrent_channels_are_ambiguous(self):
        sessions,stats=continuous_sessions([(1,0,100),(2,40,60)])
        self.assertEqual(sessions,[(1,0,40),(1,60,100)])
        self.assertEqual(stats['ambiguous_user_seconds'],20)

    def test_separate_visits_never_accumulate_into_positive(self):
        f=pd.DataFrame([event('u',1,10,0,45),event('u',1,10,50,95,session=2)])
        truth,_,_=target_from_events(f)
        self.assertEqual(truth,{})  # 90 cumulative seconds, but no >80% run

    def test_frequency_topfive_not_all_qualifying_items(self):
        rows=[event('u',i,i,i*100,i*100+90) for i in range(1,8)]
        rows += [event('u',90,7,9000,9090),event('u',90,7,9000,9090)]
        truth,pairs,_=target_from_events(pd.DataFrame(rows))
        self.assertEqual(truth['u'],{7,1,2,3,4})
        self.assertEqual(int(pairs.loc[pairs.item==7,'frequency'].iloc[0]),2)

    def test_input_threshold_independent_and_future_clipping(self):
        f=pd.DataFrame([event('u',1,10,0,50),event('u',2,20,100,181)])
        self.assertEqual(set(training_features(f,minimum=.5).item),{10,20})
        self.assertEqual(target_from_events(f,strict=True)[0],{'u':{20}})
        # Later observations cannot alter earlier unambiguous portions.
        before=continuous_sessions([(1,0,100)])[0]
        after=continuous_sessions([(1,0,100),(2,100,200)])[0]
        self.assertEqual(before,[s for s in after if s[1]<100])


class AdvancedMethodTests(unittest.TestCase):
    def test_als_objective_and_reproducibility(self):
        r=sparse.csr_matrix([[5.,4.,0.,0.],[4.,5.,0.,0.],[0.,0.,5.,4.],[0.,0.,4.,5.],[0.,0.,0.,0.]])
        short=ImplicitALS(factors=2,iterations=1,seed=42).fit(r)
        model=ImplicitALS(factors=2,iterations=8,seed=42).fit(r)
        repeat=ImplicitALS(factors=2,iterations=8,seed=42).fit(r)
        self.assertLess(model.objective(r),short.objective(r))
        np.testing.assert_allclose(model.item_factors,repeat.item_factors)
        self.assertTrue(np.all(model.user_factors[-1]==0))
        scores=model.user_factors@model.item_factors.T
        self.assertGreater(scores[0,:2].mean(),scores[0,2:].mean())

    def test_cbf_similarity_and_no_future_vocabulary_fit(self):
        meta=[{'tv_show_title':'cats','tv_show_category':'animals'},
              {'tv_show_title':'dogs','tv_show_category':'animals'},
              {'tv_show_title':'football','tv_show_category':'sport'}]
        future={'tv_show_title':'unseenword','tv_show_category':'animals'}
        model=ContentProfiles().fit_items(meta,meta+[future])
        self.assertNotIn('title_unseenword',model.vectorizer.vocabulary_)
        weights=sparse.csr_matrix([[1.,0.,0.,0.],[0.,0.,0.,0.]])
        scores,profiles=model.score(weights,[0,1,2,3])
        self.assertGreater(scores[0,1],scores[0,2])
        self.assertGreater(scores[0,3],0)  # unseen ID supported by known metadata
        self.assertTrue(np.all(scores[1]==0))

    def test_fusion_scale_invariance_and_common_cold_fallback(self):
        s=np.array([[4.,2.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
        np.testing.assert_allclose(normalized_scores(s),normalized_scores(999*s))
        fallback=np.array([5,4,3,2,1,0])
        a=reciprocal_ranks(s,fallback,top=3)
        b=reciprocal_ranks(s*99,fallback,top=3)
        np.testing.assert_allclose(a,b)
        pred=ranked_predictions(s,['warm','cold'],[10,20,30,40,50,60],fallback,{'cold'})
        self.assertEqual(pred['cold'],[60,50,40,30,20])
        self.assertEqual(pred['warm'],[10,20,30,60,50])


if __name__=='__main__':
    unittest.main()
