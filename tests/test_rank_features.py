import unittest

import numpy as np
import pandas as pd

from tvrec.rank_features import (qualified_occurrences,repeat_variants,channel_scores,
                                 time_slots,habit_scores,opportunities)


class RankingFeatureTests(unittest.TestCase):
    def test_qualification_no_cumulative_visits_or_duplicate_credit(self):
        f=pd.DataFrame({'user_id':['u']*4,'sid':[1,1,2,2],'item':[10]*4,'channel':[1]*4,
            'a':[0,0,0,50],'b':[90,90,40,90],'duration':[100]*4})
        occurrences=qualified_occurrences(f,100)
        self.assertEqual(occurrences.sid.tolist(),[1])
        with self.assertRaises(ValueError):
            qualified_occurrences(f,80)

    def test_training_only_confidence_and_mild_recency(self):
        f=pd.DataFrame({'user_id':['u','u'],'sid':[1,2],'item':[10,10],'channel':[1,1],
            'a':[0,100],'b':[90,160],'duration':[100,100]})
        v,pairs=repeat_variants(qualified_occurrences(f,200),['u'],[10],200)
        self.assertEqual(float(v['frequency'][0,0]),2)
        self.assertEqual(float(v['confidence'][0,0]),3)
        self.assertGreaterEqual(float(v['recency'][0,0]),1.5)
        self.assertLessEqual(float(v['recency'][0,0]),2)

    def test_channel_probabilities_sqrt_profile_and_unseen_items(self):
        f=pd.DataFrame({'user_id':['u']*5+['v'],'channel':[1,1,1,1,2,1],
                        'item':[10,10,10,10,20,30]})
        score=channel_scores(f,['u','cold'],[10,20,30,40])
        np.testing.assert_allclose(score[0],[8/15,1/3,2/15,0],rtol=1e-6)
        self.assertTrue(np.all(score[1]==0))
        # Removing an unavailable item from candidates does not renormalize its
        # training channel popularity mass into the remaining candidates.
        score=channel_scores(f,['u'],[30])
        self.assertAlmostEqual(float(score[0,0]),2/15,places=6)

    def test_clock_weekday_slots(self):
        dates=np.array(['2020-03-09T00:00:00','2020-03-09T16:00:00',
                        '2020-03-15T23:59:59'],dtype='datetime64[s]').astype('int64')
        self.assertEqual(time_slots(dates).tolist(),[0,4,41])

    def test_habit_matching_uses_known_schedule_not_future_views(self):
        f=pd.DataFrame({'user_id':['u'],'a':[0],'b':[3600]})
        future=pd.DataFrame({'item':[10,20],'ss':[0,43200],'duration':[3600,3600]})
        scores=habit_scores(f,future,['u'],[10,20])
        self.assertGreater(scores[0,0],scores[0,1])

    def test_smoothed_schedule_opportunities(self):
        past=pd.DataFrame({'item':[1]*10+[2]*2})
        future=pd.DataFrame({'item':[1]*2+[2]*10+[3]})
        value,_,_=opportunities(past,future,[1,2,3],10,10)
        np.testing.assert_allclose(value,[np.sqrt(6/14),np.sqrt(14/6),np.sqrt(5/4)],rtol=1e-6)
        self.assertTrue(np.all((value>=.5)&(value<=2)))


if __name__=='__main__':
    unittest.main()
