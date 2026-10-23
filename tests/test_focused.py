import unittest

import numpy as np
import pandas as pd

from tvrec.focused_features import consistency_features,loyalty_features,opportunity_features
from tvrec.pipeline import LOW


def occurrence(user,sid,item,ch,start,seconds):
    return {'user_id':user,'sid':sid,'item':item,'channel':ch,'a':start,
            'b':start+seconds,'seconds':seconds,'duration':100,'fraction':seconds/100}


class FocusedFeaturesTests(unittest.TestCase):
    def test_ablation_control_uses_frozen_v3_learning_and_temporal_settings(self):
        from tvrec.focused import PLAN
        from tvrec.rerank import PLAN as V3
        self.assertEqual(PLAN['model_parameters'],V3['model'])
        self.assertEqual(PLAN['training_folds'],V3['training_folds'])
        self.assertEqual(set(PLAN['experiments']),{'consistency','loyalty','opportunity'})
        self.assertEqual(PLAN['maximum_experiments'],3)
        self.assertFalse(PLAN['combinations_or_weight_search'])

    def test_completed_days_and_weeks_not_just_total_frequency(self):
        occ=pd.DataFrame([occurrence('u',1,10,1,LOW+1,90),
                          occurrence('u',2,10,1,LOW+86400+1,60),
                          occurrence('u',3,10,1,LOW+7*86400+1,90)])
        values=consistency_features(occ,['u','u'],[10,20],LOW+14*86400)
        self.assertAlmostEqual(values['cons_log_days50'][0],np.log(4))
        self.assertAlmostEqual(values['cons_log_weeks50'][0],np.log(3))
        self.assertAlmostEqual(values['cons_log_days80'][0],np.log(3))
        self.assertEqual(values['cons_completed_week_share'][0],1)
        self.assertAlmostEqual(values['cons_completed_day_fraction'][0],2/3)
        self.assertEqual(values['cons_log_days50'][1],0)

    def test_duplicate_events_and_future_views_rejected(self):
        row=occurrence('u',1,10,1,LOW,90)
        with self.assertRaises(ValueError):
            consistency_features(pd.DataFrame([row,row]),['u'],[10],LOW+100)
        with self.assertRaises(ValueError):
            consistency_features(pd.DataFrame([row]),['u'],[10],LOW+80)

    def test_channel_completion_affinity_hand_calculation(self):
        occ=pd.DataFrame([occurrence('u',1,10,1,LOW,90),occurrence('u',2,10,1,LOW+100,90),
                          occurrence('u',3,30,2,LOW+200,60),occurrence('v',4,20,1,LOW+300,90)])
        future=pd.DataFrame({'item':[20],'channel':[1]})
        v=loyalty_features(occ,future,['u','cold'],[20,20])
        self.assertAlmostEqual(v['loyal_raw_loyalty'][0],2/3)
        self.assertAlmostEqual(v['loyal_duration_loyalty'][0],.75)
        self.assertEqual(v['loyal_completion_ratio'][0],1)
        self.assertAlmostEqual(v['loyal_complete_popularity'][0],1/3)
        self.assertAlmostEqual(v['loyal_loyalty_complete_popularity'][0],2/9)
        self.assertEqual(v['loyal_raw_loyalty'][1],0)
        self.assertEqual(v['loyal_loyalty_complete_popularity'][1],0)

    def test_future_airings_do_not_create_personal_relevance_without_completions(self):
        occ=pd.DataFrame([occurrence('u',1,10,1,LOW,90)])
        past=pd.DataFrame({'item':[10,10]})
        future=pd.DataFrame({'item':[10]*3+[20]*100,'ss':[LOW+200]*103,'duration':[100]*103})
        v=opportunity_features(occ,past,future,['u'],[10,20],np.array([0,0]),np.array([0,1]),LOW+100,LOW+1000)
        self.assertEqual(v['opportunity_personal_completion_rate'][0],.5)
        self.assertAlmostEqual(v['opportunity_log_expected_completions'][0],np.log1p(.5*3*.25))
        for values in v.values():
            self.assertEqual(values[1],0)


if __name__=='__main__':unittest.main()
