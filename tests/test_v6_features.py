import unittest
from fractions import Fraction

from tvrec.core import ap_at_5
import numpy as np
import pandas as pd
from tvrec.habit_features import active_stability,interval_bin_parts,schedule_habit_blocks,schedule_vectors
from tvrec.pipeline import LOW


class InstructorAPRegressionTests(unittest.TestCase):
    def test_instructor_hits_at_1_2_4(self):
        self.assertAlmostEqual(ap_at_5({'A','B','C'},['A','B','D','C','E']),float(Fraction(11,12)),places=12)
        self.assertAlmostEqual(ap_at_5({'A','B','C'},['A','B','D','C','E']),.9166667,places=7)

    def test_instructor_hits_at_2_4_5(self):
        self.assertAlmostEqual(ap_at_5({'A','B','C'},['D','A','E','B','C']),float(Fraction(8,15)),places=12)
        self.assertAlmostEqual(ap_at_5({'A','B','C'},['D','A','E','B','C']),.5333333,places=7)


class HabitInteractionTests(unittest.TestCase):
    def test_exact_midnight_bins_conserve_seconds(self):
        start=LOW+23*3600;end=LOW+25*3600
        row,slot,seconds=interval_bin_parts([start],[end])
        self.assertEqual(row.tolist(),[0,0])
        self.assertEqual(seconds.tolist(),[3600,3600])
        self.assertEqual(slot.tolist(),[11,12])  # Monday night then Tuesday
        self.assertEqual(int(seconds.sum()),end-start)

    def test_channel_conditioning_adds_information_beyond_global_time_habits(self):
        data=[]
        for sid,(u,ch,hour) in enumerate([('u',1,8),('u',2,20),('v',1,20),('v',2,8)]):
            a=LOW+hour*3600
            data.append({'user_id':u,'sid':sid,'channel':ch,'item':100+sid,'a':a,'b':a+7200,
                         'seconds':7200,'duration':7200,'fraction':1.})
        occ=pd.DataFrame(data);cutoff=LOW+7*86400
        future=pd.DataFrame({'item':[10,20,30,40],'channel':[1,1,2,2],
            'ss':[cutoff+8*3600,cutoff+20*3600,cutoff+20*3600,cutoff+8*3600],
            'duration':[7200]*4})
        past=future.copy();past['ss']-=7*86400
        b,c=schedule_habit_blocks(occ,past,future,['u','cold'],[10,20,30,40],
            np.array([0,0,0,0,1]),np.array([0,1,2,3,0]),cutoff,cutoff+7*86400)
        self.assertGreater(b['habit_channel_weektime_compatibility'][0],b['habit_channel_weektime_compatibility'][1])
        self.assertGreater(b['habit_channel_weektime_compatibility'][2],b['habit_channel_weektime_compatibility'][3])
        self.assertEqual(b['habit_previously_supported_airtime_share'][0],1)
        self.assertEqual(b['habit_previously_supported_airtime_share'][1],0)
        self.assertAlmostEqual(b['habit_channel_specific_lift'][4],0)  # cold backs off
        for value in c.values():np.testing.assert_array_equal(value,np.zeros(5))  # no personal completion of these IDs

    def test_relative_stability_is_not_raw_days_and_uses_only_active_weeks(self):
        data=[]
        # Active weeks0,2,3; programme10 is completed in weeks0 and2.
        for sid,(item,day) in enumerate([(10,0),(20,1),(10,14),(20,21)]):
            a=LOW+day*86400
            data.append({'user_id':'u','sid':sid,'channel':1,'item':item,'a':a,'b':a+100,
                         'seconds':100,'duration':100,'fraction':1.})
        result=active_stability(pd.DataFrame(data),['u'],[10],LOW+28*86400)
        self.assertEqual(result['stable_completed_active_day_share'][0],.5)
        self.assertEqual(result['stable_week_return_posterior'][0],.5)  # one return in two eligible transitions
        self.assertAlmostEqual(result['stable_log_return_opportunities'][0],np.log(3))
        self.assertEqual(result['stable_completed_day_concentration'][0],.5)

    def test_future_history_rejected(self):
        occ=pd.DataFrame([{'user_id':'u','sid':1,'channel':1,'item':10,'a':LOW,'b':LOW+100,
                          'seconds':100,'duration':100,'fraction':1.}])
        with self.assertRaises(ValueError):active_stability(occ,['u'],[10],LOW+99)

    def test_schedule_exposure_is_clipped_at_prediction_boundaries(self):
        frame=pd.DataFrame({'item':[10],'channel':[1],'ss':[LOW+23*3600],'duration':[7200]})
        clipped=schedule_vectors(frame,[10],{1:0},'airing_units',LOW+24*3600,LOW+26*3600)
        self.assertEqual(float(clipped[11,0]),0)
        self.assertEqual(float(clipped[12,0]),.5)  # only one of its two hours is in-period
        share=schedule_vectors(frame,[10],{1:0},'airtime_share',LOW+24*3600,LOW+26*3600)
        self.assertEqual(float(share.sum()),1)


if __name__=='__main__':unittest.main()
