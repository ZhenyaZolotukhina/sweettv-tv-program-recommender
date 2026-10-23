import copy
import csv
import json
from pathlib import Path
import unittest

import duckdb
import numpy as np
import pandas as pd

from tvrec.final_refit import verify_selected, assert_history_only
from tvrec.models import reciprocal_ranks, ranked_predictions
from tvrec.pipeline import LOW
from tvrec.verify_final import validate_file_records, independently_fuse_row


class FrozenStrategyTests(unittest.TestCase):
    def setUp(self):
        p=Path('artifacts/stage2')
        self.strategy=json.loads((p/'selected_strategy.json').read_text())
        self.board=json.loads((p/'leaderboard.json').read_text())
        self.fusions=json.loads((p/'fusion_results.json').read_text())

    def test_exact_measured_winner_and_no_silent_parameter_change(self):
        self.assertTrue(verify_selected(self.strategy,self.board,self.fusions)['verified'])
        changed=copy.deepcopy(self.strategy)
        changed['selected_config']['weights']['cf']=.2
        with self.assertRaises(AssertionError):
            verify_selected(changed,self.board,self.fusions)
        changed=copy.deepcopy(self.strategy)
        changed['individual_configs']['cf']['seed']=43
        with self.assertRaises(AssertionError):
            verify_selected(changed,self.board,self.fusions)


class FinalLeakageTests(unittest.TestCase):
    def test_hidden_views_and_hidden_airings_rejected(self):
        c=duckdb.connect()
        c.execute("ATTACH ':memory:' AS base")
        c.execute('CREATE TABLE base.views(vs BIGINT,ve BIGINT)')
        c.execute('CREATE TABLE base.airings(sid BIGINT,ss BIGINT,period INTEGER)')
        c.execute('CREATE TABLE continuous_events(sid BIGINT,b BIGINT)')
        cutoff=LOW+100
        c.execute('INSERT INTO base.views VALUES(?,?)',[LOW+10,LOW+90])
        c.execute('INSERT INTO base.airings VALUES(1,?,11)',[LOW+10])
        c.execute('INSERT INTO continuous_events VALUES(1,?)',[LOW+90])
        frame=pd.DataFrame({'a':[LOW+10],'b':[LOW+90],'item':[10],'duration':[100]})
        features=pd.DataFrame({'last_second':[LOW+90]})
        self.assertTrue(assert_history_only(c,frame,features,cutoff)['passed'])
        c.execute('INSERT INTO base.views VALUES(?,?)',[cutoff+1,cutoff+100])
        with self.assertRaises(AssertionError):
            assert_history_only(c,frame,features,cutoff)
        c.execute('DELETE FROM base.views WHERE vs>?',[cutoff])
        c.execute('INSERT INTO base.airings VALUES(2,?,31)',[cutoff+1])
        c.execute('INSERT INTO continuous_events VALUES(2,?)',[LOW+90])
        with self.assertRaises(AssertionError):
            assert_history_only(c,frame,features,cutoff)
        c.close()


class IndependentFinalChecks(unittest.TestCase):
    def test_strict_formatting_original_ids_order_and_availability(self):
        users=['18446744073709551615','0000123']
        rows=[[u,'1 2 3 4 5'] for u in users]
        validate_file_records(users,['user_id','tv_show_id'],rows,{1,2,3,4,5})
        for value in ['0 1 2 3 4','1 1 2 3 4','1 2 3 4 NaN','1 2 3 4 6','1  2 3 4 5','01 2 3 4 5']:
            with self.assertRaises(ValueError):
                validate_file_records(users[:1],['user_id','tv_show_id'],[[users[0],value]],{1,2,3,4,5})
        with self.assertRaises(ValueError):
            validate_file_records(users,['user_id','tv_show_id'],list(reversed(rows)),{1,2,3,4,5})

    def test_independent_rrf_matches_frozen_float32_and_ties(self):
        fallback=np.array([4,2,0,3,1,5])
        component_rows={'repeat':np.array([2,0,2,0,1,0],dtype=np.float32),
                        'cf':np.array([.2,.3,0,.5,0,0],dtype=np.float32),
                        'cbf':np.array([0,.5,0,.5,0,.1],dtype=np.float32),
                        'popularity':np.array([1,0,2,.5,3,0],dtype=np.float32)}
        config={'k':10,'top':200,'weights':{'repeat':1.,'cf':.1,'cbf':.1,'popularity':.1}}
        actual,order=independently_fuse_row(component_rows,fallback,config)
        production=sum(weight*reciprocal_ranks(component_rows[name][None,:],fallback,200,10)
                       for name,weight in config['weights'].items())
        np.testing.assert_array_equal(actual,production[0])
        predicted=ranked_predictions(production,['u'],list(range(6)),fallback)['u']
        self.assertEqual(order,predicted)


if __name__=='__main__':
    unittest.main()
