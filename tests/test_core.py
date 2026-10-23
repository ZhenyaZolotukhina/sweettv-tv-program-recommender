import tempfile
import unittest
from pathlib import Path

from tvrec.core import (ap_at_5, map_at_5, disjoint_airings, overlap_seconds,
                        match_disjoint, window_overlap, validate_submission, write_submission)


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(ap_at_5([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1)

    def test_positions(self):
        self.assertEqual(ap_at_5([1], [1, 2, 3, 4, 5]), 1)
        self.assertEqual(ap_at_5([1], [2, 3, 4, 5, 1]), .2)

    def test_duplicates(self):
        self.assertAlmostEqual(ap_at_5([1, 2], [1, 1, 2, 2, 3]), (1 + 2/3)/2)

    def test_empty_and_more_than_five(self):
        self.assertEqual(ap_at_5([], [1]), 0)
        self.assertEqual(ap_at_5(range(10), list(range(5))), 1)
        self.assertEqual(map_at_5({'a': {1}}, {'a': [1], 'b': [2]}, ['a', 'b']), .5)
        self.assertIsNone(map_at_5({}, {}, []))


class IntervalTests(unittest.TestCase):
    def test_spanning_midnight_and_multiple(self):
        self.assertEqual(overlap_seconds(23*3600, 26*3600, 24*3600, 25*3600), 3600)
        self.assertEqual(overlap_seconds(1, 2, 2, 3), 0)

    def test_conflicts_gaps_duplicates_missing_ids(self):
        # Source IDs are arbitrary; no preference for labelled rows.
        self.assertEqual(disjoint_airings([(0, 10, 1), (3, 5, 2), (3, 5, 3),
                                           (12, 15, 4), (16, 16, 5)]),
                         [(0, 3, 1), (3, 5, 2), (5, 10, 1), (12, 15, 4)])

    def test_leakage(self):
        self.assertEqual(window_overlap(0, 20, 0, 10), (0, 10))
        self.assertEqual(window_overlap(0, 20, 10, 30), (10, 20))
        a, b = window_overlap(0, 20, 0, 10)
        self.assertEqual(overlap_seconds(a, b, 10, 20), 0)

    def test_production_binary_search_join(self):
        vi, si, first, last = match_disjoint([1,5,8,20], [9,5,15,19],
                                             [0,4,10], [3,7,12], 0, 11)
        self.assertEqual(list(zip(vi,si,first,last)), [(0,0,1,3),(0,1,4,7),(2,2,10,11)])
        self.assertEqual(len(match_disjoint([0],[1],[],[],0,5)[0]),0)

    def test_random_join_against_bruteforce(self):
        import random
        random.seed(42)
        for _ in range(50):
            intervals=disjoint_airings([(a,a+random.randint(1,20),i)
                         for i,a in enumerate(random.sample(range(100),20))])
            starts=[a for a,b,i in intervals]
            ends=[b for a,b,i in intervals]
            vs=[random.randint(-10,110) for _ in range(30)]
            ve=[a+random.randint(0,50) for a in vs]
            vi,si,first,last=match_disjoint(vs,ve,starts,ends,0,100)
            got=list(zip(vi,si,last-first))
            expected=[(i,j,overlap_seconds(max(a,0),min(b,100),s,t))
                      for i,(a,b) in enumerate(zip(vs,ve)) for j,(s,t) in enumerate(zip(starts,ends))
                      if overlap_seconds(max(a,0),min(b,100),s,t)>0]
            self.assertEqual(got,expected)


class FeatureTests(unittest.TestCase):
    def test_strength_variants_and_future_behavior_excluded(self):
        import duckdb
        from tvrec.pipeline import features, SPLIT
        c=duckdb.connect()
        c.execute('CREATE TABLE interactions(user_id VARCHAR,item BIGINT,seconds BIGINT,fraction DOUBLE,last_second BIGINT,phase VARCHAR)')
        c.executemany('INSERT INTO interactions VALUES(?,?,?,?,?,?)',
                      [('u',1,120,.02,SPLIT-10,'train'),
                       ('u',2,90,.5,SPLIT-5,'train'),
                       ('u',3,999999,1,SPLIT+100,'validation')])
        absolute=features(c,'seconds',100,0)
        fraction=features(c,'fraction',.1,0)
        self.assertEqual(absolute.item.tolist(),[1])
        self.assertEqual(absolute.score.tolist(),[120])
        self.assertEqual(fraction.item.tolist(),[2])
        self.assertEqual(fraction.score.tolist(),[.5])
        self.assertEqual(set(features(c,'seconds',60,0,final=True).item),{1,2,3})
        c.close()

    def test_common_fallback_repeat_ranking_and_availability(self):
        import pandas as pd
        from tvrec.pipeline import rank_baseline
        frame=pd.DataFrame({'user_id':['warm','warm'],'item':[2,999],
                            'score':[2.,1000.],'frequency':[3,100],
                            'last_second':[50,100]})
        candidates={1,2,3,4,5,6}
        fallback=[6,5,4,3,2,1]
        for method in ['repeat','repeat_popularity']:
            predictions=rank_baseline(frame,['warm','cold'],candidates,method,
                                      signal='frequency',fallback=fallback)
            self.assertEqual(predictions['cold'],fallback[:5])
            self.assertNotIn(999,predictions['warm'])
            self.assertEqual(len(set(predictions['warm'])),5)
        predictions=rank_baseline(frame,['warm'],candidates,'repeat',
                                  signal='frequency',fallback=fallback)
        self.assertEqual(predictions['warm'],[2,6,5,4,3])


class SubmissionTests(unittest.TestCase):
    def test_lossless_and_roundtrip(self):
        users = ['18446744073709551615', '0000123']  # uint64 maximum, synthetic boundary
        with tempfile.TemporaryDirectory() as d:
            write_submission(Path(d)/'out.csv', users, {u: [1,2,3,4,5] for u in users}, set(range(1,6)))
        for bad in ['1 1 2 3 4', '0 1 2 3 4', '1 2 3 4 NaN', '1 2 3 4 6', '1  2 3 4 5']:
            with self.assertRaises(ValueError):
                validate_submission(['u'], [('u', bad)], set(range(1,6)))
        with self.assertRaises(ValueError):
            validate_submission(['18446744073709551615'], [('18446744073709552000', '1 2 3 4 5')], set(range(1,6)))


if __name__ == '__main__':
    unittest.main()
