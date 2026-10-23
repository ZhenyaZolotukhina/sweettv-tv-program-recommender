"""Independently recheck saved output against the original official CSV template."""
import argparse
import csv
import hashlib
from pathlib import Path

import duckdb

from .core import validate_submission
from .pipeline import candidate_set, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('sweettv-tv-program-recommender'))
    parser.add_argument('--cache',type=Path,default=Path('cache'))
    parser.add_argument('--file',type=Path,default=Path('artifacts/private/preliminary_submission.csv'))
    args=parser.parse_args()
    with (args.data/'submission.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['user_id','tv_show_id']
        users=[r['user_id'] for r in reader]
    with args.file.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['user_id','tv_show_id']
        rows=[(r['user_id'],r['tv_show_id']) for r in reader]
    c=duckdb.connect(str(args.cache/'data.duckdb'),read_only=True)
    candidates=candidate_set(c,31)
    validate_submission(users,rows,candidates)
    c.close()
    result={'valid':True,'rows':len(rows),'five_distinct_positive_available_ids_per_row':True,
            'original_template_user_order_preserved':True,'test_candidates':len(candidates),
            'sha256':hashlib.sha256(args.file.read_bytes()).hexdigest()}
    save(args.file.parent.parent/'submission_independent_check.json',result)
    print(result)


if __name__=='__main__':
    main()
