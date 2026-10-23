"""Fast deterministic test suite with machine-readable execution evidence."""
import hashlib
import argparse
import json
import platform
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('artifacts/test_results.json'))
    args=parser.parse_args()
    started=time.perf_counter()
    suite=unittest.defaultTestLoader.discover('tests')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    paths=list(Path('tvrec').glob('*.py'))+list(Path('tests').glob('*.py'))
    evidence={'executed_at_utc':datetime.now(timezone.utc).isoformat(),
              'tests_run':result.testsRun,'passed':result.wasSuccessful(),
              'failures':len(result.failures),'errors':len(result.errors),
              'runtime_seconds':time.perf_counter()-started,'python':sys.version,
              'platform':platform.platform(),'executor':'AI coding assistant',
              'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    args.out.write_text(json.dumps(evidence,indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
