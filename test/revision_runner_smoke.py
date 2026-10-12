#!/usr/bin/env python3
"""Linux integration smoke: shared baseline locking, using a non-sampling stub.

Run with python3 test/revision_runner_smoke.py. No Julia process is started.
"""
import concurrent.futures
import argparse
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner', type=Path, default=project / 'scripts/run_sensitivity_revision.sh')
    args = parser.parse_args()
    # Keep this small scratch directory as an audit artifact; do not remove any
    # server content, including files created by this check.
    with contextlib.nullcontext(tempfile.mkdtemp(prefix='bayescox_runner_smoke_')) as scratch:
        root = Path(scratch)
        scripts = root / 'scripts'
        scripts.mkdir()
        shutil.copy2(args.runner, scripts)
        stub = root / 'julia_stub.py'
        stub.write_text('''#!/usr/bin/env python3
import json, sys, time
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index('--outdir') + 1])
out.mkdir(parents=True, exist_ok=True)
event_file = out / 'events.jsonl'
with event_file.open('a') as f:
    f.write(json.dumps(dict(event='start', time=time.time(), args=args)) + '\\n')
time.sleep(0.5)
done = out / 'complete.txt'
with event_file.open('a') as f:
    f.write(json.dumps(dict(event='reuse' if done.exists() else 'fit', time=time.time())) + '\\n')
done.touch()
''')
        stub.chmod(0o700)

        def run(suite, value):
            env = dict(os.environ, JULIA_BIN=str(stub), REVISION_OUT=str(root / 'output'),
                       PRIORS='cox-rhs', VALUES=value, FORCE_NEW_FITS='true')
            return subprocess.run(['bash', str(scripts / 'run_sensitivity_revision.sh'), suite],
                                  env=env, check=True, capture_output=True, text=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            runs = [pool.submit(run, 'lambda', '1e-3'), pool.submit(run, 'p0', '6')]
            for result in runs:
                result.result()
        outputs = list((root / 'output').rglob('events.jsonl'))
        assert len(outputs) == 1, outputs
        assert outputs[0].parent == root / 'output/lambda/cox-rhs/1e-3'
        events = [json.loads(line) for line in outputs[0].read_text().splitlines()]
        assert [row['event'] for row in events] == ['start', 'fit', 'start', 'reuse'], events
        for row in (events[0], events[2]):
            args = row['args']
            for key, value in {'--reps': '50', '--chains': '4', '--warmup': '2000', '--samples': '5000',
                               '--n': '400', '--p': '50', '--nonzero': '6', '--p0': '6',
                               '--prox-sum0-lambda': '1e-3', '--baseline': 'weibull'}.items():
                assert args[args.index(key) + 1] == value, (key, args)
        print('PASS: concurrent lambda/p0 launchers share and serialize one baseline; formal flags preserved; no posterior sampled')
        print('Audit scratch directory:', scratch)


if __name__ == '__main__':
    main()
