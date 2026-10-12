#!/usr/bin/env python3
"""Start pending stages early in spare CPU slots; the original pool finalizes.

Linux stage locks prevent a later pool worker from duplicating an active stage.
Do not change the frozen project, tuning grids, seeds or cache directories.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--source-arm', required=True)
    parser.add_argument('--julia', required=True)
    parser.add_argument('--stages', default='full_tuning,1')
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('early-stage execution requires Linux stage locks')
    project, source = Path(args.project).resolve(), Path(args.source_arm).resolve()
    stages = args.stages.split(',')
    assert 1 <= len(stages) <= 2 and len(set(stages)) == len(stages)
    assert set(stages) <= {'1', '2', '3', '4', '5', 'full_tuning'}
    worker = Path(__file__).with_name('run_t1d_cv_partition.jl')
    assert 'with_t1d_stage_lock(source, stage)' in worker.read_text()
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeDecodeError):
            continue
        if not cmd or '--source-arm' not in cmd or cmd[cmd.index('--source-arm') + 1] != str(source):
            continue
        if Path(cmd[0]).name.startswith('python') and any(Path(x).name == Path(__file__).name for x in cmd):
            raise RuntimeError(f'early-stage coordinator already live: {proc.name}')
        if Path(cmd[0]).name.startswith('julia') and '--stage' in cmd and cmd[cmd.index('--stage') + 1] in stages:
            raise RuntimeError(f'requested stage already live: {proc.name}')
    spec = json.loads((source / 'analysis_specification.json').read_text())
    logs = source / 'early_stage_logs'
    logs.mkdir(exist_ok=True)
    stage_dirs = [source / 'full_data_tuning' if stage == 'full_tuning' else
                  source / 'nested_cv' / spec['method_name'] / f'outer_{int(stage):02}' for stage in stages]
    preserved = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for directory in stage_dirs
                 for path in directory.rglob('config_*_fold_*.json') if json.loads(path.read_text()).get('status') == 'success'}
    (logs / 'preserved_inner_hashes.json').write_text(json.dumps(preserved, indent=2))
    state = dict(pid=os.getpid(), status='running', stages=stages, workers=len(stages),
                 project=str(project), source=str(source), reused_successful_inner_fits=len(preserved))
    (logs / 'state.json').write_text(json.dumps(state, indent=2))

    def run(stage):
        env = dict(os.environ, BCX_FROZEN_CODE=str(project), OPENBLAS_NUM_THREADS='1', GKSwstype='100')
        cmd = [args.julia, '-t', str(spec['chains']), '--startup-file=no', '--project=' + str(project),
               str(worker), '--source-arm', str(source), '--stage', stage]
        (logs / (stage + '.command.json')).write_text(json.dumps(cmd, indent=2))
        with (logs / (stage + '.log')).open('a') as handle:
            result = subprocess.run(cmd, cwd=project, env=env, stdout=handle, stderr=subprocess.STDOUT)
        (logs / (stage + '.exit')).write_text(str(result.returncode) + '\n')
        return result.returncode

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(stages)) as pool:
        codes = list(pool.map(run, stages))
    unchanged = all(Path(path).is_file() and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                    for path, digest in preserved.items())
    state.update(status='complete' if not any(codes) and unchanged else 'failed',
                 exit_codes=codes, preexisting_inner_files_unchanged=unchanged)
    (logs / 'state.json').write_text(json.dumps(state, indent=2))
    if any(codes) or not unchanged:
        raise SystemExit('inspect early-stage logs; completed checkpoints remain available')
    print('Early stages completed; the original clinical pool will reuse them and finalize')


if __name__ == '__main__':
    main()
