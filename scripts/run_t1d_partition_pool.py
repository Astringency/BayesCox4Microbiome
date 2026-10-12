#!/usr/bin/env python3
"""Schedule disjoint nested-CV stages, preserving the frozen sampler and caches.

The serial T1D drivers must have stopped before this coordinator starts. This
script refuses to overlap an existing coordinator or Julia clinical worker.
After all stages pass it resumes the original clinical driver for summaries,
the full-data fit and PH diagnostics. Existing successful inner fits are reused.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--revision-out', required=True)
    parser.add_argument('--julia', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--method', default='prox-sslasso', choices=['cox-rhs', 'prox-sslasso'])
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be between 1 and 16')
    project, out = Path(args.project).resolve(), Path(args.revision_out).resolve()
    # Linux deployment guard. Observing a live worker is a reason to wait, never
    # a reason to restart it. A stale state file alone is not evidence of liveness.
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeDecodeError):
            continue
        if cmd and Path(cmd[0]).name.startswith('python') and any(Path(x).name == Path(__file__).name for x in cmd):
            if str(out) in cmd:
                raise RuntimeError(f'coordinator is already live: {proc.name}')
        if not cmd or not Path(cmd[0]).name.startswith('julia'):
            continue
        is_clinical = any(Path(x).name in ('run_t1d_clinical_revision.jl', 'run_t1d_cv_partition.jl') for x in cmd)
        if is_clinical and any(str(out) in x for x in cmd):
            raise RuntimeError(f'clinical worker is still live: {proc.name}; reuse/wait before launching this pool')
    logs = out / 'partition_logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, BCX_FROZEN_CODE=str(project), OPENBLAS_NUM_THREADS='1', GKSwstype='100')
    tasks = [(arm, stage) for arm in ('adjusted', 'unadjusted') for stage in ('2', '3', '4', '5', 'full_tuning', '1')]
    sources = {arm: out / 'T1D_clinical' / arm / args.method for arm in ('adjusted', 'unadjusted')}
    specs = {arm: json.loads((source / 'analysis_specification.json').read_text()) for arm, source in sources.items()}
    for spec in specs.values():
        assert spec['outer_folds'] == spec['inner_folds'] == 5
        assert spec['seed'] == 1 and spec['hmc_n_leapfrog'] == 100
    preserved = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for source in sources.values() for p in source.rglob('config_*_fold_*.json')
                 if json.loads(p.read_text()).get('status') == 'success'}
    state = dict(pid=os.getpid(), project=str(project), workers=args.workers, chains_per_fit=4,
                 method=args.method, preexisting_successful_inner_fits=len(preserved), status='running')
    (logs / 'pool_state.json').write_text(json.dumps(state, indent=2))
    (logs / 'preserved_inner_hashes.json').write_text(json.dumps(preserved, indent=2))

    def execute(task):
        arm, stage = task
        name = arm + '_' + stage
        command = [args.julia, '-t', '4', '--startup-file=no', '--project=' + str(project),
                   str(Path(__file__).with_name('run_t1d_cv_partition.jl')), '--source-arm', str(sources[arm]), '--stage', stage]
        (logs / (name + '.command.json')).write_text(json.dumps(command, indent=2))
        with (logs / (name + '.log')).open('a') as handle:
            result = subprocess.run(command, cwd=project, env=env, stdout=handle, stderr=subprocess.STDOUT)
        (logs / (name + '.exit')).write_text(str(result.returncode) + '\n')
        print(f'{name}: exit {result.returncode}', flush=True)
        return result.returncode

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(execute, tasks))
    unchanged = all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
                    for p, digest in preserved.items())
    state.update(preexisting_inner_files_unchanged=unchanged, stage_exit_codes=codes)
    if any(codes) or not unchanged:
        state['status'] = 'stage_failure'
        (logs / 'pool_state.json').write_text(json.dumps(state, indent=2))
        raise SystemExit('partition stage failed or an existing successful cache changed; inspect logs')

    def finalize(arm):
        final_env = dict(env, JULIA_BIN=args.julia, REVISION_OUT=str(out), METHOD=args.method,
                         ADJUSTED='true' if arm == 'adjusted' else 'false',
                         REAL_SAMPLES=str(specs[arm]['n_samples']), REAL_WARMUP=str(specs[arm]['warmup']),
                         CHAINS=str(specs[arm]['chains']))
        with (logs / ('finalize_' + arm + '.log')).open('a') as handle:
            result = subprocess.run(['bash', 'scripts/run_t1d_clinical_revision.sh'], cwd=project,
                                    env=final_env, stdout=handle, stderr=subprocess.STDOUT)
        (logs / ('finalize_' + arm + '.exit')).write_text(str(result.returncode) + '\n')
        return result.returncode

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        final_codes = list(pool.map(finalize, sources))
    state.update(status='complete' if not any(final_codes) else 'finalization_failure', finalization_exit_codes=final_codes)
    (logs / 'pool_state.json').write_text(json.dumps(state, indent=2))
    if any(final_codes):
        raise SystemExit('final clinical fit or PH step failed; inspect logs')
    print('All partition stages and clinical finalization completed; existing inner checkpoints were preserved', flush=True)


if __name__ == '__main__':
    main()
