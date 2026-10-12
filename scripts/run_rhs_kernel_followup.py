#!/usr/bin/env python3
"""Use CPU slots freed by five existing RHS jobs for only required corrections.

The two already-corrected baseline repetitions are retained. Historical files
remain untouched. Each shard/constraint has a separate output root, preventing
concurrent table writers. No speculative repeat-to-pass convergence selection.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--revision-root', required=True)
    p.add_argument('--julia', required=True)
    p.add_argument('--workers', type=int, default=5)
    args = p.parse_args()
    if not 1 <= args.workers <= 5:
        p.error('--workers must be between 1 and 5')
    root = Path(args.revision_root).resolve()
    project, out = root / 'code_v5', root / 'results'
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeDecodeError):
            continue
        if cmd and Path(cmd[0]).name.startswith('python') and str(root) in cmd and any(Path(x).name == Path(__file__).name for x in cmd):
            raise RuntimeError(f'followup coordinator already live: {proc.name}')
        if cmd and Path(cmd[0]).name.startswith('julia'):
            owned_outputs = ['fixed_parallel/baseline_shards/'] + ['fixed_parallel/zero_sum/' + c + '/cox-rhs' for c in ('none', 'soft_center', 'prox1e-3', 'prox1e-6')]
            if any(str(out / suffix) in x for suffix in owned_outputs for x in cmd):
                raise RuntimeError(f'followup worker still live: {proc.name}; wait before starting another coordinator')
    log = out / 'rhs_kernel_followup_logs'
    log.mkdir(parents=True, exist_ok=True)
    dependencies = ['rhs_sensitivity_1e-2', 'rhs_sensitivity_1e-4', 'rhs_sensitivity_1e-5', 'p0_3', 'p0_12']
    while True:
        live = [name for name in dependencies if subprocess.run(['tmux', 'has-session', '-t', 'bcx_' + name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0]
        if not live:
            break
        (log / 'queue_status.json').write_text(json.dumps(dict(status='waiting_for_existing_jobs', live_dependencies=live), indent=2))
        time.sleep(15)
    for name in dependencies:
        assert (out / 'job_logs' / (name + '.exit')).read_text().strip() == '0', name
    remaining = [i for i in range(1, 51) if i not in (33, 50)]
    jobs = []
    for shard, reps in enumerate((remaining[:24], remaining[24:]), 1):
        jobs.append((f'baseline_shard{shard}', 'scripts/run_sensitivity_revision.sh', ['lambda'],
            dict(REVISION_OUT=str(out / 'fixed_parallel/baseline_shards' / f'shard{shard}'),
                 PRIORS='cox-rhs', VALUES='1e-3', FORCE_NEW_FITS='true', REP_IDS=','.join(map(str, reps)))))
    for constraint, lam in [('none', '1e-3'), ('soft_center', '1e-3'), ('prox', '1e-3'), ('prox', '1e-6')]:
        name = 'zero_' + (constraint if constraint != 'prox' else 'prox' + lam)
        jobs.append((name, 'scripts/run_zero_sum_revision.sh', [],
            dict(REVISION_OUT=str(out / 'fixed_parallel'), PRIORS='cox-rhs', CONSTRAINTS=constraint, SUM0_LAMBDA=lam)))
    (log / 'queue_status.json').write_text(json.dumps(dict(status='running', jobs=[j[0] for j in jobs], workers=args.workers,
        baseline_repetitions_reused=[33, 50], baseline_repetitions_to_fit=remaining), indent=2))
    baseline_done = set()
    finish_lock = threading.Lock()

    def run(job):
        name, script, script_args, overrides = job
        env = dict(os.environ, JULIA_BIN=args.julia, HISTORICAL_ROOT='/large_storage/zhangxf/BayesCox/results', **overrides)
        command = ['bash', script] + script_args
        (log / (name + '.command.json')).write_text(json.dumps(dict(project=str(project), command=command, env_overrides=overrides), indent=2))
        with (log / (name + '.log')).open('a') as handle:
            result = subprocess.run(command, cwd=project, env=env, stdout=handle, stderr=subprocess.STDOUT)
        (log / (name + '.exit')).write_text(str(result.returncode) + '\n')
        print(name, result.returncode, flush=True)
        if name.startswith('baseline_') and result.returncode == 0:
            with finish_lock:
                baseline_done.add(name)
                if len(baseline_done) == 2:
                    subprocess.run(['python3', str(Path(__file__).with_name('resolve_revision_fixed_baseline.py')),
                                    '--results-root', str(out)], check=True)
        return result.returncode

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run, jobs))
    (log / 'queue_status.json').write_text(json.dumps(dict(status='complete' if not any(codes) else 'failed',
        jobs=[j[0] for j in jobs], exit_codes=codes), indent=2))
    if any(codes):
        raise SystemExit('inspect failed followup job logs; original files are preserved')


if __name__ == '__main__':
    main()
