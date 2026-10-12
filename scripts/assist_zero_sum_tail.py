#!/usr/bin/env python3
"""Prepare one unstarted future repetition for a live zero-sum simulation.

Use the live worker's frozen project and exact arguments, changing only the
output directory and repetition subset. Existing results are never overwritten.
Publication briefly suspends the verified original worker to make its cache
check and the atomic publication mutually exclusive; sampling itself runs in
an isolated staging directory. The pause duration is recorded for runtime audit.
Requires Linux /proc, pidfd, and renameat2(RENAME_NOREPLACE).
"""
import argparse
import csv
import ctypes
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from advance_rhs_zero_queue import checked_signal, process_info


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def rename_no_replace(source, destination):
    function = ctypes.CDLL(None, use_errno=True).renameat2
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def same_process(expected):
    current = process_info(expected['pid'])
    return current is not None and all(current[key] == expected[key] for key in ('command', 'start_ticks'))


def running_process(pid):
    # Another helper may be completing a millisecond-scale publication pause.
    # Wait for its own SIGCONT; never resume a process owned by another helper.
    for _ in range(400):
        current = process_info(pid)
        if current is None or current['state'] not in ('T', 't'):
            return current
        time.sleep(.005)
    raise RuntimeError('Source worker remains paused; leave it to its existing owner')


def publish(expected, prepared, destination, started_file):
    """Pause only for the final guard and atomic rename, never for copying."""
    assert prepared.is_dir() and not destination.exists()
    suspended = False
    start = time.monotonic()
    published = False
    try:
        current = running_process(expected['pid'])
        assert current and current['state'] not in ('T', 't', 'Z') and same_process(expected)
        start = time.monotonic()
        suspended = True
        assert checked_signal(expected, signal.SIGSTOP), 'Original worker identity changed'
        for _ in range(400):
            task_states = [(p / 'stat').read_text().rsplit(') ', 1)[1].split()[0]
                           for p in (Path('/proc') / str(expected['pid']) / 'task').glob('*')]
            if task_states and all(state in ('T', 't') for state in task_states):
                break
            time.sleep(.005)
        else:
            raise RuntimeError('Original worker did not enter a bounded publication pause')
        if not started_file.exists() and not destination.exists():
            rename_no_replace(prepared, destination)
            published = True
    finally:
        if suspended:
            assert checked_signal(expected, signal.SIGCONT), 'Could not resume the original worker'
    return dict(published=published, original_resume_signal_sent=True,
                publication_pause_seconds=time.monotonic() - start)


def validate_draws(run, replicate, method, constraint):
    assert (run / 'complete.txt').is_file()
    meta = json.loads((run / 'metadata.json').read_text())
    assert (meta['n'], meta['p'], meta['nonzero_count'], meta['baseline']) == (400, 150, 12, 'gompertz')
    assert (meta['chains'], meta['n_samples'], meta['warmup'], meta['saved_sample_thin']) == (4, 5000, 2000, 1)
    assert meta['replicate_id'] == replicate and meta['method_name'] == method and meta['constraint'] == constraint
    assert meta['seed'] == replicate and meta['n_test'] == 1000
    assert Path(meta['test_data']).is_file()
    counts = {str(i): 0 for i in range(1, 5)}
    with (run / 'samples_wide.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            chain = row['chain']
            counts[chain] += 1
            assert int(row['iter']) == counts[chain]
            assert all(math.isfinite(float(row[f'beta[{j}]'])) for j in range(1, 151))
            assert math.isfinite(float(row['logposterior']))
    assert all(n == 5000 for n in counts.values())
    return {str(path.relative_to(run)): digest(path) for path in run.rglob('*') if path.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision-root', required=True, type=Path)
    parser.add_argument('--source-pid', required=True, type=int)
    parser.add_argument('--replicate', required=True, type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.revision_root.resolve()
    original = running_process(args.source_pid)
    assert original and original['state'] not in ('T', 't', 'Z')
    command = original['command']
    assert Path(command[0]).name == 'julia' and 'scripts/run_simulation.jl' in command

    def option(name):
        assert command.count(name) == 1, name
        return command[command.index(name) + 1]

    for key, expected in {'--reps': '50', '--n': '400', '--p': '150', '--nonzero': '12',
                          '--baseline': 'gompertz', '--chains': '4', '--samples': '5000',
                          '--warmup': '2000', '--seed': '1', '--resume': 'true',
                          '--sample-thin': '1', '--first-rep-sample-thin': '1'}.items():
        assert option(key) == expected, key
    assert '--rep-ids' not in command
    output = Path(option('--outdir')).resolve()
    assert output.is_relative_to(root / 'results') and output.parent.parent.name == 'zero_sum'
    method, constraint = option('--priors'), option('--constraint')
    assert method in ('cox-rhs', 'prox-sslasso') and output.name == method
    project = (Path('/proc') / str(args.source_pid) / 'cwd').resolve()
    assert project.parent == root and project.name.startswith('code_v')
    assert '"rep-ids"' in (project / 'scripts/run_simulation.jl').read_text()
    for name, expected in json.loads((project / 'source_manifest.json').read_text()).items():
        assert digest(project / name) == expected, name
    assert 1 <= args.replicate <= 50
    tag = f'{method}_{output.parent.name}_rep{args.replicate:02}'
    label = f'rep{args.replicate}_{method}_gompertz_n400_p150_nz12'
    destination = output / 'runs' / label
    started_file = output / 'test_data' / f'rep{args.replicate}_gompertz_n400_p150_nz12.csv'
    started = [int(re.match(r'rep(\d+)_', p.name).group(1)) for p in (output / 'test_data').glob('rep*_gompertz_n400_p150_nz12.csv')]
    assert started and args.replicate >= max(started) + 6, 'Use a distant, unstarted repetition'
    assert not destination.exists() and not started_file.exists(), 'Original worker already started this repetition'
    assert os.getloadavg()[0] + 4 < .85 * os.cpu_count(), 'Wait for spare CPU capacity'
    stage = root / 'zero_tail_assist_staging' / tag
    assert not stage.exists(), 'Existing staging result requires inspection and reuse, not resampling'
    new_command = list(command)
    new_command[new_command.index('--outdir') + 1] = str(stage)
    new_command.extend(['--rep-ids', str(args.replicate)])
    state = dict(status='preflight_passed', original=original, project=str(project),
                 command=new_command, stage=str(stage), destination=str(destination),
                 original_frontier=max(started), replicate=args.replicate,
                 helper_sha256=digest(Path(__file__).resolve()),
                 frozen_manifest_sha256=digest(project / 'source_manifest.json'))
    if args.dry_run:
        print(json.dumps(state, indent=2))
        return
    import fcntl
    logs = root / 'logs/zero_tail_assist'
    logs.mkdir(parents=True, exist_ok=True)
    lock = (logs / (tag + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not stage.exists() and not destination.exists() and not started_file.exists()
    state_file = logs / (tag + '.state.json')

    def save():
        state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary = state_file.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_file)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'interrupted by signal {signum}')

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    worker = None
    save()
    try:
        with (logs / (tag + '.log')).open('x') as log:
            worker = subprocess.Popen(new_command, cwd=project,
                                      env=dict(os.environ, OPENBLAS_NUM_THREADS='1', GKSwstype='100'),
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='sampling_future_repetition', worker_pid=worker.pid)
            save()
            while worker.poll() is None:
                if started_file.exists() or not same_process(original):
                    state.update(status='cancelled_original_reached_repetition')
                    return
                time.sleep(.5)
        assert worker.returncode == 0, f'Future worker exited {worker.returncode}'
        run = stage / 'runs' / label
        hashes = validate_draws(run, args.replicate, method, constraint)
        prepared = root / 'zero_tail_assist_publications' / tag
        assert not prepared.exists()
        shutil.copytree(run, prepared)
        assert all(digest(prepared / name) == expected for name, expected in hashes.items())
        state.update(status='ready_to_publish', fit_hashes=hashes)
        save()
        # Different assisted repetitions can finish together for the same source.
        # Serialize their short publication pauses so none resumes another's lock.
        with (logs / f'{method}_{output.parent.name}.publication.lock').open('a') as publication_lock:
            fcntl.flock(publication_lock, fcntl.LOCK_EX)
            outcome = publish(original, prepared, destination, started_file)
        state.update(outcome, status='complete' if outcome['published'] else 'cancelled_original_reached_repetition')
        if outcome['published']:
            assert all(digest(destination / name) == expected for name, expected in hashes.items())
            state['published_hashes_match'] = True
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        if worker is not None and worker.poll() is None:
            # This process group contains only the new staging worker we created.
            try:
                os.killpg(worker.pid, signal.SIGTERM)
                worker.wait(timeout=30)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
        save()
    print(json.dumps({k: state.get(k) for k in ('status', 'destination', 'published_hashes_match', 'publication_pause_seconds')}))


if __name__ == '__main__':
    main()
