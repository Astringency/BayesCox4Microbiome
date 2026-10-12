#!/usr/bin/env python3
"""Advance the last RHS zero-sum job after CPU slots become available.

Only the Python dispatcher is suspended; its existing sampler children continue.
After the advanced job finishes, the dispatcher resumes and reuses those fits.
Signals target a verified PID/start-time/command identity, never a process group
belonging to the existing jobs. Frozen model sources are unchanged.
"""
import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def process_info(pid):
    directory = Path('/proc') / str(pid)
    try:
        fields = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
        command = (directory / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return dict(pid=int(pid), start_ticks=int(fields[19]), command=command,
                    state=fields[0], cpu_ticks=int(fields[11]) + int(fields[12]))
    except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError):
        return None


def checked_signal(expected, sig):
    try:
        descriptor = os.pidfd_open(expected['pid'])
    except ProcessLookupError:
        return False
    try:
        current = process_info(expected['pid'])
        if current is None or any(current[key] != expected[key] for key in ('start_ticks', 'command')):
            return False
        signal.pidfd_send_signal(descriptor, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(descriptor)


def live_samplers():
    result = {}
    for directory in Path('/proc').glob('[0-9]*'):
        info = process_info(int(directory.name))
        if info and Path(info['command'][0]).name.startswith('julia') and '--outdir' in info['command']:
            output = info['command'][info['command'].index('--outdir') + 1]
            result.setdefault(output, []).append(info)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision-root', required=True)
    parser.add_argument('--coordinator-pid', type=int, required=True)
    parser.add_argument('--julia', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(args.revision_root).resolve()
    logs, project = root / 'logs', root / 'code_v5'
    queue = root / 'results/rhs_kernel_followup_logs'
    output = root / 'results/fixed_parallel/zero_sum/prox1e-6/cox-rhs'
    parent = process_info(args.coordinator_pid)
    assert parent and parent['state'] not in ('T', 't', 'Z'), 'Dispatcher must be live and running'
    descriptor = os.pidfd_open(parent['pid'])
    os.close(descriptor)
    command = parent['command']
    assert Path(command[0]).name.startswith('python')
    assert any(Path(x).name == 'run_rhs_kernel_followup.py' for x in command)
    assert command[command.index('--revision-root') + 1] == str(root)
    assert not (queue / 'zero_prox1e-6.command.json').exists(), 'Queued job already started'
    assert all((queue / (name + '.exit')).read_text().strip() == '0'
               for name in ('baseline_shard1', 'baseline_shard2'))
    workers = live_samplers()
    assert str(output) not in workers, 'Do not overlap an existing sampler'
    existing = []
    for constraint in ('none', 'soft_center', 'prox1e-3'):
        key = str(root / 'results/fixed_parallel/zero_sum' / constraint / 'cox-rhs')
        assert len(workers.get(key, [])) == 1
        existing.extend(workers[key])
    assert os.getloadavg()[0] + 4 < os.cpu_count(), 'Wait for spare capacity'
    completed = list(output.rglob('complete.txt'))
    assert completed and len(completed) < 50
    preserved = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                 for marker in completed for path in
                 (marker, marker.parent / 'metadata.json', marker.parent / 'samples_wide.csv')}
    overrides = dict(JULIA_BIN=args.julia, REVISION_OUT=str(root / 'results/fixed_parallel'),
                     PRIORS='cox-rhs', CONSTRAINTS='prox', SUM0_LAMBDA='1e-6', REPS='50',
                     CHAINS='4', SAMPLES='5000', WARMUP='2000', N='400', P='150', NONZERO='12')
    state = dict(status='preflight_passed', coordinator=parent, existing_sampler_processes=existing,
                 project=str(project), command=['bash', 'scripts/run_zero_sum_revision.sh'],
                 env_overrides=overrides, preserved_fit_hashes=preserved,
                 n_preexisting_repetitions=len(completed))
    if args.dry_run:
        print(json.dumps(state, indent=2))
        return
    import fcntl
    lock = (logs / 'advance_rhs_zero_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_file = logs / 'advance_rhs_zero_queue.state.json'

    def save():
        state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary = state_file.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_file)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'interrupted by signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    suspended, worker = False, None
    save()
    try:
        suspended = True
        assert checked_signal(parent, signal.SIGSTOP)
        for _ in range(100):
            if process_info(parent['pid'])['state'] in ('T', 't'):
                break
            time.sleep(.05)
        assert process_info(parent['pid'])['state'] in ('T', 't')
        assert not (queue / 'zero_prox1e-6.command.json').exists()
        for expected in existing:
            current = process_info(expected['pid'])
            assert current and current['start_ticks'] == expected['start_ticks']
            assert current['state'] not in ('T', 't', 'Z'), 'Existing sampler must continue'
        with (logs / 'advance_rhs_zero_queue.sampling.log').open('a') as handle:
            worker = subprocess.Popen(state['command'], cwd=project, env=dict(os.environ, **overrides),
                                      stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='sampling_queued_cell', worker_pid=worker.pid,
                         dispatcher_suspended=True, existing_samplers_left_running=True)
            save()
            code = worker.wait()
        assert code == 0, f'Advanced job exited {code}; original queue will resume'
        assert len(list(output.rglob('complete.txt'))) == 50
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                   for path, digest in preserved.items()), 'An existing fit changed'
        state.update(status='complete', n_completed_repetitions=50, preexisting_fits_unchanged=True)
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        try:
            if worker is not None and worker.poll() is None:
                # This process group was created above; it contains only this new job.
                try:
                    os.killpg(worker.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
        finally:
            if suspended:
                resumed = checked_signal(parent, signal.SIGCONT)
                state.update(coordinator_resume_signal_sent=resumed, dispatcher_suspended=False)
                if not resumed:
                    state.update(status='failed', error='Dispatcher identity changed before resume')
            save()
    if state['status'] != 'complete':
        raise SystemExit(state.get('error', 'Advanced job did not complete'))
    print('Advanced job complete; original dispatcher resumed and will reuse its 50 fits')


if __name__ == '__main__':
    main()
