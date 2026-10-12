#!/usr/bin/env python3
"""Use spare cores for distant missing repetitions of existing zero-sum jobs.

The original jobs keep their frozen code and continue to run. Each helper uses
isolated output, guards against the original reaching its repetition, and only
publishes into an absent run directory. Source projects without --rep-ids remain
on their original queue. This coordinator does not restart any existing job.
"""
import argparse
import datetime
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from assist_zero_sum_tail import same_process
from advance_rhs_zero_queue import process_info


def frontier(output):
    values = [int(re.match(r'rep(\d+)_', p.name).group(1))
              for p in (output / 'test_data').glob('rep*_gompertz_n400_p150_nz12.csv')]
    return max(values) if values else 0


def candidates(root, source, attempted):
    if not same_process(source['process']):
        return []
    output, method, setting = Path(source['output']), source['method'], source['setting']
    result = []
    for rep in range(50, frontier(output) + 5, -1):
        tag = f'{method}_{setting}_rep{rep:02}'
        run = output / 'runs' / f'rep{rep}_{method}_gompertz_n400_p150_nz12'
        if (tag not in attempted and not run.exists()
                and not (root / 'zero_tail_assist_staging' / tag).exists()
                and not (root / 'logs/zero_tail_assist' / (tag + '.state.json')).exists()):
            result.append((rep, tag))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision-root', type=Path, required=True)
    parser.add_argument('--source-pids', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.workers <= 8
    root = args.revision_root.resolve()
    pilot = json.loads((root / 'logs/zero_tail_pilot.verified.json').read_text())
    assert pilot['all_checks_passed'] and pilot['existing_fits_unchanged'] and pilot['resume_without_sampling_verified']
    sources, unsupported = [], []
    for pid in map(int, args.source_pids.split(',')):
        process = process_info(pid)
        assert process and process['state'] not in ('T', 't', 'Z'), pid
        command = process['command']
        assert Path(command[0]).name == 'julia' and 'scripts/run_simulation.jl' in command
        output = Path(command[command.index('--outdir') + 1]).resolve()
        assert output.is_relative_to(root / 'results') and output.parent.parent.name == 'zero_sum'
        project = (Path('/proc') / str(pid) / 'cwd').resolve()
        assert project.parent == root and project.name.startswith('code_v')
        if '"rep-ids"' not in (project / 'scripts/run_simulation.jl').read_text():
            unsupported.append(dict(pid=pid, output=str(output), reason='Frozen source has no repetition-subset option'))
            continue
        sources.append(dict(process=process, output=str(output), method=output.name, setting=output.parent.name))
    assert sources and len({s['output'] for s in sources}) == len(sources)
    if args.dry_run:
        print(json.dumps(dict(sources=[dict(pid=s['process']['pid'], output=s['output'],
                                           available_repetitions=[r for r, _ in candidates(root, s, set())]) for s in sources],
                              unsupported=unsupported, workers=args.workers), indent=2))
        return
    import fcntl
    logs = root / 'logs/zero_tail_assist'
    logs.mkdir(parents=True, exist_ok=True)
    lock = (logs / 'pool.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_file = logs / 'pool_state.json'
    assert not state_file.exists(), 'Inspect and reuse an existing pool rather than restarting it'
    state = dict(pid=os.getpid(), status='running', workers=args.workers, sources=sources,
                 unsupported=unsupported, completed=[], failures=[])
    active, attempted = {}, set()
    stop_requested = False

    def interrupted(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)

    def save():
        state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        state['active'] = [dict(tag=tag, pid=item['process'].pid, source_pid=item['source']['process']['pid'],
                                replicate=item['replicate']) for tag, item in active.items()]
        temporary = state_file.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_file)

    def options():
        available = []
        for source in sources:
            repetitions = candidates(root, source, attempted)
            if repetitions:
                n_active = sum(item['source'] is source for item in active.values())
                remaining = 50 - len(list(Path(source['output']).rglob('complete.txt')))
                available.append((remaining / (1 + n_active), source, repetitions[0]))
        return available

    save()
    try:
        while True:
            for tag, item in list(active.items()):
                code = item['process'].poll()
                if code is None:
                    continue
                item['log'].close()
                detail = logs / (tag + '.state.json')
                result = json.loads(detail.read_text()) if detail.exists() else {}
                record = dict(tag=tag, exit_code=code, status=result.get('status'), source_pid=item['source']['process']['pid'])
                if code == 0 and result.get('status') in ('complete', 'cancelled_original_reached_repetition'):
                    state['completed'].append(record)
                elif not (root / 'zero_tail_assist_staging' / tag).exists() and (
                        frontier(Path(item['source']['output'])) + 6 > item['replicate']
                        or not same_process(item['source']['process'])):
                    # The source advanced after selection but before helper preflight;
                    # no staging directory means that no posterior fit was launched.
                    record['status'] = 'not_started_original_advanced'
                    state['completed'].append(record)
                else:
                    state['failures'].append(record)
                del active[tag]
                print(json.dumps(record), flush=True)
            if stop_requested:
                for item in active.values():
                    if not item.get('termination_sent'):
                        item['process'].terminate()
                        item['termination_sent'] = True
            if not stop_requested and not state['failures']:
                while len(active) < args.workers and os.getloadavg()[0] + 4 < .85 * os.cpu_count():
                    available = options()
                    if not available:
                        break
                    _, source, (rep, tag) = max(available, key=lambda entry: entry[0])
                    attempted.add(tag)
                    command = ['/usr/bin/python3', str(Path(__file__).with_name('assist_zero_sum_tail.py')),
                               '--revision-root', str(root), '--source-pid', str(source['process']['pid']),
                               '--replicate', str(rep)]
                    (logs / (tag + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
                    log = (logs / (tag + '.launcher.log')).open('x')
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    active[tag] = dict(process=process, log=log, source=source, replicate=rep)
                    print('started', tag, process.pid, flush=True)
            save()
            if not active and (stop_requested or state['failures'] or not options()):
                break
            time.sleep(1)
        state['status'] = 'interrupted' if stop_requested else 'failed' if state['failures'] else 'complete'
        save()
        if state['status'] != 'complete':
            raise SystemExit('Inspect pool state and per-repetition logs; existing samplers are left running')
    finally:
        save()
    print('Tail assistance finished; original workers complete their remaining repetitions and summaries', flush=True)


if __name__ == '__main__':
    main()
