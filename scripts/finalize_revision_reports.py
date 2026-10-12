#!/usr/bin/env python3
"""Wait for the deployed revision jobs, then refresh and archive saved results.

This coordinator never starts a sampler or restarts a failed job. Both terminal
tmux sessions and successful exit records are required before final reporting.
"""
import argparse
import csv
import datetime
import json
import os
import subprocess
import time
from pathlib import Path


JOBS = {
    'bayescox_exact_rhs_fixed': 'logs/exact_rhs_fixed.exit',
    'bcx_rhs_kernel_followup': 'logs/rhs_kernel_followup.exit',
    'bcx_ssl_exact_corrected': 'results/corrected_ssl/job_logs/ssl_exact_corrected.exit',
    'bcx_ssl_zero_none': 'results/corrected_ssl/job_logs/ssl_zero_none.exit',
    'bcx_ssl_zero_soft_center': 'results/corrected_ssl/job_logs/ssl_zero_soft_center.exit',
    'bcx_ssl_zero_prox': 'results/corrected_ssl/job_logs/ssl_zero_prox.exit',
    'bcx_ssl_zero_prox1e-6': 'results/corrected_ssl/job_logs/ssl_zero_prox1e-6.exit',
    'bcx_t1d_partition_pool': 'logs/t1d_partition_pool.exit',
}


def job_status(root):
    records = []
    jobs = dict(JOBS)
    if (root / 'logs/advance_rhs_zero_queue.command.json').is_file():
        jobs['bcx_rhs_zero_advanced'] = 'logs/advance_rhs_zero_queue.exit'
    if (root / 'logs/zero_tail_pilot.command.json').is_file():
        jobs['bcx_zero_tail_pilot'] = 'logs/zero_tail_pilot.exit'
    if (root / 'logs/zero_tail_pool.command.json').is_file():
        jobs['bcx_zero_tail_pool'] = 'logs/zero_tail_pool.exit'
    for session, filename in jobs.items():
        live = subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True)
        if live.returncode not in (0, 1):
            raise RuntimeError(f'Cannot inspect tmux session {session}: {live.stderr.decode()}')
        path = root / filename
        records.append(dict(session=session, live=live.returncode == 0,
                            exit_code=path.read_text().strip() if path.exists() else None,
                            exit_file=str(path)))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision-root', required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--status-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.revision_root).resolve()
    if args.status_only:
        print(json.dumps(job_status(root), indent=2))
        return
    if not 1 <= args.poll_seconds <= 60:
        parser.error('--poll-seconds must be between 1 and 60')
    import fcntl  # The deployed job coordinator runs on Linux.
    logs, utilities = root / 'logs', Path(__file__).resolve().parent
    logs.mkdir(parents=True, exist_ok=True)
    lock = (logs / 'final_reports.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_file = logs / 'final_reports_state.json'
    state = dict(pid=os.getpid(), posterior_sampling_performed=False)

    def save(status, **fields):
        state.update(status=status, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields)
        temporary = state_file.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_file)

    try:
        while True:
            jobs = job_status(root)
            failed = [job for job in jobs if not job['live'] and job['exit_code'] != '0']
            if failed:
                raise RuntimeError('A terminal job has no successful exit record: ' +
                                   ', '.join(job['session'] for job in failed))
            save('waiting_for_jobs', jobs=jobs)
            if not any(job['live'] for job in jobs):
                break
            time.sleep(args.poll_seconds)
        save('refreshing_saved_results')
        exit_file = logs / 'refresh_reports.exit'
        exit_file.write_text('running\n')
        with (logs / 'refresh_reports.log').open('w') as handle:
            result = subprocess.run(['bash', str(utilities / 'refresh_revision_reports.sh'), str(root)],
                                    stdout=handle, stderr=subprocess.STDOUT)
        exit_file.write_text(str(result.returncode) + '\n')
        if result.returncode:
            raise RuntimeError('Read-only report refresh failed; inspect refresh_reports.log')
        output = root / 'results'
        with (output / 'performance_aggregates/requested_progress.csv').open() as handle:
            progress = list(csv.DictReader(handle))
        assert len(progress) == 31 and all(int(row['n_complete']) == 50 for row in progress), \
            'Some requested simulation cells are incomplete'
        clinical = json.loads((output / 'T1D_saved_score_audit/saved_posterior_cindex_audit.json').read_text())
        assert clinical['all_20_folds_available'] and clinical['n_verified_scores'] == 60
        advance_file = logs / 'advance_rhs_zero_queue.state.json'
        if advance_file.is_file():
            advance = json.loads(advance_file.read_text())
            assert advance['status'] == 'complete' and advance['preexisting_fits_unchanged']
            assert advance['coordinator_resume_signal_sent'] and not advance['dispatcher_suspended']
        if (logs / 'zero_tail_pool.command.json').is_file():
            subprocess.run(['/usr/bin/python3', str(utilities / 'audit_zero_tail_assistance.py'),
                            '--revision-root', str(root), '--outdir', str(output / 'zero_tail_assistance_audit'),
                            '--require-complete'], check=True)
            publication = json.loads((output / 'zero_tail_assistance_audit/publication_audit.json').read_text())
            with (output / 'diagnostic_aggregates_v2/diagnostics_by_run.csv').open() as handle:
                assisted = [row for row in csv.DictReader(handle)
                            if row.get('runtime_context') == 'assisted_fresh_julia_process']
            assert len(assisted) == publication['n_published_fits'], 'Not every assisted fit is included exactly once'
        archive = root / 'reports_final_all_simulations.zip'
        subprocess.run(['python3', str(utilities / 'snapshot_revision_reports.py'),
                        '--revision-root', str(root), '--out', str(archive)], check=True)
        save('complete', completed_simulation_cells=31, verified_outer_scores=60, archive=str(archive))
        print('All requested simulation cells are complete; saved-result archive:', archive, flush=True)
    except Exception as error:
        save('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
