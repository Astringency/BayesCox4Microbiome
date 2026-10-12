#!/usr/bin/env python3
"""Verify that published future fits remain intact; never fit or move files."""
import argparse
import csv
import datetime
import json
from collections import defaultdict
from pathlib import Path

from assist_zero_sum_tail import digest, same_process
from advance_rhs_zero_queue import process_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision-root', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    root, logs = args.revision_root.resolve(), args.revision_root.resolve() / 'logs/zero_tail_assist'
    pool_file = logs / 'pool_state.json'
    pool = json.loads(pool_file.read_text()) if pool_file.exists() else {}
    records, pending, cancelled = [], [], []
    paused_seconds = defaultdict(float)
    for path in sorted(logs.glob('*.state.json')):
        state = json.loads(path.read_text())
        assert state['status'] != 'failed', path
        if state['status'] == 'cancelled_original_reached_repetition':
            cancelled.append(path.name)
            continue
        if state['status'] != 'complete':
            pending.append(path.name)
            continue
        assert state['published'] and state['published_hashes_match'] and state['original_resume_signal_sent'], path
        target = Path(state['destination']).resolve()
        assert target.is_relative_to(root / 'results')
        for name in ('samples_wide.csv', 'metadata.json', 'complete.txt'):
            assert digest(target / name) == state['fit_hashes'][name], (target, name)
        meta = json.loads((target / 'metadata.json').read_text())
        assert meta['replicate_id'] == state['replicate']
        assert Path(meta['test_data']).resolve().is_relative_to(root / 'zero_tail_assist_staging')
        assert digest(Path(state['project']) / 'source_manifest.json') == state['frozen_manifest_sha256']
        if args.require_complete and same_process(state['original']):
            assert process_info(state['original']['pid'])['state'] not in ('T', 't'), path
        pause = float(state['publication_pause_seconds'])
        paused_seconds[str(state['original']['pid'])] += pause
        records.append(dict(source=str(target), method=meta['method_name'], constraint=meta['constraint'],
                            replicate=state['replicate'], samples_sha256=state['fit_hashes']['samples_wide.csv'],
                            original_pid=state['original']['pid'], publication_pause_seconds=pause,
                            runtime_context='assisted_fresh_julia_process'))
    if args.require_complete:
        assert pool.get('status') == 'complete' and not pool.get('failures') and not pending
    assert len({row['source'] for row in records}) == len(records)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if records:
        with (args.outdir / 'published_fit_checks.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    audit = dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 all_available_checks_passed=True, n_published_fits=len(records),
                 pending_helpers=pending, cancelled_helpers=cancelled, pool_status=pool.get('status'),
                 all_published_samples_unchanged=True, all_original_resume_signals_sent=True,
                 publication_pause_seconds_by_original_pid=dict(paused_seconds),
                 max_publication_pause_seconds=max((r['publication_pause_seconds'] for r in records), default=0),
                 posterior_sampling_performed=False,
                 runtime_limitation='Assisted fits use fresh Julia processes; runtime includes compilation and shared-server load. Original fits can include the recorded brief publication pauses.')
    (args.outdir / 'publication_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
