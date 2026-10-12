#!/usr/bin/env python3
"""Bundle a completed report refresh and its audit provenance, without sampling."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision-root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(args.revision_root).resolve()
    results = root / 'results'
    exit_path = root / 'logs/refresh_reports.exit'
    assert exit_path.read_text().strip() == '0', 'complete the report refresh successfully before making a snapshot'
    files = {}
    for folder in ('diagnostic_aggregates_v2', 'performance_aggregates', 'T1D_summary',
                   'data_identity_audit', 'T1D_saved_score_audit', 'zero_sum_saved_audit', 'zero_tail_assistance_audit'):
        for path in sorted((results / folder).glob('*.csv')):
            files[str(path.relative_to(results)).replace('\\', '/')] = path.read_bytes()
    for name in ('legacy_kernel_audit.json', 'legacy_ssl_kernel_audit.json', 'legacy_rhs_target_stress.json',
                 'baseline_reuse_status.json', 'resolved_fixed_rhs_baseline.json', 'car_data_identity_audit.json',
                 'data_identity_audit/data_identity_audit.json',
                 'T1D_saved_score_audit/saved_posterior_cindex_audit.json',
                 'zero_sum_saved_audit/saved_statistics_audit.json',
                 'saved_outputs_audit/saved_outputs_audit.json',
                 'zero_tail_assistance_audit/publication_audit.json'):
        path = results / name
        if path.is_file():
            files[name] = path.read_bytes()
    for name in ('rhs_kernel_followup_logs/queue_status.json', 'corrected_ssl/partition_logs/pool_state.json',
                 'corrected_ssl/T1D_clinical/unadjusted/prox-sslasso/early_stage_logs/state.json'):
        path = results / name
        if path.is_file():
            files[name] = path.read_bytes()
    for name in ('code_v6_ready.json', 'logs/final_source_cache_audit.json',
                 'logs/final_reports_state.json', 'logs/rhs_zero_prox1e-6_first.verified.json',
                 'logs/advance_rhs_zero_queue.state.json', 'logs/advance_rhs_zero_queue.signal_checks.json',
                 'logs/advance_rhs_zero_queue.started_verified.json',
                 'logs/zero_tail_assist/pool_state.json', 'logs/zero_tail_pilot.verified.json',
                 'logs/zero_tail_publication_checks.json', 'logs/zero_tail_pool_checks.json'):
        path = root / name
        if path.is_file():
            files[name] = path.read_bytes()
    for name in ('diagnostic_aggregates_v2/aggregation_errors.csv', 'performance_aggregates/aggregation_errors.csv'):
        assert not list(csv.DictReader(io.StringIO(files[name].decode('utf-8-sig')))), name
    zero_audit_name = 'zero_sum_saved_audit/saved_statistics_audit.json'
    if zero_audit_name in files:
        audit = json.loads(files[zero_audit_name])
        checks = list(csv.DictReader(io.StringIO(files['zero_sum_saved_audit/saved_statistics_checks.csv'].decode('utf-8-sig'))))
        assert audit['all_checks_passed'] and audit['posterior_sampling_performed'] is False
        assert len(checks) == audit['n_verified_runs'] == 10
        assert sum(int(row['n_scalar_checks']) for row in checks) == audit['n_scalar_checks']
        assert len({row['test_data_sha256'] for row in checks}) == 1
    publication_name = 'zero_tail_assistance_audit/publication_audit.json'
    if publication_name in files:
        audit = json.loads(files[publication_name])
        assert audit['all_available_checks_passed'] and audit['all_published_samples_unchanged']
        checks_name = 'zero_tail_assistance_audit/published_fit_checks.csv'
        checks = list(csv.DictReader(io.StringIO(files[checks_name].decode('utf-8-sig')))) if checks_name in files else []
        assert len(checks) == audit['n_published_fits']
    progress = list(csv.DictReader(io.StringIO(files['performance_aggregates/requested_progress.csv'].decode('utf-8-sig'))))
    assert len(progress) == 31
    outputs_audit_name = 'saved_outputs_audit/saved_outputs_audit.json'
    if outputs_audit_name in files:
        audit = json.loads(files[outputs_audit_name])
        assert audit['all_available_checks_passed'] and audit['posterior_sampling_performed'] is False
        assert audit['completed_simulation_cells'] == sum(int(row['n_complete']) == 50 for row in progress)
        assert audit['complete'] == all(int(row['n_complete']) == 50 for row in progress)
    # Read every payload before writing the ZIP; its manifest always describes
    # these exact bytes even if subsequent fits finish while the archive is made.
    manifest = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='snapshot; inspect progress denominators and clinical pool status for completion',
        completed_simulation_cells=sum(int(row['n_complete']) == 50 for row in progress),
        files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr('snapshot_manifest.json', json.dumps(manifest, indent=2))
    temporary.replace(output)
    print(json.dumps(dict(archive=str(output), n_files=len(files),
        completed_simulation_cells=manifest['completed_simulation_cells'])))


if __name__ == '__main__':
    main()
