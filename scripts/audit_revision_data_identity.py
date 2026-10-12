#!/usr/bin/env python3
"""Check paired simulation cohorts from completed eligible runs, without sampling."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from aggregate_revision_diagnostics import read_csv, write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    fields = ['n', 'p', 'nonzero', 'baseline', 'scenario', 'replicate']
    groups, records, pending = defaultdict(list), [], []
    for run in read_csv(Path(args.diagnostics)):
        if run['primary_eligible'].lower() != 'true':
            continue
        source = Path(run['source'])
        if not (source / 'complete.txt').is_file():
            pending.append(str(source))
            continue
        metadata = json.loads((source / 'metadata.json').read_text())
        test = Path(metadata['test_data'])
        spec = json.loads(test.with_suffix('.json').read_text())
        assert spec['n_test'] == metadata['n_test'] == 1000, source
        assert spec['training_seed'] == metadata['seed'], source
        assert spec['test_seed'] == metadata['test_seed'], source
        with test.open() as handle:
            assert sum(1 for _ in handle) - 1 == 1000, test
        train = test.parent.parent / 'training_data' / test.name
        record = {key: run[key] for key in fields}
        record.update(source=str(source), test_path=str(test),
            test_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
            dgp_spec_sha256=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
            training_csv_saved=train.is_file(),
            training_sha256=hashlib.sha256(train.read_bytes()).hexdigest() if train.is_file() else '')
        records.append(record)
        groups[tuple(run[key] for key in fields)].append(record)
    checks = []
    for key, rows in groups.items():
        for field in ('test_sha256', 'dgp_spec_sha256', 'training_sha256'):
            hashes = {row[field] for row in rows if row[field]}
            assert len(hashes) <= 1, (dict(zip(fields, key)), field, [row['source'] for row in rows])
        checks.append(dict(**dict(zip(fields, key)), n_runs=len(rows),
            n_saved_training_csvs=sum(row['training_csv_saved'] for row in rows),
            test_and_dgp_specs_identical=True, saved_training_csvs_identical=True))
    out = Path(args.outdir)
    write_csv(out / 'data_identity_by_run.csv', records)
    write_csv(out / 'data_identity_by_repetition.csv', checks)
    summary = dict(n_completed_runs=len(records), n_repetition_groups=len(checks),
        n_groups_with_multiple_runs=sum(row['n_runs'] > 1 for row in checks),
        n_runs_with_saved_training_csv=sum(row['training_csv_saved'] for row in records),
        all_available_checks_passed=True, pending_incomplete_outputs=pending,
        limitation='Baseline training CSVs were not saved; their common DGP specification and seeds are checked, not training-data byte equality')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'data_identity_audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
