#!/usr/bin/env python3
"""Independently reproduce outer test C-indices from saved draws; never sample.

This verifier implements the recorded T1D preprocessing (fixed pseudocount,
closure, log transform, no standardization) and the project's Harrell convention:
strictly ordered survival times, event at the earlier time, half credit for tied
risk scores. MAP means the saved draw with maximum joint log posterior.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def cindex(times, events, risk):
    comparisons = wins = 0
    for i, (time, event) in enumerate(zip(times, events)):
        if event:
            for j, other_time in enumerate(times):
                if time < other_time:
                    comparisons += 1
                    wins += 1 if risk[i] > risk[j] else .5 if risk[i] == risk[j] else 0
    assert comparisons > 0
    return wins / comparisons, comparisons


def check_outer_partition(directory, n_subjects, n_folds):
    seen_folds, test_subjects = set(), []
    for path in sorted(directory.glob('outer_*/split_indices.json')):
        fold = int(path.parent.name.removeprefix('outer_'))
        assert 1 <= fold <= n_folds and fold not in seen_folds
        seen_folds.add(fold)
        split = json.loads(path.read_text())
        train, test = split['outer_train'], split['outer_test']
        assert train and test and not set(train) & set(test)
        assert sorted(train + test) == list(range(1, n_subjects + 1))
        assert len(split['inner_validation']) == n_folds
        assert sorted(i for part in split['inner_validation'] for i in part) == sorted(train)
        test_subjects.extend(test)
    assert len(test_subjects) == len(set(test_subjects)), 'A subject occurs in multiple outer test folds'
    complete = len(seen_folds) == n_folds
    if complete:
        assert sorted(test_subjects) == list(range(1, n_subjects + 1))
    return dict(n_split_files=len(seen_folds), n_distinct_test_subjects=len(test_subjects),
                no_repeated_test_subjects=True, complete_partition=complete)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision-root', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    root, data_path, out = Path(args.revision_root), Path(args.data), Path(args.outdir)
    data = read_rows(data_path)
    assert all(row['Sex'] in ('M', 'F') and row['Antibiotics'] in ('Tylosin', 'Control') for row in data)
    records, split_reference, completed, partition_checks = [], {}, [], []
    for method, suite in [('cox-rhs', 'T1D_clinical'), ('prox-sslasso', 'corrected_ssl/T1D_clinical')]:
        for arm in ('unadjusted', 'adjusted'):
            source = root / 'results' / suite / arm / method
            spec = json.loads((source / 'analysis_specification.json').read_text())
            assert spec['standardize'] is False and spec['pseudocount'] == 1e-6
            assert spec['outer_folds'] == spec['inner_folds'] == 5
            covariates = spec['component_cols']
            directory = source / 'nested_cv' / method
            partition_checks.append(dict(method=method, adjustment=arm,
                **check_outer_partition(directory, len(data), spec['outer_folds'])))
            seen_completed_folds = set()
            for checkpoint in sorted(directory.glob('outer_*.json')):
                cached = json.loads(checkpoint.read_text())
                if cached.get('status') != 'success':
                    continue
                fold = int(cached['outer_fold'])
                assert 1 <= fold <= spec['outer_folds'] and fold not in seen_completed_folds
                seen_completed_folds.add(fold)
                local = directory / f'outer_{fold:02}'
                split = json.loads((local / 'split_indices.json').read_text())
                train, test = split['outer_train'], split['outer_test']
                assert not set(train) & set(test)
                assert sorted(train + test) == list(range(1, len(data) + 1))
                assert sorted(i for part in split['inner_validation'] for i in part) == sorted(train)
                if fold in split_reference:
                    assert split == split_reference[fold], 'splits differ between methods/adjustments'
                split_reference[fold] = split
                fit = local / 'fit'
                if not (fit / 'samples_wide.csv').is_file():
                    replay = root / 'results/T1D_outer_audit' / arm / method / f'outer_{fold:02}'
                    assert json.loads((replay / 'validation.json').read_text())['reproduces_cached_scores']
                    fit = replay / 'fit'
                sample_file = fit / 'samples_wide.csv'
                draws = read_rows(sample_file)
                assert len(draws) == spec['n_samples'] * spec['chains']
                assert all(sum(int(row['chain']) == c for row in draws) == spec['n_samples']
                           for c in range(1, spec['chains'] + 1))
                columns = [f'beta[{j}]' for j in range(1, len(covariates) + 1)]
                clinical = ['clinical[Sex_M]', 'clinical[Antibiotics_Tylosin]'] if arm == 'adjusted' else []
                assert sorted(k for k in draws[0] if k.startswith('clinical[')) == sorted(clinical)
                columns += clinical
                values = [[float(row[column]) for row in draws] for column in columns]
                assert all(math.isfinite(value) for column in values for value in column)
                logpost = [float(row['logposterior']) for row in draws]
                assert all(math.isfinite(value) for value in logpost)
                best = max(range(len(draws)), key=logpost.__getitem__)
                estimates = dict(mean=[math.fsum(column) / len(column) for column in values],
                    median=[statistics.median(column) for column in values], MAP=[column[best] for column in values])
                design, times, events = [], [], []
                for index in test:
                    row = data[index - 1]
                    x = [float(row[column]) + spec['pseudocount'] for column in covariates]
                    total = math.fsum(x)
                    z = [math.log(value / total) for value in x]
                    if clinical:
                        z += [float(row['Sex'] == 'M'), float(row['Antibiotics'] == 'Tylosin')]
                    design.append(z)
                    times.append(float(row['time']))
                    events.append(int(row['status']))
                for estimate, beta in estimates.items():
                    risk = [math.fsum(x * b for x, b in zip(row, beta)) for row in design]
                    observed, comparable = cindex(times, events, risk)
                    expected = float(cached['test_Cindex_' + estimate])
                    difference = abs(observed - expected)
                    assert difference <= 1e-12, (method, arm, fold, estimate, observed, expected)
                    records.append(dict(method=method, adjustment=arm, outer_fold=fold, estimate=estimate,
                        n_train=len(train), n_test=len(test), n_comparable_pairs=comparable,
                        n_saved_draws=len(draws), recomputed_Cindex=observed, cached_Cindex=expected,
                        absolute_difference=difference, posterior_source=str(fit),
                        posterior_sha256=hashlib.sha256(sample_file.read_bytes()).hexdigest()))
                completed.append(dict(method=method, adjustment=arm, outer_fold=fold))
    out.mkdir(parents=True, exist_ok=True)
    if records:
        with (out / 'saved_posterior_cindex_checks.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    report = dict(n_verified_outer_folds=len(completed), expected_outer_folds=20,
        n_verified_scores=len(records), all_available_scores_match=True, all_20_folds_available=len(completed) == 20,
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(), posterior_sampling_performed=False,
        completed_folds=completed, outer_partition_checks=partition_checks)
    (out / 'saved_posterior_cindex_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'completed_folds'}))
    if args.require_complete:
        assert len(completed) == 20, 'Some outer folds are still incomplete'


if __name__ == '__main__':
    main()
