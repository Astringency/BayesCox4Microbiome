#!/usr/bin/env python3
"""Audit saved S4 formal results and paired data; never sample or filter mixing."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from aggregate_revision_diagnostics import number, read_csv
from aggregate_revision_metrics import METRICS, matches, requested_cells, truth


def verify(root, require_complete=False):
    reports = root / 'reports'
    progress = read_csv(reports / 'performance/requested_progress.csv')
    assert {(r['suite'], r['setting'], r['method']) for r in progress} == {
        (suite, label, settings['method']) for suite, label, settings in requested_cells('s4')}
    for name in ('diagnostics', 'performance'):
        assert not read_csv(reports / name / 'aggregation_errors.csv')
    runs = read_csv(reports / 'diagnostics/diagnostics_by_run.csv')
    assert len({r['source'] for r in runs}) == len(runs)
    assert all(truth(r['primary_eligible']) for r in runs)
    cells = requested_cells('s4')
    counts, identities, trees, fingerprints = Counter(), defaultdict(list), set(), set()
    for row in runs:
        assert sum(matches(row, settings) for _, _, settings in cells) == 1
        source = Path(row['source'])
        meta = json.loads((source / 'metadata.json').read_text())
        for key, expected in dict(n=200, p=50, nonzero_count=6, n_test=1000,
                chains=4, n_samples=5000, warmup=2000, saved_sample_thin=1,
                dgp_version='phylo_bm_v1', phylogenetic_covariance_model='brownian_shared_path').items():
            assert meta[key] == expected, (source, key)
        assert meta['use_car'] == (row['car_graph'] != 'independent')
        assert meta['sample_car_strength'] == meta['use_car']
        assert int(float(row['sum_beta_n_draws'])) == 20000
        assert int(row['n_beta_parameters']) == 50
        assert truth(row['transition_diagnostics_available'])
        for name in ('complete.txt', 'samples_wide.csv', 'posterior_summary.csv',
                     'rank_diagnostics.csv', 'zero_sum_summary.csv', 'transition_diagnostics.csv'):
            assert (source / name).is_file(), (source, name)
        rep = int(row['replicate'])
        assert 1 <= rep <= 50
        counts[row['car_graph']] += 1
        trees.add(meta['phylogenetic_tree_sha256'])
        fingerprints.add(meta['phylogenetic_covariance_sha256'])
        test = Path(meta['test_data'])
        train = test.parent.parent / 'training_data' / test.name
        for path, n in ((train, 200), (test, 1000)):
            with path.open() as handle:
                assert sum(1 for _ in handle) - 1 == n, path
        identities[rep].append(tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in (train, test)))
    assert len(trees) <= 1 and len(fingerprints) <= 1
    assert all(len(set(values)) == 1 for values in identities.values())
    assert all(len(values) <= 3 for values in identities.values())
    scores = read_csv(reports / 'performance/metrics_by_run.csv')
    estimates = defaultdict(list)
    for row in scores:
        estimates[row['source']].append(row['estimate'])
        for metric in METRICS:
            value = number(row[metric])
            assert value is not None and value >= 0, (row['source'], metric)
            if metric.startswith(('Cindex', 'coverage')) or metric == 'empirical_FDP':
                assert value <= 1
    assert set(estimates) == {r['source'] for r in runs}
    assert all(sorted(values) == ['MAP', 'mean', 'median'] for values in estimates.values())
    for row in progress:
        selected = [r for r in runs if r['car_graph'] == row['setting']]
        reps = {int(r['replicate']) for r in selected}
        assert len(selected) == len(reps) == int(row['n_complete'])
        assert {int(x) for x in row['missing_replicates'].split(',') if x} == set(range(1, 51)) - reps
    complete = len(runs) == 150 and all(len(v) == 3 for v in identities.values())
    result = dict(all_available_checks_passed=True, complete=complete,
        completed_fits=len(runs), expected_fits=150, counts=dict(counts),
        paired_repetitions=sum(len(v) == 3 for v in identities.values()),
        convergence_pass_by_graph={g: sum(truth(r['convergence_screen_pass']) for r in runs
            if r['car_graph'] == g) for g in counts},
        tree_sha256=next(iter(trees), None), covariance_sha256=next(iter(fingerprints), None),
        primary_summary='All eligible repetitions, including poor mixing; screened results are supplementary')
    (reports / 'saved_results_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    if require_complete:
        assert complete, 'S4 formal experiment is incomplete'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.require_complete)))
