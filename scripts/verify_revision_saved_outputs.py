#!/usr/bin/env python3
"""Check revision budgets and report completeness using saved outputs; never fit.

This audit checks all reported primary fits, including poorly mixing fits. Draw
counts are checked against saved zero-sum summaries, not by rereading every large
sample CSV. Independent numerical audits and convergence results remain separate.
"""
import argparse
import csv
import datetime
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


LAUNCHERS = (
    'run_sensitivity_revision.sh', 'run_scenarios_revision.sh',
    'run_zero_sum_revision.sh', 'run_t1d_clinical_revision.sh',
    'run_car_revision.sh', 'run_revision_diagnostics.sh',
)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def is_true(value):
    return str(value).lower() == 'true'


def expected_cells():
    cells = set()
    for method in ('cox-rhs', 'prox-sslasso'):
        cells.update(('lambda', setting, method) for setting in ('0.01', '0.001', '0.0001', '1e-05'))
        cells.update(('scenarios', setting, method) for setting in ('S1', 'S2', 'S3'))
        cells.update(('zero_sum', setting, method) for setting in ('none', 'soft_center', 'prox0.001', 'prox1e-06', 'exact'))
    cells.update(('p0', str(p0), 'cox-rhs') for p0 in (3, 6, 12))
    cells.update(('car', graph, 'cox-rhs') for graph in ('independent', 'correct', 'misspecified'))
    cells.add(('car', 'independent', 'prox-sslasso'))
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision-root', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    root = args.revision_root.resolve()
    results = root / 'results'
    progress = read_csv(results / 'performance_aggregates/requested_progress.csv')
    keys = [(row['suite'], row['setting'], row['method']) for row in progress]
    assert len(keys) == len(set(keys)) == 31 and set(keys) == expected_cells()
    for row in progress:
        assert int(row['expected_replicates']) == 50
        assert 0 <= int(row['n_complete']) <= 50
        missing = [int(x) for x in row['missing_replicates'].split(',') if x]
        assert len(missing) == len(set(missing)) == 50 - int(row['n_complete'])
        assert set(missing) <= set(range(1, 51))
        assert (row['status'] == 'complete') == (not missing)
    for folder in ('performance_aggregates', 'diagnostic_aggregates_v2'):
        assert not read_csv(results / folder / 'aggregation_errors.csv'), folder

    diagnostics = read_csv(results / 'diagnostic_aggregates_v2/diagnostics_by_run.csv')
    primary = [row for row in diagnostics if is_true(row['primary_eligible'])]
    sources = {row['source'] for row in primary}
    assert len(sources) == len(primary), 'duplicate primary diagnostic sources'
    covered = set()
    numeric_settings = ('n', 'p', 'nonzero', 'p0', 'prox_sum0_lambda')
    string_settings = ('method', 'baseline', 'scenario', 'constraint', 'car_graph')
    for cell in progress:
        selected = [row for row in primary
                    if all(row[name] == cell[name] for name in string_settings)
                    and all((float(row[name]) if row[name] else None) ==
                            (float(cell[name]) if cell[name] else None)
                            for name in numeric_settings)]
        repetitions = [int(row['replicate']) for row in selected]
        assert len(repetitions) == len(set(repetitions)) == int(cell['n_complete'])
        missing = {int(x) for x in cell['missing_replicates'].split(',') if x}
        assert set(repetitions) == set(range(1, 51)) - missing
        covered.update(row['source'] for row in selected)
    assert covered == sources, 'primary fits outside the requested experiment matrix'
    method_counts = Counter()
    for row in primary:
        source = Path(row['source'])
        meta = json.loads((source / 'metadata.json').read_text())
        for key, expected in dict(chains=4, n_samples=5000, warmup=2000, saved_sample_thin=1).items():
            assert meta[key] == expected, (source, key)
        assert int(row['replicate']) == meta['replicate_id'] in range(1, 51)
        assert meta['n_test'] == 1000
        assert meta['method_name'] == row['method']
        assert meta['constraint'] == row['constraint']
        for field in ('n', 'p'):
            assert int(row[field]) == meta[field], (source, field)
        assert int(row['nonzero']) == meta['nonzero_count']
        assert float(row['sum_beta_n_draws']) == 20000
        assert is_true(row['transition_diagnostics_available'])
        assert math.isfinite(float(row['total_seconds'])) and float(row['total_seconds']) > 0
        assert 0 <= float(row['mean_acceptance_probability']) <= 1
        assert 0 <= int(float(row['n_divergent_sample'])) <= 20000
        assert int(row['n_beta_parameters']) == int(row['p'])
        assert int(row['beta_n_rhat_unavailable']) == 0
        for name in ('complete.txt', 'samples_wide.csv', 'posterior_summary.csv',
                     'rank_diagnostics.csv', 'zero_sum_summary.csv',
                     'transition_diagnostics.csv', 'transition_summary.csv'):
            assert (source / name).is_file(), (source, name)
        if row['method'] == 'prox-sslasso':
            assert (source / 'selection_mixing.csv').is_file(), source
        if row['constraint'] == 'exact':
            assert float(row['sum_beta_fraction_within_roundoff']) == 1, source
        method_counts[row['method']] += 1

    metric_rows = read_csv(results / 'performance_aggregates/metrics_by_run.csv')
    estimates = defaultdict(list)
    for row in metric_rows:
        if not is_true(row['primary_eligible']):
            continue
        assert row['source'] in sources
        estimates[row['source']].append(row['estimate'])
        assert float(row['n_test']) == 1000
        for name in ('Cindex_train', 'Cindex_test', 'coverage_95_all',
                     'coverage_95_nonzero', 'coverage_95_zero', 'empirical_FDP'):
            assert 0 <= float(row[name]) <= 1, (row['source'], name)
        assert math.isfinite(float(row['MSE'])) and float(row['MSE']) >= 0
    assert set(estimates) == sources
    assert all(sorted(labels) == ['MAP', 'mean', 'median'] for labels in estimates.values())

    identity = json.loads((results / 'data_identity_audit/data_identity_audit.json').read_text())
    identities = read_csv(results / 'data_identity_audit/data_identity_by_run.csv')
    assert identity['all_available_checks_passed'] and not identity['pending_incomplete_outputs']
    assert identity['n_completed_runs'] == len(identities) == len(primary)
    assert {row['source'] for row in identities} == sources
    clinical = json.loads((results / 'T1D_saved_score_audit/saved_posterior_cindex_audit.json').read_text())
    assert clinical['all_available_scores_match'] and clinical['all_20_folds_available']
    assert clinical['n_verified_outer_folds'] == 20 and clinical['n_verified_scores'] == 60
    assert len(clinical['outer_partition_checks']) == 4
    assert all(row['complete_partition'] and row['no_repeated_test_subjects']
               and row['n_distinct_test_subjects'] == 173 for row in clinical['outer_partition_checks'])
    assert all((root / 'code_v6/scripts' / name).is_file() for name in LAUNCHERS)

    complete_cells = sum(row['status'] == 'complete' for row in progress)
    complete = complete_cells == 31
    if complete:
        # 31 logical cells share the 50-run RHS lambda=0.001 / p0=6 baseline.
        assert len(primary) == 1500
        assert method_counts == {'cox-rhs': 850, 'prox-sslasso': 650}
    report = dict(
        verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='All primary fits in the supplied completed report refresh; later fits are not covered',
        all_available_checks_passed=True, complete=complete,
        n_primary_fits=len(primary), n_primary_fits_by_method=dict(method_counts),
        completed_simulation_cells=complete_cells, expected_simulation_cells=31,
        expected_distinct_fits=1500, expected_repetitions_per_cell=50,
        chains=4, warmup_per_chain=2000, retained_per_chain=5000, saved_thin=1,
        n_test=1000, all_saved_summary_draw_counts_match=True,
        all_hmc_transition_diagnostics_present=True, all_ssl_selection_mixing_present=True,
        all_exact_saved_summaries_within_roundoff=True,
        clinical_outer_folds=20, independently_reproduced_clinical_scores=60,
        suite_scripts=list(LAUNCHERS), posterior_sampling_performed=False,
        limitations=[
            'Draw counts and exact residuals use saved summaries, not a new scan of every posterior draw.',
            identity['limitation'],
            'Completeness does not imply convergence or proportional-hazards adequacy; retain diagnostic failures in primary summaries.',
        ])
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / 'saved_outputs_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    if args.require_complete and not complete:
        raise SystemExit('Requested experiments remain incomplete; no sampling was performed.')


if __name__ == '__main__':
    main()
