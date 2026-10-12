#!/usr/bin/env python3
"""Join saved performance metrics to audited diagnostic eligibility; never fit.

All eligible repetitions remain in the primary table, including poor-mixing
ones. The additional convergence-screened table is explicitly a sensitivity
summary. Incomplete cells report their actual counts, not an apparent 50 repeats.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from aggregate_revision_diagnostics import GROUPS, number, read_csv, write_csv

METRICS = ['Cindex_train', 'Cindex_test', 'MSE', 'coverage_95_all',
           'coverage_95_nonzero', 'coverage_95_zero', 'empirical_FDP', 'FP', 'TP', 'FN', 'TN']


def truth(value):
    return str(value).lower() == 'true'


def summarize(rows, group_keys):
    grouped = defaultdict(list)
    for row in rows:
        if truth(row['primary_eligible']):
            grouped[tuple(row.get(k, '') for k in group_keys)].append(row)
    summaries = []
    for key, values in grouped.items():
        for selection in ('all_eligible', 'convergence_screen_pass'):
            selected = values if selection == 'all_eligible' else [r for r in values if truth(r['convergence_screen_pass'])]
            out = dict(zip(group_keys, key))
            out.update(selection=selection, n_replicates=len(selected), n_available=len(values),
                       n_convergence_screen_pass=sum(truth(r['convergence_screen_pass']) for r in values))
            for metric in METRICS:
                vals = [v for r in selected if (v := number(r.get(metric))) is not None]
                out[metric + '_n'] = len(vals)
                out[metric + '_mean'] = statistics.mean(vals) if vals else None
                out[metric + '_sd'] = statistics.stdev(vals) if len(vals) > 1 else None
                out[metric + '_mcse'] = statistics.stdev(vals) / len(vals)**.5 if len(vals) > 1 else None
            summaries.append(out)
    return summaries


def requested_cells(profile='revision'):
    if profile == 's4':
        return [('car_s4', graph, dict(method='cox-rhs', n=200, p=50, nonzero=6,
                baseline='weibull', scenario='phylo_bm', constraint='prox',
                prox_sum0_lambda=.001, p0=6, car_graph=graph))
                for graph in ('independent', 'correct', 'misspecified')]
    if profile != 'revision':
        raise ValueError('unknown study profile: ' + profile)
    cells = []
    for method in ('cox-rhs', 'prox-sslasso'):
        prior_p0 = 6 if method == 'cox-rhs' else None
        base = dict(method=method, nonzero=6, p=50, baseline='weibull', scenario='baseline', constraint='prox')
        for lam in (1e-2, 1e-3, 1e-4, 1e-5):
            cells.append(('lambda', str(lam), dict(base, n=400, prox_sum0_lambda=lam,
                          car_graph='abundance' if method == 'cox-rhs' else 'independent', p0=prior_p0)))
        for label, scenario in [('S1', 'weak'), ('S2', 'correlated_null'), ('S3', 'counts')]:
            cells.append(('scenarios', label, dict(base, n=200, scenario=scenario,
                          prox_sum0_lambda=.001 if method == 'cox-rhs' else 1e-6,
                          car_graph='abundance' if method == 'cox-rhs' else 'independent', p0=prior_p0)))
        for constraint, lam in [('none', None), ('soft_center', None), ('prox', .001), ('prox', 1e-6), ('exact', None)]:
            cells.append(('zero_sum', constraint if lam is None else f'prox{lam:g}',
                          dict(base, n=400, p=150, nonzero=12, baseline='gompertz', constraint=constraint,
                          prox_sum0_lambda=lam, car_graph='abundance' if method == 'cox-rhs' else 'independent',
                          p0=12 if method == 'cox-rhs' else None)))
        for graph in (('independent', 'correct', 'misspecified') if method == 'cox-rhs' else ('independent',)):
            cells.append(('car', graph, dict(base, n=200, scenario='phylo', car_graph=graph,
                          prox_sum0_lambda=.001 if method == 'cox-rhs' else 1e-6, p0=prior_p0)))
    for p0 in (3, 6, 12):
        cells.append(('p0', str(p0), dict(method='cox-rhs', n=400, p=50, nonzero=6,
                      baseline='weibull', scenario='baseline', constraint='prox', prox_sum0_lambda=.001,
                      p0=p0, car_graph='abundance')))
    return cells


def matches(row, settings):
    for key, expected in settings.items():
        actual = row.get(key)
        if expected is None:
            if actual not in ('', None):
                return False
        elif isinstance(expected, (int, float)):
            if number(actual) != expected:
                return False
        elif actual != expected:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--historical-metrics')
    parser.add_argument('--profile', choices=('revision', 's4'), default='revision')
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    historical = defaultdict(list)
    for row in (read_csv(Path(args.historical_metrics)) if args.historical_metrics else []):
        historical[row['source']].append(row)
    rows, errors = [], []
    for diag in read_csv(Path(args.diagnostics)):
        for key in ('n', 'p', 'nonzero', 'p0', 'prox_sum0_lambda'):
            diag[key] = number(diag.get(key))
        source = Path(diag['source'])
        try:
            metrics = historical.get(str(source))
            if metrics is None:
                metrics = read_csv(source / 'metrics.csv')
                fd = read_csv(source / 'false_discovery.csv')
                if fd:
                    for metric in metrics:
                        metric.update({k: fd[0].get(k) for k in ('FP', 'TP', 'FN', 'TN')})
                        metric['empirical_FDP'] = fd[0].get('FDR')
            if not metrics:
                raise ValueError('no saved or replayed performance metrics')
            for metric in metrics:
                row = dict(diag, estimate=metric['estimate'], n_test=metric.get('n_test'),
                           test_seed=metric.get('test_seed'))
                row.update({k: number(metric.get(k)) for k in METRICS})
                rows.append(row)
        except Exception as e:
            errors.append(dict(source=str(source), error=str(e)))
    # Duplicating a repeated fit would overstate precision; make it an error.
    seen = {}
    for row in rows:
        if not truth(row['primary_eligible']):
            continue
        key = tuple(row.get(k) for k in GROUPS + ['replicate', 'estimate'])
        if key in seen:
            errors.append(dict(source=row['source'], error='duplicate eligible cell/replicate/estimate: ' + seen[key]))
        seen[key] = row['source']
    progress, requested = [], []
    for suite, label, settings in requested_cells(args.profile):
        subset = [dict(r, suite=suite, setting=label) for r in rows if matches(r, settings) and truth(r['primary_eligible'])]
        means = [r for r in subset if r['estimate'] == 'mean']
        reps = {int(r['replicate']) for r in means}
        progress.append(dict(suite=suite, setting=label, **settings, expected_replicates=50,
                             n_complete=len(reps), missing_replicates=','.join(str(i) for i in range(1, 51) if i not in reps),
                             n_convergence_screen_pass=sum(truth(r['convergence_screen_pass']) for r in means),
                             status='complete' if reps == set(range(1, 51)) else 'incomplete'))
        requested.extend(subset)
    out = Path(args.outdir)
    write_csv(out / 'metrics_by_run.csv', rows)
    write_csv(out / 'metrics_by_setting.csv', summarize(rows, GROUPS + ['estimate']))
    write_csv(out / 'requested_metrics.csv', summarize(requested, ['suite', 'setting'] + GROUPS + ['estimate']))
    write_csv(out / 'requested_progress.csv', progress)
    write_csv(out / 'aggregation_errors.csv', errors, columns=['source', 'error'])
    print(json.dumps(dict(n_metric_rows=len(rows), n_requested_cells=len(progress),
                          n_complete_cells=sum(r['status'] == 'complete' for r in progress), n_errors=len(errors))))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
