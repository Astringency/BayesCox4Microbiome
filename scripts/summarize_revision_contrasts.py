#!/usr/bin/env python3
"""Paired ablation differences using the same simulation repetition/test cohort.

Differences are setting minus reference. Positive C-index differences improve
prediction; positive MSE differences increase error. Coverage differences are
descriptive, not a claim that greater-than-nominal coverage is preferable.
"""
import argparse
import math
import statistics
from pathlib import Path

from aggregate_revision_diagnostics import number, read_csv, write_csv
from aggregate_revision_metrics import METRICS, matches, requested_cells, truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics-by-run', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--profile', choices=('revision', 's4'), default='revision')
    args = parser.parse_args()
    rows = [r for r in read_csv(Path(args.metrics_by_run)) if truth(r['primary_eligible'])]
    cells = {}
    for suite, label, settings in requested_cells(args.profile):
        key = (suite, label, settings['method'])
        cells[key] = {(int(r['replicate']), r['estimate']): r for r in rows if matches(r, settings)}
    references = {'lambda': '0.001', 'p0': '6', 'car': 'independent', 'car_s4': 'independent'}
    differences, summaries = [], []
    for (suite, label, method), cell in cells.items():
        reference_label = references.get(suite)
        if reference_label is None or label == reference_label:
            continue
        reference = cells[(suite, reference_label, method)]
        local = []
        for key in sorted(cell.keys() & reference.keys()):
            row, baseline = cell[key], reference[key]
            assert row['test_seed'] and row['test_seed'] == baseline['test_seed'], 'paired test-cohort identity mismatch'
            result = dict(suite=suite, setting=label, method=method, reference=reference_label,
                replicate=key[0], estimate=key[1], test_seed=row['test_seed'],
                source=row['source'], reference_source=baseline['source'],
                both_convergence_screen_pass=truth(row['convergence_screen_pass']) and truth(baseline['convergence_screen_pass']))
            for metric in METRICS:
                a, b = number(row.get(metric)), number(baseline.get(metric))
                result[metric + '_difference'] = a - b if a is not None and b is not None else None
            local.append(result)
        differences.extend(local)
        for estimate in ('mean', 'median', 'MAP'):
            for selection in ('all_eligible_pairs', 'both_convergence_screen_pass'):
                subset = [r for r in local if r['estimate'] == estimate and
                          (selection == 'all_eligible_pairs' or r['both_convergence_screen_pass'])]
                result = dict(suite=suite, setting=label, method=method, reference=reference_label,
                    estimate=estimate, selection=selection, n_pairs=len(subset), expected_pairs=50,
                    all_50_pairs_available=sum(r['estimate'] == estimate for r in local) == 50)
                for metric in METRICS:
                    vals = [r[metric + '_difference'] for r in subset if r[metric + '_difference'] is not None]
                    sd = statistics.stdev(vals) if len(vals) > 1 else None
                    result[metric + '_n_pairs'] = len(vals)
                    result[metric + '_mean_difference'] = statistics.mean(vals) if vals else None
                    result[metric + '_paired_sd'] = sd
                    result[metric + '_paired_mcse'] = sd / math.sqrt(len(vals)) if sd is not None else None
                summaries.append(result)
    out = Path(args.outdir)
    write_csv(out / 'paired_differences.csv', differences)
    write_csv(out / 'paired_contrasts.csv', summaries)
    print(f'{len(differences)} paired replicate/estimate rows; {len(summaries)} contrast summaries')


if __name__ == '__main__':
    main()
