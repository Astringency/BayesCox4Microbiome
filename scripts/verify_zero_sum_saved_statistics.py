#!/usr/bin/env python3
"""Audit one saved formal repetition in each zero-sum cell, without sampling.

Independently recompute coefficient summaries, coefficient-sum statistics,
interval coverage/FDP, test C-indices, and recorded HMC transition summaries.
This checks numerical reporting, not posterior convergence or all repetitions.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def quantile(ordered, probability):
    position = (len(ordered) - 1) * probability
    lower = int(position)
    weight = position - lower
    return ordered[lower] if weight == 0 else (1 - weight) * ordered[lower] + weight * ordered[lower + 1]


def harrell(times, events, risk):
    denominator = numerator = 0
    for i, (time, event) in enumerate(zip(times, events)):
        if event:
            for j, other in enumerate(times):
                if time < other:
                    denominator += 1
                    numerator += 1 if risk[i] > risk[j] else .5 if risk[i] == risk[j] else 0
    assert denominator > 0
    return numerator / denominator


def verify_run(run):
    assert (run / 'complete.txt').is_file(), run
    meta = json.loads((run / 'metadata.json').read_text())
    assert (meta['n'], meta['p'], meta['nonzero_count']) == (400, 150, 12)
    assert (meta['chains'], meta['n_samples'], meta['warmup'], meta['saved_sample_thin']) == (4, 5000, 2000, 1)
    assert meta['pseudocount'] == 0 and meta['standardize'] is False
    count = 0

    def check(actual, saved, name, atol=1e-11):
        nonlocal count
        actual, saved = float(actual), float(saved)
        assert math.isfinite(actual) and math.isfinite(saved), (run, name)
        assert math.isclose(actual, saved, rel_tol=1e-10, abs_tol=atol), (run, name, actual, saved)
        count += 1

    coefficients = [[] for _ in range(meta['p'])]
    chains, iterations, sums, roundoff = [], defaultdict(list), [], []
    best_logpost, map_beta = -math.inf, None
    with (run / 'samples_wide.csv').open(newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        beta_indices = [header.index(f'beta[{j}]') for j in range(1, meta['p'] + 1)]
        chain_index, iter_index, lp_index = [header.index(key) for key in ('chain', 'iter', 'logposterior')]
        for row in reader:
            chain = row[chain_index]
            chains.append(chain)
            iterations[chain].append(int(row[iter_index]))
            beta = [float(row[j]) for j in beta_indices]
            assert all(math.isfinite(value) for value in beta)
            for column, value in zip(coefficients, beta):
                column.append(value)
            total = math.fsum(beta)
            sums.append(total)
            roundoff.append(abs(total) <= 100 * math.ulp(1.) * max(1., math.fsum(map(abs, beta))))
            lp = float(row[lp_index])
            assert math.isfinite(lp)
            if lp > best_logpost:
                best_logpost, map_beta = lp, beta
    assert Counter(chains) == {str(j): 5000 for j in range(1, 5)}
    assert all(values == list(range(1, 5001)) for values in iterations.values())
    saved_summary = rows(run / 'posterior_summary.csv')
    assert [r['variable'] for r in saved_summary] == [f'beta[{j}]' for j in range(1, 151)]
    estimates = dict(mean=[], median=[], map=map_beta)
    covered, active, selected = [], [], []
    for j, (column, saved) in enumerate(zip(coefficients, saved_summary)):
        ordered = sorted(column)
        values = dict(mean=statistics.fmean(column), median=quantile(ordered, .5),
                      lower=quantile(ordered, .025), upper=quantile(ordered, .975), map=map_beta[j])
        for key, value in values.items():
            check(value, saved[key], f'beta[{j + 1}].{key}')
        for key in ('mean', 'median'):
            estimates[key].append(values[key])
        truth = float(saved['true_beta'])
        active.append(truth != 0)
        covered.append(values['lower'] <= truth <= values['upper'])
        selected.append(values['lower'] > 0 or values['upper'] < 0)
    assert sum(active) == 12
    coverage = dict(coverage_95_all=statistics.fmean(covered),
                    coverage_95_nonzero=statistics.fmean(c for c, a in zip(covered, active) if a),
                    coverage_95_zero=statistics.fmean(c for c, a in zip(covered, active) if not a))
    fd = dict(TP=sum(a and s for a, s in zip(active, selected)),
              FP=sum(not a and s for a, s in zip(active, selected)),
              FN=sum(a and not s for a, s in zip(active, selected)),
              TN=sum(not a and not s for a, s in zip(active, selected)))
    fd['FDR'] = fd['FP'] / max(fd['TP'] + fd['FP'], 1)
    saved_fd = rows(run / 'false_discovery.csv')
    assert len(saved_fd) == 1
    for key, value in fd.items():
        check(value, saved_fd[0][key], key)

    saved_zero = rows(run / 'zero_sum_summary.csv')
    assert {r['chain'] for r in saved_zero} == {'1', '2', '3', '4', 'pooled'} and len(saved_zero) == 5
    for saved in saved_zero:
        indices = list(range(len(chains))) if saved['chain'] == 'pooled' else [i for i, c in enumerate(chains) if c == saved['chain']]
        totals = [sums[i] for i in indices]
        ordered = sorted(totals)
        values = dict(n_draws=len(totals), p=150, mean=statistics.fmean(totals), sd=statistics.stdev(totals),
                      mean_abs=statistics.fmean(map(abs, totals)), rms=math.sqrt(statistics.fmean(x*x for x in totals)),
                      max_abs=max(map(abs, totals)), fraction_within_roundoff=statistics.fmean(roundoff[i] for i in indices),
                      sum_posterior_means=math.fsum(statistics.fmean(column[i] for i in indices) for column in coefficients),
                      sum_marginal_medians=math.fsum(statistics.median(column[i] for i in indices) for column in coefficients))
        for key, q in [('q025', .025), ('q25', .25), ('median', .5), ('q75', .75), ('q975', .975)]:
            values[key] = quantile(ordered, q)
        for exponent in (2, 3, 6, 8):
            values[f'prob_abs_lt_1e_{exponent}'] = statistics.fmean(abs(x) < 10**(-exponent) for x in totals)
        for key, value in values.items():
            check(value, saved[key], f"sum.{saved['chain']}.{key}")
    if meta['constraint'] == 'exact':
        assert all(roundoff), 'An exact draw exceeds its coefficient-scale rounding tolerance'

    test_path = Path(meta['test_data'])
    test = rows(test_path)
    assert len(test) == meta['n_test'] == 1000
    design = []
    for row in test:
        raw = [float(row[f'x{j}']) for j in range(1, 151)]
        assert all(x > 0 and math.isfinite(x) for x in raw)
        total = math.fsum(raw)
        design.append([math.log(x / total) for x in raw])
    times, events = [float(r['time']) for r in test], [int(r['event']) for r in test]
    saved_metrics = rows(run / 'metrics.csv')
    assert {r['estimate'] for r in saved_metrics} == {'mean', 'median', 'MAP'} and len(saved_metrics) == 3
    test_scores = {}
    for metric in saved_metrics:
        for key, value in coverage.items():
            check(value, metric[key], key)
        beta = estimates[metric['estimate'].lower()]
        score = harrell(times, events, [math.fsum(x*b for x, b in zip(row, beta)) for row in design])
        check(score, metric['Cindex_test'], 'Cindex_test', atol=1e-12)
        test_scores[metric['estimate']] = score

    transitions = defaultdict(list)
    for row in rows(run / 'transition_diagnostics.csv'):
        transitions[row['chain'], row['phase']].append(row)
    saved_transition = rows(run / 'transition_summary.csv')
    assert len(transitions) == len(saved_transition) == 8
    for saved in saved_transition:
        group = transitions[saved['chain'], saved['phase']]
        expected = 5000 if saved['phase'] == 'sample' else 2000
        start = meta['warmup'] + 1 if saved['phase'] == 'sample' else 1
        assert len(group) == expected and sorted(int(r['iteration']) for r in group) == list(range(start, start + expected))
        divergent = sum(r['divergent'].lower() == 'true' for r in group)
        values = dict(n_transitions=len(group), n_divergent=divergent, divergence_rate=divergent / len(group),
                      mean_acceptance_probability=statistics.fmean(float(r['acceptance_probability']) for r in group))
        for key, value in values.items():
            check(value, saved[key], 'transition.' + key)
    return dict(source=str(run), replicate=meta['replicate_id'], method=meta['method_name'],
                constraint=meta['constraint'], n_draws=len(sums), n_scalar_checks=count,
                **coverage, empirical_FDP=fd['FDR'], Cindex_test_mean=test_scores['mean'],
                max_abs_sum_beta=max(map(abs, sums)), samples_sha256=sha256(run / 'samples_wide.csv'),
                test_data_sha256=sha256(test_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision-root', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    parser.add_argument('--replicate', type=int, default=1)
    args = parser.parse_args()
    assert 1 <= args.replicate <= 50
    records = []
    for branch, method in [('fixed_parallel', 'cox-rhs'), ('corrected_ssl', 'prox-sslasso')]:
        for constraint in ('none', 'soft_center', 'prox1e-3', 'prox1e-6', 'exact'):
            run = args.revision_root / 'results' / branch / 'zero_sum' / constraint / method / 'runs' / f'rep{args.replicate}_{method}_gompertz_n400_p150_nz12'
            record = verify_run(run)
            record['setting'] = constraint
            records.append(record)
            print(method, constraint, record['n_scalar_checks'], 'checks passed', flush=True)
    assert len({r['test_data_sha256'] for r in records}) == 1, 'Comparison cells use different test observations'
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / 'saved_statistics_checks.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    audit = dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 scope='One specified formal repetition in each of the ten zero-sum cells; numerical reporting only',
                 posterior_sampling_performed=False, n_verified_runs=len(records),
                 n_scalar_checks=sum(r['n_scalar_checks'] for r in records),
                 replicate=args.replicate, all_checks_passed=True, shared_test_data_verified=True)
    (args.outdir / 'saved_statistics_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
