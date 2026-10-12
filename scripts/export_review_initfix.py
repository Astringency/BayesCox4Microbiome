#!/usr/bin/env python3
"""Export and verify the completed initialization reruns, without sampling."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from aggregate_revision_diagnostics import number, read_csv, summarize_run, write_csv
from aggregate_revision_metrics import METRICS
from verify_t1d_saved_cindex import cindex


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_metrics(source):
    meta = json.loads((source / 'metadata.json').read_text())
    assert not meta['standardize'] and meta['pseudocount'] == 0
    p, nz = meta['p'], meta['nonzero_count']
    summary = read_csv(source / 'posterior_summary.csv')
    summary.sort(key=lambda r: int(r['variable'].split('[')[1][:-1]))
    beta = [float(r['true_beta']) for r in summary]
    assert len(beta) == p and sum(b != 0 for b in beta) == nz
    covered = [float(r['lower']) <= b <= float(r['upper']) for r, b in zip(summary, beta)]
    selected = [float(r['lower']) > 0 or float(r['upper']) < 0 for r in summary]
    assert all(float(r['lower']) <= float(r['upper']) for r in summary)
    cover = dict(coverage_95_all=sum(covered)/p,
                 coverage_95_nonzero=sum(c for c, b in zip(covered, beta) if b != 0)/nz,
                 coverage_95_zero=sum(c for c, b in zip(covered, beta) if b == 0)/(p-nz))
    fp = sum(s and b == 0 for s, b in zip(selected, beta))
    tp = sum(s and b != 0 for s, b in zip(selected, beta))
    fd = read_csv(source / 'false_discovery.csv')[0]
    assert all(int(fd[k]) == v for k, v in dict(FP=fp, TP=tp, FN=nz-tp, TN=p-nz-fp).items())
    assert abs(float(fd['FDR']) - fp/max(tp+fp, 1)) < 1e-12
    test = Path(meta['test_data'])
    train = test.parent.parent / 'training_data' / test.name
    cohorts = [('train', train), ('test', test)] if train.exists() else [('test', test)]
    tables = {label: read_csv(path) for label, path in cohorts}
    matrices = {label: [[float(r[f'x{j}']) for j in range(1, p+1)] for r in table]
                for label, table in tables.items()}
    pc = min(v for row in matrices['train'] for v in row if v > 0)/2 if 'train' in matrices else None
    designs = {}
    for label, matrix in matrices.items():
        design = []
        for values in matrix:
            assert pc is not None or all(v > 0 for v in values)
            replaced = [v if v > 0 else pc for v in values]
            total = sum(replaced)
            design.append([math.log(v/total) for v in replaced])
        designs[label] = design
    errors = []
    for row in read_csv(source / 'metrics.csv'):
        estimate = [float(r[row['estimate'].lower()]) for r in summary]
        computed = dict(cover, MSE=sum((a-b)**2 for a, b in zip(estimate, beta))/p)
        for label, design in designs.items():
            table = tables[label]
            risk = [math.fsum(a*b for a, b in zip(z, estimate)) for z in design]
            computed['Cindex_'+label] = cindex([float(r['time']) for r in table],
                                             [int(r['event']) for r in table], risk)[0]
        for metric, value in computed.items():
            error = abs(value-float(row[metric])); errors.append(error)
            assert error < 1e-10, (source, row['estimate'], metric, value, row[metric])
    return dict(source=str(source), scenario=meta['scenario'], replicate=meta['replicate_id'],
                n_score_checks=3*len(cohorts), n_estimates=3, max_abs_difference=max(errors),
                coverage_and_fdp_match=True, training_cindex_verified=train.exists())


def export(root, out):
    plan = json.loads((root/'plan.json').read_text())
    state = json.loads((root/'logs/pool_state.json').read_text())
    assert state['status'] == 'complete' and not state['failed']
    assert len(state['completed']) == plan['n_jobs'] == 391
    frozen = root/'code/source_manifest.json'
    assert sha(frozen) == plan['source_manifest_sha256']
    for row in json.loads(frozen.read_text())['files']:
        assert sha(root/'code'/row['path']) == row['sha256'], row['path']
    diagnostics, metrics, numerical, provenance = [], [], [], []
    for i, job in enumerate(plan['jobs'], 1):
        source = Path(job['new_run']); old = Path(job['source'])
        assert (source/'complete.txt').is_file()
        verification = json.loads((source/'rerun_verification.json').read_text())
        assert verification['all_model_and_sampler_settings_match']
        assert verification['original_summary_files_unchanged']
        for name, expected in job['original_hashes'].items():
            assert sha(old/name) == expected, old/name
        diag = summarize_run(source, source)
        diag.update(original_source=str(old), result_version='initfix_random', init='rand',
                    replacement_tag=job['tag'], runtime_context='initialization_rerun_fresh_process',
                    source_manifest_sha256=plan['source_manifest_sha256'])
        ranks = read_csv(source/'rank_diagnostics.csv')
        for label in ('beta', 'gamma'):
            subset = [r for r in ranks if r['parameters'].startswith(label+'[')]
            diag[label+'_n_parameters'] = len(subset)
            for metric in ('rhat', 'ess_bulk', 'ess_tail'):
                diag[label+'_n_'+metric+'_unavailable'] = sum(number(r[metric]) is None for r in subset)
        diagnostics.append(diag)
        fd = read_csv(source/'false_discovery.csv')[0]
        saved_metrics = read_csv(source/'metrics.csv')
        assert {m['estimate'] for m in saved_metrics} == {'mean', 'median', 'MAP'}
        for m in saved_metrics:
            values = dict(m, **{k: fd[k] for k in ('TP', 'FP', 'TN', 'FN')}, empirical_FDP=fd['FDR'])
            metrics.append(dict(diag, estimate=m['estimate'], n_test=m['n_test'], test_seed=m['test_seed'],
                                **{k: number(values.get(k)) for k in METRICS}))
        numerical.append(audit_metrics(source))
        provenance.append(dict(tag=job['tag'], original_source=str(old), source=str(source),
            metadata_sha256=sha(source/'metadata.json'), metrics_sha256=sha(source/'metrics.csv'),
            diagnostics_sha256=sha(source/'rank_diagnostics.csv'),
            posterior_summary_sha256=sha(source/'posterior_summary.csv'),
            verification_sha256=sha(source/'rerun_verification.json')))
        if i % 25 == 0:
            print(f'Exported and independently verified {i}/391', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'diagnostics_by_run.csv', diagnostics)
    write_csv(out/'metrics_by_run.csv', metrics)
    write_csv(out/'numerical_audit_by_run.csv', numerical)
    write_csv(out/'source_checksums.csv', provenance)
    audit = dict(all_metrics_match=True, verified_runs=len(numerical),
        verified_cindices=sum(r['n_score_checks'] for r in numerical),
        max_abs_difference=max(r['max_abs_difference'] for r in numerical),
        original_files_unchanged=True, frozen_source_verified=True, posterior_resampled=False,
        limitation='Baseline train CSVs were not saved; test C-index, coefficient MSE/coverage/FDP were verified for all runs; train C-index additionally verified where training CSV exists.')
    (out/'numerical_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.outdir)
