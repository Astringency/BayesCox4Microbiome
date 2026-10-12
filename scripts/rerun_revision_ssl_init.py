#!/usr/bin/env python3
"""Rerun the predeclared, eligible SSL failures in isolated output directories.

The default changes initialization only. Keep every completed rerun, including
failed convergence screens. Never retry a statistically disappointing fit.
"""
import argparse
import collections
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

from aggregate_revision_diagnostics import number, summarize_run, write_csv


MODEL_KEYS = [
    'n', 'p', 'nonzero_count', 'baseline', 'baseline_rate', 'baseline_shape',
    'censoring_upper', 'scenario', 'car_graph', 'seed', 'pseudocount',
    'standardize', 'tie_method', 'constraint', 'prox_sum0_lambda',
    'prox_l1_lambda', 'alpha_prior_kind', 'alpha_prior_shape', 'alpha_prior_scale',
    'sslasso_spike_scale', 'sslasso_slab_scale', 'sslasso_theta_a', 'sslasso_theta_b',
    'sampler', 'nuts_backend', 'warmup', 'n_samples', 'chains', 'thin',
    'target_accept', 'hmc_n_leapfrog', 'hmc_step_size', 'max_depth',
    'prox_block_step_size', 'prox_block_min_step_size', 'prox_block_max_step_size',
    'prox_block_adapt_every', 'prox_block_divergence_threshold', 'prox_block_metric',
    'prox_root_tol', 'prox_root_maxiter', 'use_car', 'sample_car_strength',
    'noncentered', 'weak_multiplier', 'null_correlation', 'depth_median', 'depth_log_sd',
]
CLI_KEYS = [k for k in MODEL_KEYS if k not in
            ('nonzero_count', 'seed', 'censoring_upper', 'n_samples')]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value(v):
    return str(v).lower() if isinstance(v, bool) else str(v)


def make_command(meta, output, julia, l1_lambda=None):
    rep = int(meta['replicate_id'])
    command = [julia, '-t', str(meta['chains']), '--startup-file=no', '--project=.',
               'scripts/run_simulation.jl', '--outdir', str(output), '--priors', 'prox-sslasso',
               '--reps', str(max(50, rep)), '--rep-ids', str(rep),
               '--nonzero', str(meta['nonzero_count']),
               '--seed', str(int(meta['seed']) - rep + 1),
               '--samples', str(meta['n_samples']), '--init', 'rand',
               '--multichains', 'true', '--approx', 'false',
               '--diagnose', 'false', '--progress', 'false', '--resume', 'true',
               '--sample-thin', '1', '--first-rep-sample-thin', '1',
               '--save-wide-csv', 'true', '--save-jld2', 'false', '--p0', 'truth']
    for key in CLI_KEYS:
        v = l1_lambda if key == 'prox_l1_lambda' and l1_lambda is not None else meta.get(key)
        if v is not None:
            command.extend(['--' + key.replace('_', '-'), value(v)])
    return command


def build_plan(args):
    source = args.revision_root / 'results/diagnostic_aggregates_v2/diagnostics_by_run.csv'
    selected = [r for r in read_csv(source) if r['primary_eligible'] == 'True'
                and r['method'] == 'prox-sslasso' and r['convergence_screen_pass'] != 'True']
    assert len(selected) == 391, f'Unexpected audited population: {len(selected)}'
    assert not (args.root / 'plan.json').exists(), 'Existing plan is immutable; use its run/report actions.'
    jobs = []
    for row in selected:
        old = Path(row['source'])
        meta = json.loads((old / 'metadata.json').read_text())
        assert meta['chains'] == 4 and meta['warmup'] == 2000 and meta['n_samples'] == 5000
        assert meta['sampler'] == 'hmc' and meta['hmc_n_leapfrog'] == 100
        group = (f"zero_{meta['constraint']}" +
                 (f"_{meta['prox_sum0_lambda']:g}" if meta['constraint'] == 'prox' else '')) \
            if meta['scenario'] == 'baseline' else meta['scenario']
        tag = f"{group}_rep{int(meta['replicate_id']):02d}"
        output = args.root / 'fits' / tag
        assert not output.is_relative_to(args.revision_root)
        jobs.append(dict(tag=tag, group=group, source=str(old), metadata=meta,
            original_diagnostics=row, output=str(output), new_run=str(output / 'runs' / old.name),
            original_hashes={f: digest(old/f) for f in
                             ['metadata.json', 'metrics.csv', 'rank_diagnostics.csv', 'posterior_summary.csv']},
            command=make_command(meta, output, args.julia, args.l1_lambda)))
    assert len({j['tag'] for j in jobs}) == len(jobs)
    # Interleave settings, so early results cover the entire requested matrix.
    groups = collections.defaultdict(list)
    for job in sorted(jobs, key=lambda j: int(j['metadata']['replicate_id'])):
        groups[job['group']].append(job)
    ordered = []
    while any(groups.values()):
        for group in sorted(groups):
            if groups[group]:
                ordered.append(groups[group].pop(0))
    plan = dict(created_utc=now(), selection_file=str(source), selection_sha256=digest(source),
                intended_change='init=rand using the repaired implementation',
                l1_lambda_override=args.l1_lambda, source_project=str(args.project),
                source_manifest_sha256=digest(args.project/'source_manifest.json'),
                n_jobs=len(ordered), n_original_eligible_ssl=650,
                n_original_passes_not_rerun=259, jobs=ordered)
    dump(args.root/'plan.json', plan)
    write_csv(args.root/'reports/selected_runs.csv', [dict(tag=j['tag'], group=j['group'],
        replicate=j['metadata']['replicate_id'], source=j['source'], output=j['output'],
        original_beta_rhat_max=j['original_diagnostics']['beta_rhat_max']) for j in ordered])
    print(json.dumps(dict(planned=len(ordered), groups=dict(collections.Counter(j['group'] for j in ordered)))))


def csv_data_match(old_path, new_path):
    old_hash, new_hash = digest(old_path), digest(new_path)
    if old_hash == new_hash:
        return dict(byte_identical=True, max_absolute_difference=0.)
    old, new = read_csv(old_path), read_csv(new_path)
    assert len(old) == len(new) and list(old[0]) == list(new[0]), 'Data dimensions differ'
    max_diff = 0.
    for x, y in zip(old, new):
        for key in x:
            a, b = float(x[key]), float(y[key])
            assert math.isfinite(a) and math.isfinite(b)
            assert a == b if key == 'event' else math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (key, a, b)
            max_diff = max(max_diff, abs(a-b))
    return dict(byte_identical=False, max_absolute_difference=max_diff)


def verify_job(job, plan):
    old, new = Path(job['source']), Path(job['new_run'])
    assert (new/'complete.txt').is_file(), 'Pipeline completion marker missing'
    for name, sha in job['original_hashes'].items():
        assert digest(old/name) == sha, f'Original file changed: {old/name}'
    meta = json.loads((new/'metadata.json').read_text())
    assert meta['init'] == 'rand'
    for key in MODEL_KEYS:
        expected = job['metadata'].get(key)
        if key == 'prox_l1_lambda' and plan['l1_lambda_override'] is not None:
            expected = plan['l1_lambda_override']
        assert meta.get(key) == expected, (key, expected, meta.get(key))
    test_data = csv_data_match(job['metadata']['test_data'], meta['test_data'])
    train_name = Path(job['metadata']['test_data']).name
    old_train = old.parent.parent/'training_data'/train_name
    new_train = new.parent.parent/'training_data'/train_name
    training = csv_data_match(old_train, new_train) if old_train.exists() else None
    dump(new/'rerun_verification.json', dict(verified_utc=now(), old_source=str(old),
         original_summary_files_unchanged=True, all_model_and_sampler_settings_match=True,
         test_data=test_data, training_data=training,
         training_identity_basis='saved training CSV' if training else 'unchanged generator, seed, DGP parameters and matched test cohort'))


def report(args, plan):
    pairs, pending, errors = [], [], []
    for job in plan['jobs']:
        new = Path(job['new_run'])
        if not (new/'rerun_verification.json').exists():
            pending.append(job['tag'])
            continue
        diag = summarize_run(new, new)
        old_metrics = next(r for r in read_csv(Path(job['source'])/'metrics.csv') if r['estimate']=='mean')
        metrics = next(r for r in read_csv(new/'metrics.csv') if r['estimate']=='mean')
        row = dict(tag=job['tag'], group=job['group'], replicate=job['metadata']['replicate_id'],
                   original_source=job['source'], new_source=str(new))
        for key in ['beta_rhat_max','beta_ess_bulk_min','beta_ess_tail_min','gamma_rhat_max',
                    'gamma_ess_bulk_min','gamma_ess_tail_min','convergence_screen_pass','n_divergent_sample']:
            row['old_'+key] = job['original_diagnostics'].get(key)
            row['new_'+key] = diag.get(key)
        for key in ['MSE','Cindex_test','coverage_95_nonzero']:
            row['old_'+key], row['new_'+key] = number(old_metrics.get(key)), number(metrics.get(key))
        row['new_seconds'] = diag['total_seconds']
        pairs.append(row)
    write_csv(args.root/'reports/paired_comparison.csv', pairs)
    groups = []
    for group in sorted({j['group'] for j in plan['jobs']}):
        rr = [r for r in pairs if r['group']==group]
        row = dict(group=group, selected=sum(j['group']==group for j in plan['jobs']),
                   completed=len(rr), beta_pass=sum(r['new_convergence_screen_pass'] for r in rr))
        for key in ['MSE','Cindex_test','beta_rhat_max','beta_ess_bulk_min','beta_ess_tail_min']:
            for prefix in ['old','new']:
                vals = [number(r[prefix+'_'+key]) for r in rr]
                vals = [v for v in vals if v is not None]
                row[prefix+'_'+key+'_mean'] = sum(vals)/len(vals) if vals else None
        groups.append(row)
    write_csv(args.root/'reports/group_comparison.csv', groups)
    summary = dict(updated_utc=now(), selected=len(plan['jobs']), completed=len(pairs),
        pending=len(pending), beta_pass=sum(r['new_convergence_screen_pass'] for r in pairs),
        beta_fail=sum(not r['new_convergence_screen_pass'] for r in pairs),
        l1_lambda_override=plan['l1_lambda_override'],
        population='Previously failed, kernel-eligible Prox-SSLASSO simulations only; not an unbiased all-replicate success rate.',
        original_files_preserved=True, groups=groups)
    dump(args.root/'reports/summary.json', summary)
    return summary


def run(args, plan):
    assert digest(args.project/'source_manifest.json') == plan['source_manifest_sha256']
    for row in json.loads((args.project/'source_manifest.json').read_text())['files']:
        assert digest(args.project/row['path']) == row['sha256'], row['path']
    logs = args.root/'logs'; logs.mkdir(exist_ok=True)
    lock = (logs/'pool.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected = set(args.tags.split(',')) if args.tags else None
    jobs = [j for j in plan['jobs'] if selected is None or j['tag'] in selected]
    if selected:
        assert selected == {j['tag'] for j in jobs}, 'Unknown job tag'
    pending, completed = [], []
    for job in jobs:
        if (Path(job['new_run'])/'complete.txt').exists():
            verify_job(job, plan)
            completed.append(job['tag'])
        else:
            pending.append(job)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', GKSwstype='100')
    active, failed = {}, []
    state = dict(started_utc=now(), pid=os.getpid(), workers=args.workers, selected=len(jobs))
    last_report = 0.
    while pending or active:
        while pending and len(active) < args.workers and not failed:
            job = pending.pop(0)
            handle = (logs/(job['tag']+'.log')).open('a')
            dump(logs/(job['tag']+'.command.json'), dict(cwd=str(args.project), argv=job['command']))
            process = subprocess.Popen(job['command'], cwd=args.project, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            active[job['tag']] = (job, process, handle, time.time())
        for tag, (job, process, handle, started) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            detail = dict(exit_code=code, seconds=time.time()-started, completed_utc=now())
            try:
                assert code == 0, f'Julia exited {code}'
                verify_job(job, plan)
                completed.append(tag)
            except Exception as exc:
                detail['error'] = repr(exc)
                failed.append(dict(tag=tag, **detail))
            dump(logs/(tag+'.exit.json'), detail)
            del active[tag]
            print(json.dumps(dict(tag=tag, **detail)), flush=True)
        state.update(updated_utc=now(), status='running', completed=completed, failed=failed,
                     remaining=len(pending), active=[dict(tag=k,pid=v[1].pid,elapsed=time.time()-v[3]) for k,v in active.items()])
        dump(logs/'pool_state.json', state)
        if time.time()-last_report > 60:
            report(args, plan); last_report=time.time()
        if failed and not active:
            break
        time.sleep(2)
    state.update(status='failed' if failed else 'complete', updated_utc=now())
    dump(logs/'pool_state.json', state)
    report(args, plan)
    if failed:
        raise SystemExit('Execution/verification failure; inspect logs. No convergence-based retries were performed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan','run','report'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--revision-root', type=Path)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--julia', default='julia')
    parser.add_argument('--l1-lambda', type=float)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--tags', help='Optional predeclared tags for a pilot; all jobs remain in the plan.')
    args = parser.parse_args()
    args.root, args.project = args.root.resolve(), args.project.resolve()
    assert 1 <= args.workers <= 28
    args.root.mkdir(parents=True, exist_ok=True)
    if args.action == 'plan':
        assert args.revision_root is not None
        build_plan(args)
    else:
        plan = json.loads((args.root/'plan.json').read_text())
        if args.action == 'run':
            run(args, plan)
        else:
            print(json.dumps(report(args, plan), indent=2))


if __name__ == '__main__':
    main()
