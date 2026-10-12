#!/usr/bin/env python3
"""Aggregate saved diagnostics without sampling or modifying source runs.

Historical lambda values absent from metadata are inferred only for documented
script/directory conventions, with a separate provenance column. Quality flags
remain visible; poor mixing is not silently removed from the all-eligible table.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from backfill_revision_execution_settings import settings_for


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def extrema(rows, key, function):
    vals = [v for r in rows if (v := number(r.get(key))) is not None]
    return function(vals) if vals else None


def write_csv(path, rows, columns=None):
    if not rows and columns is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def context(source, meta):
    method = meta.get('method_name', meta.get('prior_name'))
    lam, provenance = meta.get('prox_sum0_lambda'), 'metadata'
    if lam is None and meta.get('constraint') == 'prox':
        if 'prox1e-6' in source.parts:
            lam, provenance = 1e-6, 'directory name + constraint script'
        elif method == 'cox-rhs' and 'revision_20261011' not in source.parts:
            lam, provenance = 1e-3, 'historical HS runner uses default lambda'
        elif method == 'prox-sslasso':
            lam, provenance = None, 'unrecorded; current shell script is insufficient evidence for a past run'
    if meta.get('constraint') != 'prox':
        lam, provenance = None, 'not applicable'
    return dict(source=str(source), method=method, n=meta.get('n'), p=meta.get('p'),
        nonzero=meta.get('nonzero_count'), baseline=meta.get('baseline'),
        scenario=meta.get('scenario', 'baseline'), constraint=meta.get('constraint'),
        prox_sum0_lambda=lam, lambda_provenance=provenance,
        p0=meta.get('p0') if method == 'cox-rhs' else None,
        car_graph=meta.get('car_graph', 'abundance') if meta.get('use_car') else 'independent',
        replicate=meta.get('replicate_id'), saved_sample_thin=meta.get('saved_sample_thin'))


def summarize_run(source, diagnostics, revision_root=None):
    meta_path = source / 'metadata.json'
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    inferred_revision_settings = False
    if revision_root is not None and source.is_relative_to(revision_root):
        launch_settings = settings_for(meta_path.relative_to(revision_root).parts)
        if launch_settings:
            inferred_revision_settings = 'prox_sum0_lambda' not in meta
            meta = dict(launch_settings, **meta)
    settings = source / 'execution_settings.json'
    if settings.is_file():
        meta = dict(json.loads(settings.read_text())['settings'], **meta)
    recovered = diagnostics / 'recovered_settings.json'
    if recovered.is_file():
        r = json.loads(recovered.read_text())
        if r.get('verified'):
            meta['prox_sum0_lambda'] = r['prox_sum0_lambda']
    if 'nonzero_count' not in meta or meta.get('method_name') not in ('cox-rhs', 'prox-sslasso'):
        return None
    row = context(source, meta)
    row['git_commit_hash'] = meta.get('git_commit_hash')
    row['sampler'] = meta.get('sampler')
    if settings.is_file() or inferred_revision_settings:
        row['lambda_provenance'] = 'versioned revision runner; execution_settings.json'
    if inferred_revision_settings and not settings.is_file():
        row['lambda_provenance'] = 'versioned revision runner and controlled output hierarchy'
    if recovered.is_file() and r.get('verified'):
        row['lambda_provenance'] = r['provenance']
    ranks = read_csv(diagnostics / 'rank_diagnostics.csv')
    beta = [r for r in ranks if r.get('parameters', '').startswith('beta[')]
    gamma = [r for r in ranks if r.get('parameters', '').startswith('gamma[')]
    for label, subset in [('beta', beta), ('gamma', gamma), ('all', ranks)]:
        row[label + '_rhat_max'] = extrema(subset, 'rhat', max)
        row[label + '_ess_bulk_min'] = extrema(subset, 'ess_bulk', min)
        row[label + '_ess_tail_min'] = extrema(subset, 'ess_tail', min)
        row[label + '_n_rhat_unavailable'] = sum(number(r.get('rhat')) is None for r in subset)
    row['n_beta_parameters'] = len(beta)
    row['convergence_screen_pass'] = bool(len(beta) == meta.get('p') and
        row['beta_n_rhat_unavailable'] == 0 and row['beta_rhat_max'] <= 1.01 and
        row['beta_ess_bulk_min'] is not None and row['beta_ess_bulk_min'] >= 400 and
        row['beta_ess_tail_min'] is not None and row['beta_ess_tail_min'] >= 400)
    row['primary_eligible'], row['exclusion_reason'] = True, ''
    validation = source / 'validation_status.json'
    if validation.is_file():
        status = json.loads(validation.read_text())
        row['primary_eligible'] = status.get('eligible_for_primary_summary', True)
        row['exclusion_reason'] = status.get('reason', json.dumps(status)) if not row['primary_eligible'] else ''
    dpath = source / 'diagnostics.json'
    d = json.loads(dpath.read_text()) if dpath.is_file() else {}
    row['total_seconds'] = number(d.get('total_seconds', meta.get('runtime')))
    row['runtime_context'] = ('assisted_fresh_julia_process' if '/zero_tail_assist_staging/' in str(meta.get('test_data', '')).replace('\\', '/')
        else 'original_revision_worker' if revision_root is not None and source.is_relative_to(revision_root)
        else 'historical_saved_runtime')
    transition = read_csv(diagnostics / 'transition_summary.csv')
    if not transition:
        transition = read_csv(source / 'transition_summary.csv')
    sampling = [r for r in transition if r['phase'] == 'sample']
    row['transition_diagnostics_available'] = bool(sampling)
    row['n_divergent_sample'] = sum(int(r['n_divergent']) for r in sampling) if sampling else None
    row['mean_acceptance_probability'] = statistics.mean(float(r['mean_acceptance_probability']) for r in sampling) if sampling else None
    mixing = read_csv(diagnostics / 'selection_mixing.csv')
    row['gamma_constant_chain_fraction'] = statistics.mean(r['constant_chain'].lower() == 'true' for r in mixing) if mixing else None
    row['gamma_switch_rate_mean'] = statistics.mean(float(r['switch_rate']) for r in mixing if number(r['switch_rate']) is not None) if mixing else None
    zero = [r for r in read_csv(diagnostics / 'zero_sum_summary.csv') if r['chain'] == 'pooled']
    if zero:
        for k, v in zero[0].items():
            if k not in ('chain', 'p'):
                row['sum_beta_' + k] = number(v)
    return row


GROUPS = ['method', 'n', 'p', 'nonzero', 'baseline', 'scenario', 'constraint',
          'prox_sum0_lambda', 'p0', 'car_graph']


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        if r['primary_eligible']:
            groups[tuple(r.get(k) for k in GROUPS)].append(r)
    result = []
    for key, values in groups.items():
        for selection in ('all_eligible', 'convergence_screen_pass'):
            subset = values if selection == 'all_eligible' else [r for r in values if r['convergence_screen_pass']]
            out = dict(zip(GROUPS, key))
            out.update(selection=selection, n_replicates=len(subset), n_available=len(values),
                n_convergence_screen_pass=sum(r['convergence_screen_pass'] for r in values))
            out['n_assisted_fresh_julia_process'] = sum(r.get('runtime_context') == 'assisted_fresh_julia_process' for r in subset)
            metric_keys = [k for k in values[0] if k.startswith('sum_beta_') or
                k in ('beta_rhat_max', 'beta_ess_bulk_min', 'beta_ess_tail_min', 'total_seconds',
                      'mean_acceptance_probability', 'n_divergent_sample', 'gamma_switch_rate_mean',
                      'gamma_constant_chain_fraction')]
            for metric in metric_keys:
                vals = [v for r in subset if (v := number(r.get(metric))) is not None]
                out[metric + '_n'] = len(vals)
                out[metric + '_mean'] = statistics.mean(vals) if vals else None
                out[metric + '_sd'] = statistics.stdev(vals) if len(vals) > 1 else None
            result.append(out)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--historical-diagnostics')
    p.add_argument('--new-root')
    p.add_argument('--outdir', required=True)
    p.add_argument('--replacement-manifest')
    p.add_argument('--kernel-audit', help='Static source audit with affected recorded commits')
    args = p.parse_args()
    sources = {}
    if args.historical_diagnostics:
        for r in read_csv(Path(args.historical_diagnostics) / 'run_inventory.csv'):
            if r['status'] == 'complete':
                sources[Path(r['source'])] = Path(r['output_dir'])
    if args.new_root:
        for f in Path(args.new_root).rglob('metadata.json'):
            if ('historical_diagnostics' not in f.parts and (f.parent / 'rank_diagnostics.csv').is_file()
                    and (f.parent / 'complete.txt').is_file()):
                sources[f.parent] = f.parent
    rows, errors = [], []
    kernel_rules = []
    if args.kernel_audit:
        kernel_rules = json.loads(Path(args.kernel_audit).read_text())['rules']
    replacements = {}
    if args.replacement_manifest:
        manifest = json.loads(Path(args.replacement_manifest).read_text())
        replacements = {r['original_source']: r for r in manifest['replacements']}
    for source, diag in sources.items():
        try:
            row = summarize_run(source, diag, Path(args.new_root) if args.new_root else None)
            if row:
                for rule in kernel_rules:
                    if all(row.get(k) == v for k, v in rule['match'].items()):
                        row['primary_eligible'] = False
                        row['exclusion_reason'] = rule['reason']
                        if row['prox_sum0_lambda'] is None and row['constraint'] == 'prox':
                            row['prox_sum0_lambda'] = rule.get('launcher_prox_sum0_lambda')
                            row['lambda_provenance'] = 'launcher at recorded commit; kernel-invalid legacy result'
                if str(source) in replacements:
                    replacement = replacements[str(source)]
                    row['primary_eligible'] = False
                    row['exclusion_reason'] = replacement['reason'] + '; replacement=' + replacement['replacement']
                rows.append(row)
        except Exception as e:
            errors.append(dict(source=str(source), error=str(e)))
    outdir = Path(args.outdir)
    write_csv(outdir / 'diagnostics_by_run.csv', rows)
    write_csv(outdir / 'diagnostics_by_setting.csv', aggregate(rows))
    write_csv(outdir / 'aggregation_errors.csv', errors, columns=['source', 'error'])
    print(json.dumps(dict(n_runs=len(rows), n_errors=len(errors),
        n_excluded=sum(not r['primary_eligible'] for r in rows),
        n_beta_convergence_pass=sum(r['convergence_screen_pass'] for r in rows))))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
