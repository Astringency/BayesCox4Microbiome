#!/usr/bin/env python3
"""Read-only clinical comparison and nested-CV split/selection audit."""
import argparse
import json
import math
import statistics
from pathlib import Path

from aggregate_revision_diagnostics import extrema, number, read_csv, write_csv


def interval_selects(row):
    lo, hi = number(row.get('lower')), number(row.get('upper'))
    return None if lo is None or hi is None else lo > 0 or hi < 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rhs-root', required=True)
    p.add_argument('--ssl-root', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--outer-audit-root')
    args = p.parse_args()
    roots = {'cox-rhs': Path(args.rhs_root), 'prox-sslasso': Path(args.ssl_root)}
    summaries, folds, coefficients, clinical, ph, quality, audits, parameters = [], [], [], [], [], [], [], []
    paired_folds, paired_summaries = [], []
    for method, root in roots.items():
        arm_coeffs, arm_splits, arm_folds = {}, {}, {}
        for arm in ('unadjusted', 'adjusted'):
            directory = root / arm / method
            label = dict(method=method, adjustment=arm)
            config_path = directory / 'full_data_tuning/selected_config.json'
            reuse_path = directory / 'reuse_manifest.json'
            if config_path.is_file():
                chosen = json.loads(config_path.read_text())
                config_row = dict(**label, **chosen, parameter_source='full-data inner tuning; not test performance')
                fitted_path = directory / 'runs' / ('realdata_' + method) / 'metadata.json'
                if fitted_path.is_file():
                    fitted = json.loads(fitted_path.read_text())
                    config_row['prox_sum0_lambda'] = fitted.get('prox_sum0_lambda')
                    if method == 'cox-rhs':
                        config_row['p0_selection'] = 'automatic on full training data' if chosen.get('p0') is None else 'specified'
                        config_row['p0'] = fitted.get('p0')
                parameters.append(config_row)
            elif reuse_path.is_file():
                old = json.loads(reuse_path.read_text())['metadata']
                parameters.append(dict(**label, **{k: old.get(k) for k in
                    ('rhs_local_scale', 'rhs_slab_df', 'rhs_slab_s2', 'p0', 'prox_sum0_lambda')},
                    parameter_source='reused historical full-data fit; not the outer CV fold configuration'))
            cv = directory / 'nested_cv'
            summaries.extend(dict(r, **label) for r in read_csv(cv / 'nested_cv_summary.csv'))
            arm_folds[arm] = read_csv(cv / 'nested_cv_folds.csv')
            folds.extend(dict(r, **label) for r in arm_folds[arm])
            arm_coeffs[arm] = {r['variable']: r for r in read_csv(directory / 'coefficients.csv')}
            for row in read_csv(directory / 'clinical_coefficients.csv'):
                out = dict(row, **label)
                for key in ('mean', 'median', 'lower', 'upper'):
                    value = number(row.get(key))
                    # exp(E[log HR]) is not E[HR]. Quantiles transform exactly.
                    out['HR_' + ('exp_mean_log' if key == 'mean' else key)] = math.exp(value) if value is not None else None
                clinical.append(out)
            for path in (directory / 'PH').glob('*_ph.csv'):
                ph.extend(dict(r, **label) for r in read_csv(path))
            ranks = read_csv(directory / 'runs' / ('realdata_' + method) / 'rank_diagnostics.csv')
            if not ranks:
                ranks = read_csv(directory / 'historical_diagnostics/rank_diagnostics.csv')
            for kind, subset in [('beta', [r for r in ranks if r['parameters'].startswith('beta[')]),
                                 ('clinical', [r for r in ranks if r['parameters'].startswith('clinical[')]), ('all', ranks)]:
                if subset:
                    quality.append(dict(**label, stage='full_data', parameters=kind, n_parameters=len(subset),
                        rhat_max=extrema(subset, 'rhat', max), ess_bulk_min=extrema(subset, 'ess_bulk', min),
                        ess_tail_min=extrema(subset, 'ess_tail', min)))
            arm_splits[arm] = {}
            for row in arm_folds[arm]:
                if row['status'] != 'success':
                    continue
                fold = int(row['outer_fold'])
                local = cv / method / f'outer_{fold:02}'
                split = json.loads((local / 'split_indices.json').read_text())
                train, test = set(split['outer_train']), set(split['outer_test'])
                validation = split['inner_validation']
                assert not train & test
                assert sorted(i for v in validation for i in v) == sorted(train)
                arm_splits[arm][fold] = split
                inner = read_csv(local / 'inner_cv_metrics.csv')
                selected = [r for r in inner if int(r['config_index']) == int(row['selected_config_index'])]
                assert len(selected) == len(validation) and all(r['status'] == 'success' for r in selected)
                observed = sum(float(r['validation_Cindex']) for r in selected) / len(selected)
                assert abs(observed - float(row['inner_Cindex_mean'])) < 1e-12
                for kind, subset in [('all_candidates', inner), ('selected_candidate', selected)]:
                    quality.append(dict(**label, stage='inner_cv', outer_fold=fold, parameters=kind,
                        n_fits=len(subset), rhat_max=extrema(subset, 'fit_rhat_max', max),
                        ess_bulk_min=extrema(subset, 'fit_ess_bulk_min', min),
                        n_legacy_convergence_flag_false=sum(r.get('fit_converged', '').lower() == 'false' for r in subset)))
                audits.append(dict(**label, outer_fold=fold, disjoint_outer_train_test=True,
                    inner_validation_partitions_outer_training=True, selected_score_matches_inner=True,
                    outer_posterior_saved=(local / 'fit/samples_wide.csv').is_file()))
                fit_dir = local / 'fit'
                if args.outer_audit_root:
                    replay_dir = Path(args.outer_audit_root) / arm / method / f'outer_{fold:02}'
                    validation_path = replay_dir / 'validation.json'
                    if validation_path.is_file():
                        validation = json.loads(validation_path.read_text())
                        audits[-1]['replay_matches_cached_scores'] = validation['reproduces_cached_scores']
                        if validation['reproduces_cached_scores']:
                            fit_dir = replay_dir / 'fit'
                outer_ranks = read_csv(fit_dir / 'rank_diagnostics.csv')
                for kind, subset in [('beta', [r for r in outer_ranks if r['parameters'].startswith('beta[')]),
                                     ('clinical', [r for r in outer_ranks if r['parameters'].startswith('clinical[')])]:
                    if subset:
                        quality.append(dict(**label, stage='outer_cv', outer_fold=fold, parameters=kind,
                            n_parameters=len(subset), rhat_max=extrema(subset, 'rhat', max),
                            ess_bulk_min=extrema(subset, 'ess_bulk', min), ess_tail_min=extrema(subset, 'ess_tail', min)))
        for fold in arm_splits.get('unadjusted', {}).keys() & arm_splits.get('adjusted', {}).keys():
            assert arm_splits['unadjusted'][fold] == arm_splits['adjusted'][fold], 'adjusted/unadjusted splits differ'
        completed = {}
        for arm, rows in arm_folds.items():
            successful = [row for row in rows if row['status'] == 'success']
            completed[arm] = {int(row['outer_fold']): row for row in successful}
            assert len(completed[arm]) == len(successful), 'duplicate outer-fold result'
        for estimate in ('mean', 'median', 'MAP'):
            pairs = []
            for fold in sorted(completed['unadjusted'].keys() & completed['adjusted'].keys()):
                a_score = number(completed['adjusted'][fold].get('test_Cindex_' + estimate))
                u_score = number(completed['unadjusted'][fold].get('test_Cindex_' + estimate))
                if a_score is None or u_score is None:
                    continue
                pairs.append(dict(method=method, estimate=estimate, outer_fold=fold,
                    adjusted_Cindex=a_score, unadjusted_Cindex=u_score, difference=a_score - u_score))
            paired_folds.extend(pairs)
            values = [row['difference'] for row in pairs]
            paired_summaries.append(dict(method=method, estimate=estimate, n_paired_folds=len(values),
                expected_folds=5, all_five_folds_available=len(values) == 5,
                mean_difference=statistics.mean(values) if values else None,
                sd_difference=statistics.stdev(values) if len(values) > 1 else None,
                interpretation='adjusted minus unadjusted; descriptive fold variability, not an independent-fold confidence interval'))
        a, u = arm_coeffs.get('adjusted', {}), arm_coeffs.get('unadjusted', {})
        for variable in sorted(a.keys() & u.keys()):
            out = dict(method=method, variable=variable,
                       mean_change=float(a[variable]['mean']) - float(u[variable]['mean']))
            for arm, rows in [('adjusted', a), ('unadjusted', u)]:
                out.update({arm + '_' + key: rows[variable].get(key) for key in ('mean', 'median', 'lower', 'upper')})
                out[arm + '_selected_95'] = interval_selects(rows[variable])
            coefficients.append(out)
    out = Path(args.outdir)
    for name, rows in [('nested_cv_summary', summaries), ('nested_cv_folds', folds),
                       ('taxon_comparison', coefficients), ('clinical_coefficients', clinical),
                       ('ph_tests', ph), ('sampling_quality', quality), ('split_audit', audits),
                       ('full_data_parameters', parameters), ('paired_cv_folds', paired_folds),
                       ('paired_cv_comparison', paired_summaries)]:
        write_csv(out / (name + '.csv'), rows)
    print(json.dumps(dict(n_completed_arms=len(summaries), n_completed_outer_folds=len(audits),
                          n_paired_taxa=len(coefficients), all_available_split_checks_passed=True)))


if __name__ == '__main__':
    main()
