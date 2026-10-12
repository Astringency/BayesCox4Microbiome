#!/usr/bin/env python3
"""Validate historical Cox-RHS baseline runs without changing their files.

The designated legacy HS launcher used prox's default lambda=1e-3 and
noncentered=true. Those two settings were omitted by the old metadata writer;
their provenance is explicitly recorded, rather than silently called verified
metadata. All other listed settings must match each reused run.
"""
import argparse
import json
from pathlib import Path
from audit_revision_legacy_rhs import COMMITS as UNSAFE_PARALLEL_HMC_COMMITS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--historical-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--fixed-baseline-manifest', help='Prefer an already resolved fixed-kernel baseline')
    parser.add_argument('--reps', type=int, default=50)
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--chains', type=int, default=4)
    args = parser.parse_args()
    root = Path(args.historical_root) / 'simulation/prox/n400_p50_nonzero6_weibull/runs'
    fixed_sources = None
    if args.fixed_baseline_manifest and Path(args.fixed_baseline_manifest).is_file():
        fixed = json.loads(Path(args.fixed_baseline_manifest).read_text())
        if fixed.get('status') != 'complete':
            raise ValueError('The fixed baseline manifest is not complete')
        fixed_sources = {int(Path(s).name.split('_')[0][3:]): Path(s) for s in fixed['source_runs']}
    expected = dict(n=400, p=50, nonzero_count=6, baseline='weibull', prior='cox-rhs',
                    p0=6, constraint='prox', sampler='hmc', hmc_n_leapfrog=100,
                    n_samples=args.samples, warmup=args.warmup, chains=args.chains,
                    use_car=True, sample_car_strength=True, car_weight_quantile=0.99,
                    car_precision=1, car_strength_prior_a=1, car_strength_prior_b=1,
                    rhs_tau_scale=1, rhs_local_scale=1, rhs_slab_df=4, rhs_slab_s2=2,
                    standardize=False, pseudocount=0)
    runs, errors, inferred = [], [], []
    for rep in range(1, args.reps + 1):
        run = (fixed_sources.get(rep) if fixed_sources is not None else
               root / f'rep{rep}_cox-rhs_weibull_n400_p50_nz6')
        if run is None:
            errors.append(f'missing fixed-kernel repetition: {rep}')
            continue
        needed = ['complete.txt', 'metadata.json', 'samples_wide.csv', 'posterior_summary.csv']
        if not all((run / name).is_file() for name in needed):
            errors.append(f'missing complete posterior output: {run}')
            continue
        metadata = json.loads((run / 'metadata.json').read_text())
        sidecar = run / 'execution_settings.json'
        if sidecar.is_file():
            metadata = dict(json.loads(sidecar.read_text())['settings'], **metadata)
        if metadata.get('git_commit_hash') in UNSAFE_PARALLEL_HMC_COMMITS:
            errors.append(f'{run.name}: historical parallel HMC target ownership is unverifiable; '
                          'retain for audit and use a fixed-kernel fit for primary comparisons')
        for key, value in dict(expected, seed=rep).items():
            if metadata.get(key) != value:
                errors.append(f'{run.name}: {key}={metadata.get(key)!r}, expected {value!r}')
        for key, value in dict(prox_sum0_lambda=1e-3, noncentered=True).items():
            if key not in metadata:
                inferred.append(dict(run=str(run), key=key, value=value,
                    provenance='historical run_simulation_hspriors.sh and analysis_common.jl defaults'))
            elif metadata[key] != value:
                errors.append(f'{run.name}: incompatible {key}={metadata[key]!r}')
        runs.append(str(run))
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(dict(reusable=not errors, expected=expected,
        reuse_check_scope='settings, output presence and known historical kernel exclusions; convergence must also be reported',
        source_runs=runs, legacy_script_provenance=inferred, errors=errors), indent=2))
    print(f'{len(runs)} {"fixed-kernel" if fixed_sources is not None else "historical"} baseline runs checked; reusable={not errors}')
    if errors:
        print('\n'.join(errors[:5]))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
