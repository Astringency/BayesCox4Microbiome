#!/usr/bin/env python3
"""Resolve the fixed-kernel 50-run baseline without discarding mixing failures."""
import argparse
import csv
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-root', required=True)
    args = p.parse_args()
    root = Path(args.results_root)
    folders = [root / 'repairs/lambda/cox-rhs/1e-3/runs']
    folders += [root / f'fixed_parallel/baseline_shards/shard{s}/lambda/cox-rhs/1e-3/runs' for s in (1, 2)]
    sources = {}
    expected = dict(method_name='cox-rhs', n=400, p=50, nonzero_count=6, baseline='weibull',
                    constraint='prox', prox_sum0_lambda=.001, p0=6, n_samples=5000, warmup=2000,
                    chains=4, thin=1, sampler='hmc', hmc_n_leapfrog=100, use_car=True, noncentered=True,
                    scenario='baseline', rhs_local_scale=1.0, rhs_tau_scale=1.0, rhs_slab_df=4.0,
                    rhs_slab_s2=2.0, sample_car_strength=True, car_weight_quantile=.99, car_precision=1.0)
    for folder in folders:
        for path in folder.glob('*/metadata.json'):
            source = path.parent
            assert (source / 'complete.txt').is_file(), source
            meta = json.loads(path.read_text())
            sidecar = source / 'execution_settings.json'
            if sidecar.is_file():
                meta = dict(json.loads(sidecar.read_text())['settings'], **meta)
            for key, value in expected.items():
                assert meta.get(key) == value, (source, key, meta.get(key), value)
            rep = int(meta['replicate_id'])
            assert meta['seed'] == rep and rep not in sources
            sources[rep] = str(source)
    assert sorted(sources) == list(range(1, 51))
    original = json.loads((root / 'lambda/cox-rhs/1e-3/reuse_manifest.json').read_text())
    replacements = []
    for old in original['source_runs']:
        rep = int(Path(old).name.split('_')[0][3:])
        new = sources[rep]
        with (Path(new) / 'rank_diagnostics.csv').open() as f:
            beta = [r for r in csv.DictReader(f) if r['parameters'].startswith('beta[')]
        max_rhat = max(float(r['rhat']) for r in beta)
        min_bulk = min(float(r['ess_bulk']) for r in beta)
        min_tail = min(float(r['ess_tail']) for r in beta)
        replacements.append(dict(original_source=old, replacement=new, replicate=rep,
            reason='Historical parallel-target ownership cannot be verified; use fixed-kernel repetitions',
            beta_rhat_max=max_rhat, beta_ess_bulk_min=min_bulk, beta_ess_tail_min=min_tail,
            convergence_screen_pass=math.isfinite(max_rhat) and max_rhat <= 1.01 and min_bulk >= 400 and min_tail >= 400))
    result = dict(status='complete', source_runs=[sources[i] for i in range(1, 51)], expected=expected,
        replacements=replacements, n_historical_reused=0, n_fixed_existing_reused=2, n_fixed_new=48,
        note='All fixed-kernel repetitions retained, including any mixing-screen failure; no repeat-to-pass selection')
    output = root / 'resolved_fixed_rhs_baseline.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output)
    status_file = root / 'baseline_reuse_status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    status.update(status='complete_fixed_baseline', current_manifest=output.name,
                  n_completed_baseline_repetitions=50, n_fixed_existing_reused=2,
                  n_fixed_new=48, n_historical_reused=0)
    temporary = status_file.with_suffix('.tmp')
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(status_file)
    print('Fixed baseline resolved: 2 existing corrected fits reused and 48 new fits')


if __name__ == '__main__':
    main()
