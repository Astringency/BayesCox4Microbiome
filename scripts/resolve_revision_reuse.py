#!/usr/bin/env python3
"""Replace only specifically identified invalid baseline repetitions in a manifest."""
import argparse
import csv
import json
from pathlib import Path
from audit_revision_legacy_rhs import COMMITS as UNSAFE_PARALLEL_HMC_COMMITS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original-manifest', required=True)
    p.add_argument('--repair-root', required=True)
    p.add_argument('--rep-ids', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    original = json.loads(Path(args.original_manifest).read_text())
    reps = {int(x) for x in args.rep_ids.split(',')}
    sources, replacements = [], []
    for source in original['source_runs']:
        rep = int(Path(source).name.split('_')[0][3:])
        if rep not in reps:
            metadata = json.loads((Path(source) / 'metadata.json').read_text())
            if metadata.get('git_commit_hash') in UNSAFE_PARALLEL_HMC_COMMITS:
                raise RuntimeError('Historical parallel HMC ownership cannot be certified: '
                    + source + '; use resolve_revision_fixed_baseline.py after the fixed fits finish')
            sources.append(source)
            continue
        replacement = Path(args.repair_root) / 'runs' / Path(source).name
        assert (replacement / 'complete.txt').is_file(), f'incomplete repair: {replacement}'
        meta = json.loads((replacement / 'metadata.json').read_text())
        sidecar = replacement / 'execution_settings.json'
        if sidecar.is_file():
            meta = dict(json.loads(sidecar.read_text())['settings'], **meta)
        for k, v in original['expected'].items():
            assert meta.get(k) == v, (replacement, k, meta.get(k), v)
        assert meta['seed'] == rep and meta['noncentered'] is True
        assert meta['prox_sum0_lambda'] == 1e-3
        with (replacement / 'rank_diagnostics.csv').open() as f:
            beta = [r for r in csv.DictReader(f) if r['parameters'].startswith('beta[')]
        rhat = max(float(r['rhat']) for r in beta)
        bulk = min(float(r['ess_bulk']) for r in beta)
        tail = min(float(r['ess_tail']) for r in beta)
        assert rhat <= 1.01 and bulk >= 400 and tail >= 400, (replacement, rhat, bulk, tail)
        sources.append(str(replacement))
        replacements.append(dict(original_source=source, replacement=str(replacement), replicate=rep,
            reason='Historical beta Rhat about 2.29 and bulk ESS about 5; refit with chain-local sampler targets',
            beta_rhat_max=rhat, beta_ess_bulk_min=bulk, beta_ess_tail_min=tail))
    assert len(replacements) == len(reps)
    out = dict(original, source_runs=sources, replacements=replacements,
        n_historical_reused=len(sources)-len(replacements), n_repaired=len(replacements),
        reuse_check_scope='Original settings checked; replacement posterior convergence checked')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f'{out["n_historical_reused"]} historical runs reused; {len(replacements)} repairs validated')


if __name__ == '__main__':
    main()
