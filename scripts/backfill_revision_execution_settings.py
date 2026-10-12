#!/usr/bin/env python3
"""Record known launch settings omitted by early revision metadata writers.

Only processes the controlled revision output hierarchy, never historical roots.
Values are from this revision's versioned launch commands and runner scripts.
The original metadata files are left unchanged.
"""
import argparse
import json
from pathlib import Path


def settings_for(parts):
    parts = list(parts)
    if parts and parts[0] in ('fixed_parallel', 'repairs', 'corrected_ssl'):
        parts.pop(0)
    if len(parts) < 4:
        return None
    suite = parts[0]
    if suite == 'lambda':
        method, lam = parts[1], float(parts[2])
    elif suite == 'p0':
        method, lam = parts[1], .001
    elif suite in ('scenarios', 'car'):
        method = parts[2] if suite == 'scenarios' else parts[1]
        lam = .001 if method == 'cox-rhs' else 1e-6
    elif suite == 'zero_sum':
        method = parts[2]
        lam = float(parts[1][4:]) if parts[1].startswith('prox') else .001
    else:
        return None
    return dict(prox_sum0_lambda=lam, noncentered=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--revision-results', required=True)
    args = p.parse_args()
    root = Path(args.revision_results).resolve()
    count = 0
    for f in root.rglob('metadata.json'):
        values = settings_for(f.relative_to(root).parts)
        if values is None:
            continue
        meta = json.loads(f.read_text())
        values = {k: v for k, v in values.items() if k not in meta}
        if not values:
            continue
        out = dict(source=str(f.parent), settings=values,
            provenance='Known revision_20261011 launch commands and versioned runner scripts; omitted by early metadata writer')
        (f.parent / 'execution_settings.json').write_text(json.dumps(out, indent=2))
        count += 1
    print(f'{count} execution-settings sidecars written; original metadata unchanged')


if __name__ == '__main__':
    main()
