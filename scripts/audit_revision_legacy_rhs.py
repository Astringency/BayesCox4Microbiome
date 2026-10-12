#!/usr/bin/env python3
"""Conservative eligibility audit for historical parallel explicit HMC.

Source provenance establishes the shared binding defect, not which particular
saved repetition actually shared a target. Old metadata omits execution mode;
the historical launcher requests four parallel chains. A separate local probe
has observed two chain calls receiving the same mutable target at three commits.
Unverifiable historical fits remain available for audit, not primary comparisons.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

COMMITS = ['18deb2f11425996fb9304a222e28bec31e9c3462', 'fcaa4b95ca08dc144a239f1f20069a6d8f168edd',
    '90bba37732c96d25899c254845ab2ac35b7e4307', '2fd1805a77b3afa613b71706dc4e087ea76d9578',
    '97645e54198bea78f66a17e5507848853555fe72', 'fea74afacfda951f2392884f0ff577ed28a8c7eb',
    'c97ece2858dba3777d7847b87e1da7b8991d6984', '14bf204e2ab45a12b68b0361b090881990da653a',
    'af40118d1350897137de28c9140c2123dd73ccc7', '1402e3a9e381de8d00f1625e4fe3573f2259d3a4',
    '59bb5dedd6d8b09bad51349735013f4bbe2e7f38', 'a5f9ff573f8ef88bcd95f5d31ed7d10b6b3c847a',
    '6f3543986e85f1a5606bc745da50a607ea322ff8']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--ssl-audit', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    rules = json.loads(Path(args.ssl_audit).read_text())['rules']
    for commit in COMMITS:
        path = 'mainfuncs/samplers/hmc_sampler.jl'
        source = subprocess.check_output(['git', '-C', args.repo, 'show', commit + ':' + path], text=True, encoding='utf-8')
        source_function = source[source.index('function _sample_hmc_explicit('):]
        assert 'Threads.@threads for c in 1:chains' in source_function
        assert source_function.count('target = _make_explicit_nuts_target(') >= 3
        assert 'local target = _make_explicit_nuts_target(' not in source_function
        rules.append(dict(match=dict(method='cox-rhs', sampler='hmc', git_commit_hash=commit),
            reason='Historical threaded HMC has a shared boxed target binding; per-run target ownership/execution mode was not saved, so kernel validity cannot be certified from Rhat alone',
            evidence=[dict(path=path, sha256=hashlib.sha256(source.encode()).hexdigest())],
            exclusion_scope='conservative primary exclusion; does not assert every saved fit was corrupted'))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(schema=2, rules=rules), indent=2), encoding='utf-8')
    print(f'{len(rules)} recorded method/commit rules written to {out}')


if __name__ == '__main__':
    main()
