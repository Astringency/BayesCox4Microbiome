#!/usr/bin/env python3
"""Produce inspectable source evidence for the three recorded legacy SSL commits.

Run in the original Git checkout, then pass the JSON to the read-only diagnostic
aggregator. No historical metadata or posterior file is modified. This narrowly
audits explicit HMC at known commits; it does not certify other kernels.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

COMMITS = (
    'fcaa4b95ca08dc144a239f1f20069a6d8f168edd',
    '6f3543986e85f1a5606bc745da50a607ea322ff8',
    'af40118d1350897137de28c9140c2123dd73ccc7',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    rules = []
    for commit in COMMITS:
        paths = ['mainfuncs/targets/cox_targets.jl', 'mainfuncs/samplers/hmc_sampler.jl',
                 'scripts/run_simulation_proxpriors.sh']
        sources = {p: subprocess.check_output(['git', '-C', args.repo, 'show', commit + ':' + p],
                    text=True, encoding='utf-8') for p in paths}
        target, sampler, launcher = [sources[p] for p in paths]
        assert 'gamma_ref = gamma isa Vector{Bool} ? gamma : collect(Bool, gamma)' in target
        start = sampler.index('gamma = trues(p)')
        target_at = sampler.index('target = CoxProxSSLassoTarget(', start)
        loop_at = sampler.index('for iter in 1:total_iter', target_at)
        update_at = sampler.index('gamma[j] = rand(rng)', loop_at)
        assert start < target_at < loop_at < update_at
        assert 'prox_ssl_block_hmc_step!(u, target, hmc_state' in sampler[loop_at:update_at]
        assert 'target.gamma' not in sampler[loop_at:update_at]
        match = re.search(r'--prox-sum0-lambda\s+([0-9.eE+-]+)', launcher)
        assert match
        rules.append(dict(match=dict(method='prox-sslasso', sampler='hmc', git_commit_hash=commit),
            reason='Recorded HMC source copies BitVector gamma into the continuous target; Gibbs updates do not reach that target',
            launcher_prox_sum0_lambda=float(match.group(1)),
            evidence=[dict(path=p, sha256=hashlib.sha256(s.encode()).hexdigest()) for p, s in sources.items()],
            source_checks=dict(bitvector_initialization=True, constructor_copies=True,
                continuous_target_created_before_gibbs_loop=True, no_target_indicator_refresh=True),
            caveat='Saved log density was evaluated with current gamma and cannot alone validate the transition kernel'))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(schema=1, rules=rules), indent=2), encoding='utf-8')
    print(f'{len(rules)} recorded source commits audited; wrote {out}')


if __name__ == '__main__':
    main()
