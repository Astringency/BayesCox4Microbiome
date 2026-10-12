#!/usr/bin/env python3
"""Run disjoint S4 repetition partitions; each fit has four Julia chain threads.

Partition paths remain stable across resume. Successful fits are recovered by
run_simulation.jl. Scientific settings are fixed to the approved formal budget.
The root must be dedicated to S4, with no smoke fits or previous DGP outputs.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--julia', required=True)
    parser.add_argument('--partitions', type=int, default=4)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    assert 1 <= args.partitions <= 50 and args.workers >= 1
    project = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    logs = root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    manifest = dict(scenario='phylo_bm', dgp_version='phylo_bm_v1', repetitions=50,
        chains=4, warmup=2000, retained_per_chain=5000, partitions=args.partitions,
        graphs=['independent', 'correct', 'misspecified'], n=200, p=50, nonzero=6, n_test=1000)
    manifest_path = root / 'experiment.json'
    if manifest_path.is_file():
        assert json.loads(manifest_path.read_text()) == manifest, 'incompatible experiment/resume layout'
    else:
        assert not list((root / 'car').rglob('metadata.json')), 'unmanifested fits already exist'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    def run(graph, part):
        ids = list(range(part, 51, args.partitions))
        name = f'{graph}_part{part:02}'
        out = root / 'car' / f'part{part:02}'
        env = dict(os.environ, JULIA_BIN=args.julia, SCENARIO='S4', METHOD='cox-rhs',
            GRAPHS=graph, CAR_OUT=str(out), REVISION_OUT=str(root.parent),
            REP_IDS=','.join(map(str, ids)), REPS='50', CHAINS='4',
            WARMUP='2000', SAMPLES='5000', HMC_N_LEAPFROG='100')
        command = ['bash', str(project / 'scripts/run_car_revision.sh')]
        record = dict(command=command, project=str(project), graph=graph,
            repetitions=ids, output=str(out), environment={k:env[k] for k in
                ('JULIA_BIN','SCENARIO','METHOD','GRAPHS','CAR_OUT','REP_IDS','REPS','CHAINS','WARMUP','SAMPLES')})
        (logs / f'{name}.command.json').write_text(json.dumps(record, indent=2) + '\n')
        print('Starting', name, ids, flush=True)
        with (logs / f'{name}.log').open('a') as handle:
            code = subprocess.run(command, cwd=project, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        (logs / f'{name}.exit').write_text(str(code) + '\n')
        print('Finished', name, 'exit', code, flush=True)
        return code

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run, graph, part) for graph in manifest['graphs']
                for part in range(1, args.partitions + 1)]
        codes = [job.result() for job in jobs]
    if any(codes):
        raise SystemExit('S4 partitions failed; inspect logs and resume compatible fits')
    subprocess.run(['bash', str(project / 'scripts/refresh_s4_reports.sh'), str(root)], check=True)
    subprocess.run(['python3', str(project / 'scripts/verify_s4_results.py'),
                    '--root', str(root), '--require-complete'], check=True)


if __name__ == '__main__':
    main()
