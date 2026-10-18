#!/usr/bin/env python3
"""Paired null-information calibration of the Cox-RHS global scale.

Uses the frozen review implementation and the saved training cohorts. No test
outcomes, posterior estimates, or true coefficients enter the scale calculation.
The known signal count p0=6 matches the existing sensitivity experiment.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


def calibrate(path, p=50, p0=6):
    rows = list(csv.DictReader(path.open()))
    rows.sort(key=lambda r: float(r['time']), reverse=True)
    assert len({r['time'] for r in rows}) == len(rows), 'This experiment assumes untied simulated times.'
    sums, squares, information = [0.] * p, [0.] * p, [0.] * p
    events = 0
    for n, row in enumerate(rows, 1):
        for j in range(p):
            z = math.log(float(row[f'x{j+1}']))
            sums[j] += z
            squares[j] += z * z
        if int(float(row['event'])):
            events += 1
            for j in range(p):
                information[j] += max(0., squares[j] / n - (sums[j] / n)**2)
    roots = [math.sqrt(x) for x in information]
    def expected(tau):
        return sum(tau * x / (1 + tau * x) for x in roots)
    lo, hi = 0., 1.
    while expected(hi) < p0:
        hi *= 2
    for _ in range(90):
        mid = (lo + hi) / 2
        if expected(mid) < p0:
            lo = mid
        else:
            hi = mid
    tau = (lo + hi) / 2
    default = p0 / ((p - p0) * math.sqrt(events))
    return dict(n_events=events, tau_default=default, tau_information=tau,
                multiplier=tau/default, information_diagonal=information,
                mean_risk_variance=statistics.mean(information)/events,
                min_risk_variance=min(information)/events,
                max_risk_variance=max(information)/events,
                expected_signal_count=expected(tau),
                training_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--julia', required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for rep in range(1, 51):
        batch = 1 + ((rep - 1) // 5) * 5
        source = args.source / f'batch{batch:02d}/p0/soft_center/cox-rhs/6'
        train = source / f'training_data/rep{rep}_weibull_n400_p50_nz6.csv'
        cal = calibrate(train)
        options = dict(outdir=str(args.out / 'fits' / f'rep{rep:02d}'), reps=50,
            n=400, p=50, nonzero=6, baseline='weibull', scenario='baseline',
            priors='cox-rhs', sampler='hmc', samples=5000, warmup=2000,
            thin=1, chains=4, multichains='true', noncentered='true', init='rand',
            seed=1, constraint='soft_center', p0=6, standardize='false',
            diagnose='false', progress='false', resume='true')
        options.update({'rep-ids':rep, 'nuts-backend':'explicit', 'hmc-n-leapfrog':100,
            'target-accept':0.8, 'use-car':'true', 'sample-car-strength':'true',
            'car-weight-quantile':0.99, 'car-precision':1, 'car-strength-prior-a':1,
            'car-strength-prior-b':1, 'car-distance':'bray', 'prox-sum0-lambda':1e-3,
            'prox-l1-lambda':1e-6, 'rhs-slab-df':4, 'rhs-slab-s2':2,
            'rhs-tau-scale':cal['multiplier'], 'rhs-local-scale':1,
            'sslasso-spike-scale':.04, 'sslasso-slab-scale':.5,
            'alpha-prior-kind':'invgamma', 'alpha-prior-shape':1, 'alpha-prior-scale':50,
            'sample-thin':1, 'first-rep-sample-thin':1, 'save-wide-csv':'true', 'save-jld2':'false'})
        command = [args.julia, '-t', '4', '--startup-file=no', '--project='+str(args.snapshot),
                   str(args.snapshot/'scripts/run_simulation.jl')]
        for key, value in options.items():
            command.extend(['--'+key, str(value)])
        jobs.append(dict(rep=rep, reference=str(source), calibration=cal, command=command))
    protocol = dict(question='Sensitivity to unit risk-set-variance approximation',
        calibration='Solve sum_j tau*sqrt(I_jj(0))/(1+tau*sqrt(I_jj(0)))=6 using training data only.',
        limitations='Diagonal null information is itself an approximation; not a full column-standardization study.',
        snapshot=str(args.snapshot), jobs=jobs)
    planfile = args.out/'protocol.json'
    if planfile.exists():
        assert json.loads(planfile.read_text()) == protocol, 'Existing protocol differs.'
    else:
        planfile.write_text(json.dumps(protocol, indent=2))
    print('Prepared 50 paired fits; multiplier range:',
          min(j['calibration']['multiplier'] for j in jobs),
          max(j['calibration']['multiplier'] for j in jobs), flush=True)
    if args.prepare_only:
        return
    (args.out/'logs').mkdir(exist_ok=True)
    def work(job):
        rep = job['rep']
        receipt = args.out/'logs'/f'rep{rep:02d}.json'
        if receipt.exists() and json.loads(receipt.read_text())['returncode'] == 0:
            return json.loads(receipt.read_text())
        started = time.time()
        with (args.out/'logs'/f'rep{rep:02d}.log').open('a') as log:
            proc = subprocess.run(job['command'], cwd=args.snapshot, stdout=log,
                stderr=subprocess.STDOUT, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', GKSwstype='100'))
        result = dict(rep=rep, returncode=proc.returncode, started=started, finished=time.time())
        receipt.write_text(json.dumps(result))
        print(json.dumps(result), flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(work, jobs))
    (args.out/'completion.json').write_text(json.dumps(results, indent=2))
    assert all(r['returncode'] == 0 for r in results), 'Some fits failed; inspect logs.'


if __name__ == '__main__':
    main()
