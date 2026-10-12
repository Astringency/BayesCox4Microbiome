#!/usr/bin/env python3
"""Launch explicitly selected revision suites in independent tmux sessions.

Each job has four Julia chain threads. Select groups to fit available CPU
capacity; all groups together can require over 80 logical CPUs. Existing tmux
sessions and completed jobs are skipped. Existing sampling scripts resume
completed repetitions. This launcher never removes outputs or server files.
"""
import argparse
import json
import shlex
import subprocess
from pathlib import Path


def jobs_for(group):
    if group in ('rhs_sensitivity', 'ssl_sensitivity'):
        prior = 'cox-rhs' if group.startswith('rhs') else 'prox-sslasso'
        # Cox-RHS 1e-3 is a saved baseline; it is verified separately below.
        values = ['1e-2', '1e-3', '1e-4', '1e-5']
        return [(group + '_' + v, 'scripts/run_sensitivity_revision.sh', ['lambda'],
                 dict(PRIORS=prior, VALUES=v)) for v in values]
    if group == 'p0':
        return [('p0_' + v, 'scripts/run_sensitivity_revision.sh', ['p0'], dict(VALUES=v))
                for v in ['3', '6', '12']]
    if group in ('rhs_scenarios', 'ssl_scenarios'):
        prior = 'cox-rhs' if group.startswith('rhs') else 'prox-sslasso'
        return [(group + '_' + v, 'scripts/run_scenarios_revision.sh', [],
                 dict(PRIORS=prior, SCENARIOS=v)) for v in ['S1', 'S2', 'S3']]
    if group == 'ssl_zero_missing':
        return [('ssl_zero_' + c, 'scripts/run_zero_sum_revision.sh', [],
                 dict(PRIORS='prox-sslasso', CONSTRAINTS=c)) for c in ['none', 'soft_center', 'prox']] + [
                 ('ssl_zero_prox1e-6', 'scripts/run_zero_sum_revision.sh', [],
                  dict(PRIORS='prox-sslasso', CONSTRAINTS='prox', SUM0_LAMBDA='1e-6'))]
    if group == 'rhs_car':
        return [('rhs_car_' + g, 'scripts/run_car_revision.sh', [],
                 dict(METHOD='cox-rhs', GRAPHS=g, SCENARIO='phylo')) for g in ['independent', 'correct', 'misspecified']]
    if group == 'rhs_car_s4':
        return [('rhs_car_s4_' + g, 'scripts/run_car_revision.sh', [],
                 dict(METHOD='cox-rhs', GRAPHS=g, SCENARIO='S4')) for g in ['independent', 'correct', 'misspecified']]
    if group == 'ssl_car_reference':
        return [('ssl_car_reference', 'scripts/run_car_revision.sh', [], dict(METHOD='prox-sslasso', SCENARIO='phylo'))]
    if group == 't1d_clinical':
        return [('t1d_' + prior + '_' + adjusted, 'scripts/run_t1d_clinical_revision.sh', [],
                 dict(METHOD=prior, ADJUSTED=adjusted, T1D_HISTORICAL_ROOT='./historical/T1D'))
                for prior in ['cox-rhs', 'prox-sslasso'] for adjusted in ['true', 'false']]
    if group == 'baseline_repairs':
        return [('baseline_repairs', 'scripts/run_baseline_repairs.sh', [], dict(REP_IDS='33,50'))]
    raise ValueError('unknown group: ' + group)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--julia', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--historical-root', default='/large_storage/zhangxf/BayesCox/results')
    p.add_argument('--groups', required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    project, outdir = Path(args.project).resolve(), Path(args.outdir).resolve()
    logs = outdir / 'job_logs'
    logs.mkdir(parents=True, exist_ok=True)
    for group in args.groups.split(','):
        for name, script, script_args, overrides in jobs_for(group):
            session = 'bcx_' + name.replace('.', '_')
            exit_path, log_path = logs / (name + '.exit'), logs / (name + '.log')
            env = dict(JULIA_BIN=args.julia, REVISION_OUT=str(outdir),
                       HISTORICAL_ROOT=args.historical_root, **overrides)
            command = ['env'] + [f'{k}={v}' for k, v in env.items()] + ['bash', script] + script_args
            record = dict(session=session, project=str(project), command=command, log=str(log_path))
            if exit_path.is_file() and exit_path.read_text().strip() == '0':
                print('complete; skip', session)
                continue
            exists = subprocess.run(['tmux', 'has-session', '-t', session],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if exists:
                print('running; skip', session)
                continue
            if args.dry_run:
                print(json.dumps(record))
                continue
            shell = 'cd ' + shlex.quote(str(project)) + ' && ' + shlex.join(command)
            shell += ' >> ' + shlex.quote(str(log_path)) + ' 2>&1'
            shell += '; status=$?; printf "%s\\n" "$status" > ' + shlex.quote(str(exit_path))
            (logs / (name + '.command.json')).write_text(json.dumps(record, indent=2))
            subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
            print('started', session)


if __name__ == '__main__':
    main()
