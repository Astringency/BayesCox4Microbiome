#!/usr/bin/env python3
"""Independently replay S1--S4 scores from saved cohorts and coefficient summaries.

Only small data/summary files are read. No posterior resampling, no large draw
files, and no modification of original results. Harrell's C uses strict time
ordering and half credit for tied risk scores, matching the project definition.
"""
import argparse
import json
import math
from pathlib import Path

from aggregate_revision_diagnostics import read_csv, write_csv
from aggregate_revision_metrics import truth
from verify_t1d_saved_cindex import cindex


def audit(source):
    meta=json.loads((source/'metadata.json').read_text())
    assert not meta['standardize'] and meta['pseudocount']==0
    assert meta['scenario'] in ('weak','correlated_null','counts','phylo_bm')
    test=Path(meta['test_data']); train=test.parent.parent/'training_data'/test.name
    tables=[read_csv(train),read_csv(test)]
    matrices=[[[float(r[f'x{j}']) for j in range(1,51)] for r in table] for table in tables]
    pc=min(v for row in matrices[0] for v in row if v>0)/2
    designs=[]
    for matrix in matrices:
        design=[]
        for values in matrix:
            replaced=[v if v>0 else pc for v in values]
            total=sum(replaced)
            design.append([math.log(v/total) for v in replaced])
        designs.append(design)
    summary=read_csv(source/'posterior_summary.csv')
    summary.sort(key=lambda r:int(r['variable'].split('[')[1][:-1]))
    beta=[float(r['true_beta']) for r in summary]
    assert len(beta)==50 and sum(b!=0 for b in beta)==6
    covered=[float(r['lower'])<=b<=float(r['upper']) for r,b in zip(summary,beta)]
    selected=[float(r['lower'])>0 or float(r['upper'])<0 for r in summary]
    assert all(float(r['lower'])<=float(r['upper']) for r in summary)
    cover=dict(coverage_95_all=sum(covered)/50,
        coverage_95_nonzero=sum(c for c,b in zip(covered,beta) if b!=0)/6,
        coverage_95_zero=sum(c for c,b in zip(covered,beta) if b==0)/44)
    fp=sum(s and b==0 for s,b in zip(selected,beta)); tp=sum(s and b!=0 for s,b in zip(selected,beta))
    fd=read_csv(source/'false_discovery.csv')[0]
    assert int(fd['FP'])==fp and int(fd['TP'])==tp and int(fd['FN'])==6-tp and int(fd['TN'])==44-fp
    assert abs(float(fd['FDR'])-fp/max(tp+fp,1))<1e-12
    errors=[]
    for row in read_csv(source/'metrics.csv'):
        estimate=[float(r[row['estimate'].lower()]) for r in summary]
        computed=dict(cover,MSE=sum((a-b)**2 for a,b in zip(estimate,beta))/50)
        for label,design,table in zip(('train','test'),designs,tables):
            risk=[math.fsum(a*b for a,b in zip(z,estimate)) for z in design]
            computed['Cindex_'+label]=cindex([float(r['time']) for r in table],
                [int(r['event']) for r in table],risk)[0]
        for metric,value in computed.items():
            error=abs(value-float(row[metric])); errors.append(error)
            assert error<1e-10,(source,row['estimate'],metric,value,row[metric])
    graph=meta.get('car_graph','abundance') if meta.get('use_car') else 'independent'
    return dict(source=str(source),scenario=meta['scenario'],graph=graph,
        replicate=meta['replicate_id'],n_score_checks=6,n_estimates=3,
        max_abs_difference=max(errors),coverage_and_fdp_match=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostics',nargs='+',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--expected-runs',type=int,default=450)
    args=parser.parse_args()
    sources=[]
    for path in args.diagnostics:
        sources += [Path(r['source']) for r in read_csv(path) if truth(r['primary_eligible']) and
            r['scenario'] in ('weak','correlated_null','counts','phylo_bm')]
    assert len(sources)==len(set(sources))==args.expected_runs
    results=[]
    for i,source in enumerate(sources,1):
        results.append(audit(source))
        if i%25==0: print(f'Verified {i}/{len(sources)} saved runs',flush=True)
    write_csv(args.outdir/'numerical_audit_by_run.csv',results)
    result=dict(all_metrics_match=True,verified_runs=len(results),verified_cindices=6*len(results),
        max_abs_difference=max(r['max_abs_difference'] for r in results),
        coverage_and_fdp_verified=True,posterior_resampled=False,
        limitation='Replays from saved posterior summaries; does not recompute their quantiles from large draw files')
    (args.outdir/'numerical_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
