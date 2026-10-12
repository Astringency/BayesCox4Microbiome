#!/usr/bin/env python3
"""Check the three-arm S4 pipeline before formal sampling (no fitting)."""
import argparse
import hashlib
import json
from pathlib import Path

from aggregate_revision_diagnostics import read_csv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    runs=sorted(args.root.rglob('complete.txt'))
    assert len(runs)==3, f'expected three completed smoke fits; got {len(runs)}'
    cohorts=[]; trees=[]; draws=[]; graphs=set()
    for done in runs:
        source=done.parent
        meta=json.loads((source/'metadata.json').read_text())
        assert meta['scenario']=='phylo_bm' and meta['dgp_version']=='phylo_bm_v1'
        graph=meta['car_graph']; graphs.add(graph)
        assert meta['use_car']==(graph!='independent')
        assert meta['sample_car_strength']==meta['use_car']
        assert meta['chains']==4 and meta['n_test']==1000
        test=Path(meta['test_data']); train=test.parent.parent/'training_data'/test.name
        cohorts.append(tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in (train,test,test.with_suffix('.json'))))
        trees.append(meta['phylogenetic_tree_sha256'])
        metrics=read_csv(source/'metrics.csv')
        assert {r['estimate'] for r in metrics}=={'mean','median','MAP'}
        for r in metrics:
            assert all(0<=float(r[k])<=1 for k in ('Cindex_train','Cindex_test','coverage_95_all','coverage_95_nonzero','coverage_95_zero'))
        draws.append(hashlib.sha256((source/'samples_wide.csv').read_bytes()).hexdigest())
    assert graphs=={'independent','correct','misspecified'}
    assert len(set(cohorts))==len(set(trees))==1
    assert len(set(draws))==3, 'graph changes did not affect fitted draws'
    print(json.dumps(dict(passed=True,completed_arms=3,identical_train_test_and_tree=True,
        distinct_posteriors=True,formal_performance_claim=False)))


if __name__=='__main__':
    main()
