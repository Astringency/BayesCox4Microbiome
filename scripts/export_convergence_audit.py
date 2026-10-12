#!/usr/bin/env python3
"""Package only compact evidence from the convergence investigation."""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args(); b=a.root; out=b/'summary_bundle';out.mkdir(exist_ok=True)
    for f in (b/'reports').iterdir():
        if f.is_file():shutil.copy2(f,out/f.name)
    summaries=[]
    for suite in ['probes','real_budget']:
        for f in sorted((b/suite).glob('*/summary.json')):
            d=out/suite/f.parent.name;d.mkdir(parents=True,exist_ok=True)
            for name in ['summary.json','rank_diagnostics.csv','transition_summary.csv','metadata.json']:
                src=f.parent/name
                if src.exists():shutil.copy2(src,d/name)
            m=json.loads(f.read_text())
            record=dict(suite=suite,case=f.parent.name,**{k:m[k] for k in [
                'beta_rhat_max','beta_ess_bulk_min','beta_ess_tail_min','beta_screen_pass','seconds']})
            if suite=='probes':
                record.update(alpha_mean_min=min(r['alpha_mean'] for r in m['chains']),
                              alpha_mean_max=max(r['alpha_mean'] for r in m['chains']))
            summaries.append(record)
    if len(summaries)!=9:
        raise RuntimeError(f'Expected 7 simulation and 2 real-data diagnostic fits, got {len(summaries)}')
    with (out/'pilot_summary.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in summaries for k in r))
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(summaries)
    for name in ['probes.log','small_l1.log','real_budget.log','kernel.log','audit_v2.log']:
        shutil.copy2(b/name,out/name)
    files=[dict(path=str(f.relative_to(out)),bytes=f.stat().st_size,
                sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='manifest.json']
    (out/'manifest.json').write_text(json.dumps(dict(files=files,formal_results_modified=False,
        posterior_samples_included=False,n_diagnostic_fits=9,source_root=str(b)),indent=2))
    with zipfile.ZipFile(b/'summary_bundle.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(out))
    print(json.dumps(dict(files=len(files)+1,bytes=(b/'summary_bundle.zip').stat().st_size,
        sha256=hashlib.sha256((b/'summary_bundle.zip').read_bytes()).hexdigest())))


if __name__=='__main__':main()
