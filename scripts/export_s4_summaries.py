#!/usr/bin/env python3
"""Package audited S4 summaries and provenance; never package large draws/data."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True,help='S4 deployment root containing results/S4')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve(); result=root/'results/S4'
    audit=json.loads((result/'reports/saved_results_audit.json').read_text())
    assert audit['complete'] and audit['completed_fits']==150
    for label,count in (('S1_S3',300),('S4',150)):
        a=json.loads((root/'audits'/label/'numerical_audit.json').read_text())
        assert a['all_metrics_match'] and a['verified_runs']==count
    files={}
    for folder,prefix in ((result/'reports','reports'),(root/'audits','audits'),
            (result/'car/part01/cox-rhs/independent/phylogeny/p50_nz6','phylogeny'),
            (result/'logs','execution_logs')):
        for path in folder.rglob('*'):
            if path.is_file(): files[prefix+'/'+path.relative_to(folder).as_posix()]=path
    for path in (root/'code_v3/source_manifest.json',result/'experiment.json',
                 root/'logs/formal.command.json',root/'logs/formal.log',root/'logs/formal.exit',
                 root/'logs/phylogeny_tests_v2.log',root/'logs/phylogeny_tests_v2.exit',
                 root/'logs/smoke_audit.json'):
        files['provenance/'+path.name]=path
    assert all(p.name not in ('samples_wide.csv','transition_diagnostics.csv') for p in files.values())
    manifest={'source':str(root),'large_draws_included':False,'files':{}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in sorted(files.items()):
            data=path.read_bytes(); archive.writestr(name,data)
            manifest['files'][name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        archive.writestr('manifest.json',json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(archive=str(args.output),files=len(files),bytes=args.output.stat().st_size,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))
