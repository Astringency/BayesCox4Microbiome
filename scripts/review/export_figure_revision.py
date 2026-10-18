#!/usr/bin/env python3
"""Export figure inputs on server197 from audited manifests; stdlib, no refitting."""
import csv, gzip, hashlib, json, math, statistics, sys
from pathlib import Path
root=Path(sys.argv[1]);evidence=[]
def source(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    evidence.append(dict(path=str(p),sha256=h.hexdigest()))
def read(p):
    source(p)
    with p.open() as f:return list(csv.DictReader(f))
def write(p,rows):
    op=gzip.open if p.suffix=='.gz' else open
    with op(p,'wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def quantile(x,p):
    at=(len(x)-1)*p;i=int(at);frac=at-i
    return x[i]*(1-frac)+x[min(i+1,len(x)-1)]*frac
focus=sorted(read(root/'focus_runs.csv'),key=lambda r:int(r['replicate']))
diags=[];truths=[];supports=[]
for row in focus:
    src=Path(row['source']);rep=int(row['replicate'])
    nz=[dict(variable=r['variable'],true_beta=r['true_beta']) for r in read(src/'posterior_summary.csv') if float(r['true_beta'])!=0]
    assert len(nz)==6
    supports.append(tuple(r['variable'] for r in nz));lookup={r['variable']:r['true_beta'] for r in nz}
    diags.extend([dict(d,true_beta=lookup[d['parameters']],replicate=rep) for d in read(src/'rank_diagnostics.csv') if d['parameters'] in lookup])
    truths.extend([dict(r,replicate=rep) for r in nz])
    if rep==1:
        p=src/'samples_wide.csv';source(p);cols=['chain','iter']+list(lookup)
        with p.open() as f:draws=[{c:r[c] for c in cols} for r in csv.DictReader(f)]
        assert len(draws)==20000
        assert all(sum(r['chain']==str(c) for r in draws)==5000 for c in range(1,5))
        write(root/'trace_draws.csv.gz',draws);write(root/'nonzero_truth.csv',nz)
assert len(set(supports))==1
write(root/'nonzero_diagnostics.csv',diags);write(root/'nonzero_truth_by_repetition.csv',truths)
sums=[]
for row in read(root/'zero_runs.csv'):
    src=Path(row['source'])/'samples_wide.csv';source(src);values=[];start=len(sums)
    with src.open() as f:
        reader=csv.DictReader(f);cols=[c for c in reader.fieldnames if c.startswith('beta[')]
        assert len(cols)==int(row['p'])
        for r in reader:
            value=math.fsum(float(r[c]) for c in cols);values.append(value)
            sums.append(dict(chain=r['chain'],iter=r['iter'],sum_beta=value,cell=row['cell'],method=row['method'],constraint=row['cell'].split('_',2)[2]))
    assert len(values)==int(float(row['sum_beta_n_draws']))
    assert math.isclose(statistics.mean(values),float(row['sum_beta_mean']),rel_tol=1e-9,abs_tol=1e-12)
    values.sort()
    for p,k in [(0.025,'q025'),(.25,'q25'),(.5,'median'),(.75,'q75'),(.975,'q975')]:
        assert math.isclose(quantile(values,p),float(row['sum_beta_'+k]),rel_tol=1e-8,abs_tol=1e-12),(row['cell'],k)
    print('Verified',row['cell'],len(values),flush=True)
write(root/'zero_sum_draws.csv.gz',sums)
(root/'source_manifest.json').write_text(json.dumps(evidence,indent=2)+'\n')
print('Exported',len(diags),'coefficient diagnostics and',len(sums),'posterior-sum draws')
