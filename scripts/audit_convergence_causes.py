#!/usr/bin/env python3
"""Read saved revision diagnostics and selected chains; never resample or edit fits."""
import argparse
import collections
import csv
import json
import math
import statistics as st
from pathlib import Path


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def number(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def write_csv(path, data):
    if not data:
        return
    fields = list(dict.fromkeys(k for r in data for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(data)


def chain_audit(source, label):
    source = Path(source)
    meta = json.loads((source / 'metadata.json').read_text())
    by_chain = collections.defaultdict(list)
    with (source / 'samples_wide.csv').open() as f:
        for r in csv.DictReader(f):
            by_chain[int(r['chain'])].append(r)
    result, coefficients = [], []
    for chain, rr in by_chain.items():
        betas = [k for k in rr[0] if k.startswith('beta[')]
        gammas = [k for k in rr[0] if k.startswith('gamma[')]
        alpha = [math.exp(float(r['eta'])) for r in rr] if 'eta' in rr[0] else []
        cfg = {**meta, **meta.get('selected_config', {})}
        spike = cfg.get('sslasso_spike_scale', .04)
        slab = cfg.get('sslasso_slab_scale', .5)
        nselected = [sum(float(r[k]) for k in gammas) for r in rr]
        norm = [sum(abs(float(r[k])) for k in betas) for r in rr]
        lp = [float(r['logposterior']) for r in rr]
        steps = meta.get('final_step_size_by_chain')
        record = dict(label=label, source=str(source), chain=chain, n_draws=len(rr),
                      mean_beta_l1=st.mean(norm), mean_selected=st.mean(nselected),
                      mean_lp=st.mean(lp), mean_lp_first_half=st.mean(lp[:len(lp)//2]),
                      mean_lp_second_half=st.mean(lp[len(lp)//2:]),
                      final_step=steps[chain-1][0] if steps else None)
        if alpha:
            ratios = [sum(abs(float(r[b])) / (slab if float(r[b.replace('beta','gamma')]) else spike)
                          for b in betas) / al for r, al in zip(rr, alpha)]
            record.update(alpha_mean=st.mean(alpha), alpha_min=min(alpha), alpha_max=max(alpha),
                          alpha_first_half=st.mean(alpha[:len(alpha)//2]),
                          alpha_second_half=st.mean(alpha[len(alpha)//2:]),
                          weighted_l1_over_alpha_mean=st.mean(ratios),
                          fraction_outside_l1_epigraph=st.mean(v > 1 for v in ratios))
        result.append(record)
        for beta in betas:
            gamma = beta.replace('beta', 'gamma')
            vals = [float(r[beta]) for r in rr]
            g = [int(float(r[gamma])) for r in rr] if gamma in rr[0] else []
            coefficients.append(dict(label=label, chain=chain, parameter=beta,
                                     mean=st.mean(vals), sd=st.stdev(vals),
                                     pip=st.mean(g) if g else None,
                                     switches=sum(a != b for a, b in zip(g, g[1:])) if g else None))
    return result, coefficients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    eligible = [r for r in rows(a.root/'results/diagnostic_aggregates_v2/diagnostics_by_run.csv')
                if r['primary_eligible'] == 'True']
    groups = collections.defaultdict(list)
    for r in eligible:
        groups[(r['method'], r['scenario'], r['n'], r['p'], r['constraint'],
                r['prox_sum0_lambda'], r['p0'], r['car_graph'])].append(r)
    summary = []
    for key, rr in sorted(groups.items()):
        s = dict(zip(['method','scenario','n','p','constraint','sum0_lambda','p0','car_graph'], key))
        s.update(n_fits=len(rr), beta_pass=sum(r['convergence_screen_pass']=='True' for r in rr))
        for col in ['beta_rhat_max','beta_ess_bulk_min','beta_ess_tail_min',
                    'gamma_rhat_max','mean_acceptance_probability','gamma_constant_chain_fraction']:
            vals = [number(r[col]) for r in rr]
            vals = [v for v in vals if v is not None]
            if vals:
                s[col+'_median'] = st.median(vals)
                s[col+'_worst'] = min(vals) if 'ess' in col else max(vals)
        s['divergences'] = sum(number(r['n_divergent_sample']) or 0 for r in rr)
        summary.append(s)
    write_csv(a.out/'simulation_groups.csv', summary)
    # Discordance is a diagnostic flag, not proof of a posterior inclusion probability.
    selection = []
    for r in eligible:
        if r['method'] != 'prox-sslasso':
            continue
        m = rows(Path(r['source'])/'selection_mixing.csv')
        byparam = collections.defaultdict(list)
        for v in m:
            byparam[v['parameter']].append(v)
        spans = [max(float(v['inclusion_probability']) for v in vv)-
                 min(float(v['inclusion_probability']) for v in vv) for vv in byparam.values()]
        selection.append(dict(source=r['source'], scenario=r['scenario'], n=r['n'], p=r['p'],
                              constraint=r['constraint'], sum0_lambda=r['prox_sum0_lambda'],
                              beta_pass=r['convergence_screen_pass'], max_pip_chain_range=max(spans),
                              n_gamma_pip_range_over_half=sum(x > .5 for x in spans),
                              n_constant_gamma_chains=sum(v['constant_chain'].lower()=='true' for v in m)))
    write_csv(a.out/'selection_by_run.csv', selection)
    selected = []
    for scenario in ['weak', 'correlated_null', 'counts']:
        rr = [r for r in eligible if r['method']=='prox-sslasso' and r['scenario']==scenario]
        rr.sort(key=lambda r: float(r['beta_rhat_max']))
        for tag, r in [('best',rr[0]),('worst',rr[-1])]:
            selected.append((r['source'],scenario+'_'+tag))
    for n,p,con,lam,label in [('400','50','prox','0.001','n400_control'),
                             ('400','150','none','','p150_none'),
                             ('400','150','prox','1e-06','p150_tight'),
                             ('400','150','exact','','p150_exact')]:
        rr = [r for r in eligible if r['method']=='prox-sslasso' and r['n']==n and r['p']==p
              and r['constraint']==con and (not lam or r['prox_sum0_lambda']==lam)]
        if rr:
            selected.append((min(rr,key=lambda r:int(r['replicate']))['source'],label))
    for arm in ['adjusted','unadjusted']:
        selected.append((str(a.root/'results/corrected_ssl/T1D_clinical'/arm/'prox-sslasso/runs/realdata_prox-sslasso'), 'T1D_'+arm))
    chains, coefs = [], []
    for path, label in selected:
        cc, bb = chain_audit(path,label)
        chains.extend(cc); coefs.extend(bb)
        print(label, [{k:round(v,4) if isinstance(v,float) else v for k,v in c.items()
                       if k in ['chain','alpha_mean','mean_beta_l1','mean_selected','mean_lp','final_step']} for c in cc], flush=True)
    write_csv(a.out/'selected_chain_summary.csv', chains)
    write_csv(a.out/'selected_chain_coefficients.csv', coefs)
    inner = []
    for method, prefix in [('cox-rhs',''),('prox-sslasso','corrected_ssl/')]:
        for arm in ['adjusted','unadjusted']:
            path = a.root/'results'/prefix/'T1D_clinical'/arm/method
            for f in sorted(path.rglob('inner_cv_metrics.csv')):
                rr = rows(f)
                ok = lambda r: (number(r.get('fit_rhat_max')) is not None and
                    float(r['fit_rhat_max']) <= 1.01 and
                    (number(r.get('fit_ess_bulk_min')) or 0)>=400 and
                    (number(r.get('fit_ess_tail_min')) or 0)>=400)
                by_config = collections.defaultdict(list)
                for r in rr:
                    by_config[r['config_index']].append(r)
                inner.append(dict(method=method, adjustment=arm, source=str(f), n_fits=len(rr),
                    beta_strict_pass=sum(ok(r) for r in rr),
                    n_candidates=len(by_config),
                    candidates_all_five_beta_pass=sum(len(v)==5 and all(ok(r) for r in v)
                                                      for v in by_config.values()),
                    legacy_flag_false=sum(r.get('fit_converged','').lower()=='false' for r in rr),
                    rhat_max=max((number(r.get('fit_rhat_max')) or 0) for r in rr)))
    write_csv(a.out/'inner_cv_quality.csv', inner)
    (a.out/'audit.json').write_text(json.dumps(dict(eligible_fits=len(eligible),
        simulation_groups=len(summary), ssl_selection_fits=len(selection),
        chain_audit_sources=[dict(source=p,label=l) for p,l in selected],
        resampled=False), indent=2))


if __name__ == '__main__':
    main()
