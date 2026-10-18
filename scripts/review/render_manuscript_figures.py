#!/usr/bin/env python3
"""Revised publication figures using current review outputs.

Adapted from scripts/plot_simulation.R (plot_coef_sum, plot_trace_density,
plot_acf_rhat) and the baseline realdata_*_survival_analysis.R C-index plots.
Changes: current posterior draws, rank diagnostics across all repetitions,
exact-zero benchmark, common outer folds, and explicit SD/interval meanings.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

REPO=Path(__file__).resolve().parents[2]
REVIEW=REPO/'results/review_20261012'
WORK=REPO/'results/manuscript_revision_20261014'
PAPER=Path('/mnt/e/000GoodGoodStudy/C06LaTex/Paper/Microbiome/v7_en_tf_review')
OUT=PAPER/'figs/review'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.labelsize':9,
    'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8})
colors=['#0072B2','#D55E00','#009E73','#CC79A7']
sources={}
def read(path,**kwargs):
    sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path,**kwargs)
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',dpi=180)
    plt.close(fig)

diag=read(REVIEW/'rerun_20261013/combined/diagnostics_by_run.csv')
constraints=['none','exact','soft_center','prox1e-3','prox1e-6']
labels=['None','Exact','Soft-center',r'Prox: $\lambda_\Sigma=10^{-3}$',r'Prox: $\lambda_\Sigma=10^{-6}$']
fig,axs=plt.subplots(1,2,figsize=(8,3.2),sharex=True,sharey=True,layout='constrained')
for ax,method,title in zip(axs,['rhs','ssl'],['(a) Cox-RHS','(b) Prox-SSLASSO']):
    for i,c in enumerate(constraints):
        row=diag[(diag.cell==f'zero_{method}_{c}')&(diag.replicate==1)].iloc[0]
        ax.errorbar(row.sum_beta_mean,i,xerr=[[row.sum_beta_mean-row.sum_beta_q025],[row.sum_beta_q975-row.sum_beta_mean]],
            fmt='o',color=colors[0 if method=='rhs' else 1],capsize=3,markersize=4)
    ax.axvline(0,color='.4',ls='--',lw=.7)
    ax.set(title=title,xlabel=r'Posterior sum $\sum_j\beta_j$')
axs[0].set_yticks(range(5),labels); axs[0].invert_yaxis()
save(fig,'zero_sum')

trace=read(WORK/'bcx_review_trace.csv.gz')
indices=[18,20,22,24,26,28]; truth=[2.08,1.31,-1.39,-1.86,2.12,-2.26]
fig,axs=plt.subplots(6,2,figsize=(7.5,8.0),gridspec_kw={'width_ratios':[3,1]},layout='constrained')
for k,(j,beta) in enumerate(zip(indices,truth)):
    col=f'beta[{j}]'; lo,hi=trace[col].min(),trace[col].max(); grid=np.linspace(lo,hi,180)
    for chain in range(1,5):
        d=trace[trace.chain==chain]
        axs[k,0].plot(d['iter'],d[col],lw=.22,alpha=.65,color=colors[chain-1],rasterized=True,label=str(chain))
        axs[k,1].plot(grid,gaussian_kde(d[col])(grid),lw=.8,color=colors[chain-1])
    axs[k,0].axhline(beta,color='.2',ls='--',lw=.65)
    axs[k,1].axvline(beta,color='.2',ls='--',lw=.65)
    axs[k,0].set_ylabel(r'$\beta_{'+str(j)+'}$')
    if k<5: axs[k,0].tick_params(labelbottom=False)
    axs[k,1].set_yticks([])
axs[0,0].set_title('Retained iterations');axs[0,1].set_title('Marginal density')
axs[5,0].set_xlabel('Iteration');axs[5,1].set_xlabel('Coefficient')
axs[0,0].legend(title='Chain',ncol=4,loc='upper right',frameon=False)
save(fig,'trace_density')

fig,axs=plt.subplots(1,3,figsize=(8.2,3.5),layout='constrained')
groups=[('cox-rhs','soft_center','RHS / SC'),('cox-rhs','prox','RHS / prox'),('prox-sslasso','prox','SSL / prox')]
for i,(method,constraint,label) in enumerate(groups):
    data=diag[(diag.suite=='main')&(diag.method==method)&(diag.constraint==constraint)]
    assert len(data)==800
    offset=np.linspace(-.22,.22,len(data))
    for ax,key in zip(axs,['beta_rhat_max','beta_ess_bulk_min','beta_ess_tail_min']):
        ax.scatter(i+offset,data[key],s=4,alpha=.35,color=colors[i],rasterized=True)
for ax,key,title,threshold in zip(axs,['r','b','t'],[r'Maximum $\hat R_\beta$','Minimum bulk ESS','Minimum tail ESS'],[1.01,400,400]):
    ax.axhline(threshold,color='.25',ls='--',lw=.8)
    ax.set_xticks(range(3),[g[2] for g in groups],rotation=20,ha='right')
    ax.set_title(title)
    if key!='r': ax.set_yscale('log')
save(fig,'diagnostics')

cv=read(REVIEW/'baselines/T1D_comparison_summary.csv')
folds=read(REVIEW/'baselines/T1D_comparison_by_fold.csv')
methods=[('cox-rhs','soft_center','Cox-RHS / soft-center'),('cox-rhs','prox','Cox-RHS / prox'),
         ('prox-sslasso','soft_center','Prox-SSLASSO / soft-center'),('prox-sslasso','prox','Prox-SSLASSO / prox'),
         ('FLORAL_min','package_default','FLORAL / min'),('FLORAL_1se','package_default','FLORAL / 1se'),
         ('coda4microbiome_lambda.min','package_default','coda4microbiome / min'),
         ('coda4microbiome_lambda.1se','package_default','coda4microbiome / 1se')]
fig,axs=plt.subplots(1,2,figsize=(9,4.5),sharey=True,sharex=True,layout='constrained')
for ax,adj,title in zip(axs,['unadjusted','adjusted'],['(a) Unadjusted','(b) Sex and antibiotic adjusted']):
    for i,(method,constraint,label) in enumerate(methods):
        r=cv.query('method==@method and constraint==@constraint and adjustment==@adj').iloc[0]
        d=folds.query('method==@method and constraint==@constraint and adjustment==@adj').sort_values('outer_fold')
        assert len(d)==5
        assert np.isclose(d.test_Cindex.mean(),r.Cindex_mean)
        assert np.isclose(d.test_Cindex.std(ddof=1),r.Cindex_sd)
        ax.scatter(d.test_Cindex,i+np.linspace(-.12,.12,5),s=13,facecolors='none',edgecolors='.55',linewidths=.7)
        ax.errorbar(r.Cindex_mean,i,xerr=r.Cindex_sd,fmt='o',color=colors[0 if adj=='unadjusted' else 1],capsize=3,markersize=4,lw=1.2)
    ax.axvline(.5,color='.4',ls='--',lw=.7)
    ax.set(title=title,xlabel='Outer-fold C-index',xlim=(.4,.73))
axs[0].set_yticks(range(8),[x[2] for x in methods]);axs[0].invert_yaxis()
save(fig,'T1D_nested_cv')

# All residual panels are retained in a separate PDF for each final mouse fit.
# Each plot uses the already computed posterior-mean residuals and df=4 smooth.
for constraint in ['prox','soft_center']:
    for adjustment in ['adjusted','unadjusted']:
        for method in ['cox-rhs','prox-sslasso']:
            src=REVIEW/'T1D/final'/constraint/adjustment/method/'PH'
            residuals=read(src/'schoenfeld_residuals.csv');smooth=read(src/'schoenfeld_smooth.csv')
            # Preserve the original verified PDF, whose plotting source is check_t1d_ph.R.
            import shutil
            target=OUT/'PH'/f'{method}_{constraint}_{adjustment}.pdf';target.parent.mkdir(exist_ok=True)
            shutil.copy2(src/'posterior_mean_schoenfeld.pdf',target)
(WORK/'figure_sources.json').write_text(json.dumps(sources,indent=2))
print('Four revised figures and eight PH companion PDFs saved to',OUT)
