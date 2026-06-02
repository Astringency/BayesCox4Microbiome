@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

if not defined GKSwstype set "GKSwstype=100"

REM ---------------------------------------------------------------------------
REM Julia runner
REM ---------------------------------------------------------------------------
set "JULIA=julia +1.11.7 -t 4 --project=."

REM ---------------------------------------------------------------------------
REM Real-data CLI defaults from scripts\analyze_real_data.jl
REM ---------------------------------------------------------------------------
set "REALDATA_PRIORS=--priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso"
REM set "REALDATA_PRIORS=--priors prox-lasso,prox-sslasso"

REM ---------------------------------------------------------------------------
REM Sampler interfaces: general, NUTS, HMC, MALA, MALA-SGLD, Proximal block
REM ---------------------------------------------------------------------------
set "SAMPLER_GENERAL=--sampler nuts --samples 5000 --warmup 4000 --thin 1 --chains 4 --multichains true --target-accept 0.95 --noncentered true --seed 1 --progress true"
set "SAMPLER_NUTS=--nuts-backend explicit --max-depth 10"

REM --hmc-step-size
set "SAMPLER_HMC=--hmc-n-leapfrog 20"

set "SAMPLER_MALA=--mala-step-size 1e-4 --mala-target-accept 0.574 --mala-adapt-every 50 --mala-min-step 1e-8 --mala-max-step 1.0 --mala-precondition false --mala-precondition-mode auto --mala-precondition-diag false --mala-metric-ridge 1e-6 --mala-metric-max-cholesky-tries 8 --mala-stepsize-schedule adaptive_mcmc"
set "SAMPLER_MALA_SGLD=--mala-sgld-c0 1.0 --mala-sgld-target-accept 0.574 --mala-sgld-accept-low 0.40 --mala-sgld-accept-high 0.60 --mala-sgld-adapt-c0 true --mala-sgld-adapt-every 50 --mala-sgld-c0-min 1e-12 --mala-sgld-c0-max 1e12"

REM --prox-block-step-size
set "SAMPLER_PROX_BLOCK=--prox-block-min-step-size 1e-8 --prox-block-max-step-size 1.0 --prox-block-adapt-every 1 --prox-block-divergence-threshold 1000.0 --prox-block-metric unit_diag"

REM ---------------------------------------------------------------------------
REM Prior interfaces: horseshoe, RHS, lasso, spike-and-slab lasso, proximal L1, CAR
REM ---------------------------------------------------------------------------
set "PRIOR_HORSESHOE=--hs-tau-scale 1.0 --hs-local-scale 1.0"
set "PRIOR_RHS=--rhs-slab-df 4.0 --rhs-slab-s2 2.0 --rhs-tau-scale 1.0 --rhs-local-scale 1.0 --rhs-pseudo-sigma-df 3.0 --rhs-pseudo-sigma-scale 10.0"

REM --lasso-lambda
set "PRIOR_LASSO=--lasso-lambda2-shape 1.0 --lasso-lambda2-rate 0.1"

set "PRIOR_SSLASSO=--sslasso-spike-scale 1.0 --sslasso-slab-scale 10.0 --sslasso-theta-a 1.0 --sslasso-theta-b 1.0"

set "PRIOR_PROXIMAL_L1=--prox-l1-lambda 1e-3 --alpha-prior-shape 1.0 --alpha-prior-rate 1.0 --prox-root-tol 1e-10 --prox-root-maxiter 100"

set "PRIOR_CAR=--use-car true --car-distance bray --car-weight-quantile 0.75 --car-precision 1.0 --sample-car-strength false --car-strength-prior-a 1.0 --car-strength-prior-b 1.0 --car-degree-scale mean_active --car-kappa-init 0.5"

REM ---------------------------------------------------------------------------
REM Cox model and diagnostics interfaces
REM ---------------------------------------------------------------------------
set "COX_MODEL=--p0 auto --p0-alpha 0.05 --constraint soft_center --prox-sum0-lambda 1e-3 --pseudocount 1e-6 --standardize false --tie-method efron"
REM set "COX_MODEL=--p0 auto --p0-alpha 0.05 --constraint prox --prox-sum0-lambda 1e-3 --pseudocount 1e-6 --standardize false --tie-method efron"
set "DIAGNOSTICS=--diagnose true --diagnose-format pdf --diagnose-top-k-beta 10 --diagnose-max-lag 50"

REM Keep COMMON as the single reusable argument bundle used by all analyses.
set "COMMON=%REALDATA_PRIORS% %SAMPLER_GENERAL% %SAMPLER_NUTS% %SAMPLER_HMC% %SAMPLER_MALA% %SAMPLER_MALA_SGLD% %SAMPLER_PROX_BLOCK% %PRIOR_HORSESHOE% %PRIOR_RHS% %PRIOR_LASSO% %PRIOR_SSLASSO% %PRIOR_PROXIMAL_L1% %PRIOR_CAR% %COX_MODEL% %DIAGNOSTICS%"

REM ---------------------------------------------------------------------------
REM Dataset-specific interfaces
REM ---------------------------------------------------------------------------
set "T1D_DATA=--input data\T1D.csv --outdir results\realdata\T1D --time-col time --status-col status --comp-cols none --exclude-cols Sex,Antibiotics"
set "HCT_DATA=--input data\HCT.csv --outdir results\realdata\HCT --time-col time --status-col status --comp-cols none --exclude-cols sample_id"
set "TCGA_EXCLUDE=dataset,patient,cancer_type,sample_id,sample_type,pfi_time,pfi,dfi_time,dfi,dss_time,dss,os_censor,pfi_censor,dfi_censor,dss_censor"
set "EXTRA_ARGS=%*"

REM ---------------------------------------------------------------------------
REM 1. T1D data: data\T1D.csv
REM ---------------------------------------------------------------------------
%JULIA% scripts\analyze_real_data.jl ^
  %T1D_DATA% ^
  %COMMON% ^
  %EXTRA_ARGS%
if errorlevel 1 exit /b %ERRORLEVEL%

REM ---------------------------------------------------------------------------
REM 2. HCT data: data\HCT.csv
REM ---------------------------------------------------------------------------
%JULIA% scripts\analyze_real_data.jl ^
  %HCT_DATA% ^
  %COMMON% ^
  %EXTRA_ARGS%
if errorlevel 1 exit /b %ERRORLEVEL%

REM ---------------------------------------------------------------------------
REM 3. TCGA CIBERSORT data: 8 cancer-specific OS analyses
REM ---------------------------------------------------------------------------
call :run_tcga BLCA
if errorlevel 1 exit /b %ERRORLEVEL%
call :run_tcga BRCA
if errorlevel 1 exit /b %ERRORLEVEL%
call :run_tcga COAD
if errorlevel 1 exit /b %ERRORLEVEL%
call :run_tcga HNSC
if errorlevel 1 exit /b %ERRORLEVEL%
call :run_tcga LIHC
if errorlevel 1 exit /b %ERRORLEVEL%
call :run_tcga LUAD
if errorlevel 1 exit /b %ERRORLEVEL%
call :run_tcga OV
if errorlevel 1 exit /b %ERRORLEVEL%
call :run_tcga STAD
if errorlevel 1 exit /b %ERRORLEVEL%

exit /b 0

:run_tcga
set "CANCER=%~1"
set "TCGA_DATA=--input data\TCGA\cibersort\TCGA_%CANCER%_CIBERSORT_survival.csv --outdir results\realdata\TCGA_%CANCER%_CIBERSORT_OS --time-col os_time --status-col os --comp-cols none --exclude-cols %TCGA_EXCLUDE%"
%JULIA% scripts\analyze_real_data.jl ^
  %TCGA_DATA% ^
  %COMMON% ^
  %EXTRA_ARGS%
exit /b %ERRORLEVEL%

:optional_arg
set "%~1="
if not "%~3"=="" set "%~1=--%~2 %~3"
exit /b 0
