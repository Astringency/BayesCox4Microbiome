@echo off
setlocal
cd /d "%~dp0\.."

if not defined GKSwstype set "GKSwstype=100"

set "JULIA=julia +1.11.7 -t 4 --project=."
set "COMMON=--priors prox-lasso,prox-sslasso --sampler nuts --nuts-backend explicit --target-accept 0.95 --samples 5000 --warmup 4000 --chains 4 --multichains true --p0 auto --constraint prox --diagnose true"
set "TCGA_EXCLUDE=dataset,patient,cancer_type,sample_id,sample_type,pfi_time,pfi,dfi_time,dfi,dss_time,dss,os_censor,pfi_censor,dfi_censor,dss_censor"
set "EXTRA_ARGS=%*"

REM ---------------------------------------------------------------------------
REM 1. T1D data: data\T1D.csv
REM ---------------------------------------------------------------------------
%JULIA% scripts\analyze_real_data.jl ^
  --input data\T1D.csv ^
  --outdir results\realdata\T1D ^
  --time-col time ^
  --status-col status ^
  --comp-cols none ^
  --exclude-cols Sex,Antibiotics ^
  %COMMON% ^
  %EXTRA_ARGS%
if errorlevel 1 exit /b %ERRORLEVEL%

REM ---------------------------------------------------------------------------
REM 2. HCT data: data\HCT.csv
REM ---------------------------------------------------------------------------
%JULIA% scripts\analyze_real_data.jl ^
  --input data\HCT.csv ^
  --outdir results\realdata\HCT ^
  --time-col time ^
  --status-col status ^
  --comp-cols none ^
  --exclude-cols sample_id ^
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
%JULIA% scripts\analyze_real_data.jl ^
  --input data\TCGA\cibersort\TCGA_%CANCER%_CIBERSORT_survival.csv ^
  --outdir results\realdata\TCGA_%CANCER%_CIBERSORT_OS ^
  --time-col os_time ^
  --status-col os ^
  --comp-cols none ^
  --exclude-cols %TCGA_EXCLUDE% ^
  %COMMON% ^
  %EXTRA_ARGS%
exit /b %ERRORLEVEL%
