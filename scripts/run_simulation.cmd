@echo off
setlocal
cd /d "%~dp0\.."

if not defined GKSwstype set "GKSwstype=100"

set test_n=200
set test_p=50
set test_nonzero=6
set test_baseline=weibull

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "timestamp=%%i"

julia "+1.11.7" -t 4 --project=. scripts\run_simulation.jl ^
    --sampler nuts ^
    --nuts-backend explicit ^
    --samples 5000 ^
    --warmup 2000 ^
    --chains 4 ^
    --multichains ^
    --reps 1 ^
    --n "%test_n%" ^
    --p "%test_p%" ^
    --nonzero "%test_nonzero%" ^
    --baseline "%test_baseline%" ^
    --priors prox-lasso,prox-sslasso ^
    --constraint prox ^
    --target-accept 0.95 ^
    --seed 1 ^
    --use-car true ^
    --car-distance bray ^
    --p0 truth ^
    --p0-alpha 0.05 ^
    --standardize false ^
    --outdir "results/simulation/test_n%test_n%_p%test_p%_non%test_nonzero%_%test_baseline%_%timestamp%"

REM explicit turing 
REM soft_center prox none

REM julia +1.11.7 -t 4 --project=. scripts\run_simulation.jl ^
REM     --outdir results/simulation ^
REM     --reps 1 ^
REM     --n 200 ^
REM     --p 50 ^
REM     --nonzero 6 ^
REM     --baseline weibull ^
REM     --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso ^
REM     --sampler nuts ^
REM     --nuts-backend turing ^
REM     --samples 500 ^
REM     --warmup 200 ^
REM     --thin 1 ^
REM     --chains 4 ^
REM     --multichains false ^
REM     --target-accept 0.8 ^
REM     --seed 1 ^
REM     --use-car true ^
REM     --progress true ^
REM     --car-distance bray ^
REM     --p0 truth ^
REM     --p0-alpha 0.05 ^
REM     --constraint soft_center ^
REM     --soft-center-scale  ^
REM     --prox-sum0-lambda 1e-3 ^
REM     --prox-l1-lambda 1e-3 ^
REM     --prox-root-tol 1e-10 ^
REM     --prox-root-maxiter 100 ^
REM     --mala-step-size 1e-4 ^
REM     --mala-target-accept 0.574 ^
REM     --mala-adapt-every 50 ^
REM     --mala-min-step 1e-8 ^
REM     --mala-max-step 1.0 ^
REM     --diagnose true ^
REM     --diagnose-format pdf ^
REM     --diagnose-top-k-beta 10 ^
REM     --diagnose-max-lag 50 ^
REM     --resume true ^
REM     --plot-first-rep-only true ^
REM     --sample-thin 10 ^
REM     --first-rep-sample-thin 1 ^
REM     --save-wide-csv true ^
REM     --save-jld2 false
