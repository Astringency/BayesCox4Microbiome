@echo off
setlocal
cd /d "%~dp0\.."

if not defined GKSwstype set "GKSwstype=100"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "timestamp=%%i"

set test_n=200
set test_p=150
set test_nonzero=12
set test_baseline=weibull

set /a test_p_plus_1=test_p+1

REM explicit turing 
REM soft_center prox none

julia "+1.11.7" -t 4 --project=. scripts\run_simulation.jl ^
    --outdir "results/simulation/test_n%test_n%_p%test_p%_non%test_nonzero%_%test_baseline%_%timestamp%" ^
    --reps 1 ^
    --n "%test_n%" ^
    --p "%test_p%" ^
    --nonzero "%test_nonzero%" ^
    --baseline "%test_baseline%" ^
    --priors cox-rhs ^
    --sampler nuts ^
    --nuts-backend explicit ^
    --samples 5000 ^
    --warmup 2000 ^
    --thin 1 ^
    --chains 4 ^
    --multichains true ^
    --target-accept 0.95 ^
    --seed 1 ^
    --use-car true ^
    --progress true ^
    --car-distance "bray" ^
    --p0 truth ^
    --p0-alpha 0.05 ^
    --constraint "soft_center" ^
    --prox-sum0-lambda 1e-5 ^
    --prox-l1-lambda 1e-5 ^
    --prox-root-tol 1e-10 ^
    --prox-root-maxiter 100 ^
    --mala-step-size 1e-4 ^
    --mala-target-accept 0.574 ^
    --mala-adapt-every 50 ^
    --mala-min-step 1e-8 ^
    --mala-max-step 1.0 ^
    --diagnose true ^
    --debug false ^
    --diagnose-format "pdf" ^
    --diagnose-top-k-beta 10 ^
    --diagnose-max-lag 50 ^
    --resume true ^
    --pseudocount 0.0 ^
    --standardize false ^
    --plot-first-rep-only true ^
    --sample-thin 10 ^
    --first-rep-sample-thin 1 ^
    --save-wide-csv true ^
    --save-jld2 false ^
    --alpha-prior-shape %test_p_plus_1% ^
    --alpha-prior-rate 1
