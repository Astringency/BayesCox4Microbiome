@echo off
setlocal
cd /d "%~dp0\.."

@echo off

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "now=%%i"

set "test_n=400"
set "test_p=150"

julia +1.11.7 -t 4 --project=. scripts\run_simulation.jl ^
    --sampler nuts ^
    nuts_backend = :explicit ^
    --samples 5000 ^
    --warmup 2000 ^
    --chains 4 ^
    --reps 1 ^
    --n %test_n% ^
    --p %test_p% ^
    --nonzero 12 ^
    --baseline weibull ^
    --priors rhs ^
    --constraint prox ^
    --target-accept 0.8 ^
    --mala-target-accept 0.9 ^
    --mala-step-size 1e-8 ^
    --prox-sum0-lambda 1e-5 ^
    --prox-l1-lambda 1e-5 ^
    --seed 1 ^
    --use-car true ^
    --car-distance bray ^
    --p0 truth ^
    --p0-alpha 0.05 ^
    --outdir "results/simulation/test_n%test_n%_p%test_p%_non12_weibull_%now%"


REM julia -t 4 --project=. scripts\run_simulation.jl ^
REM   --sampler nuts ^
REM   --samples 5000 ^
REM   --warmup 2000 ^
REM   --chains 4 ^
REM   --reps 100 ^
REM   --n 200,400 ^
REM   --p 50,150,300 ^
REM   --nonzero 6,12 ^
REM   --baseline weibull,exponential,gompertz ^
REM   --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso ^
REM   --target-accept 0.9 ^
REM   --seed 1 ^
REM   --use-car true ^
REM   --progress true ^
REM   --car-distance bray ^
REM   --p0 truth ^
REM   --p0-alpha 0.05 ^
REM   --outdir results\simulation
