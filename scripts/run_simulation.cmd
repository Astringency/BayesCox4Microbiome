@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0\.."

REM set "GKSwstype=100"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "current_time=%%i"

REM TEST
set "test_n=400"
set "test_p=150"

julia +1.11.7 -t 4 --project=. scripts\run_simulation.jl ^
    --sampler ula ^
    --samples 5000 ^
    --warmup 2000 ^
    --chains 4 ^
    --multichains ^
    --reps 1 ^
    --n %test_n% ^
    --p %test_p% ^
    --nonzero 12 ^
    --baseline weibull ^
    --priors lasso ^
    --constraint soft_center ^
    --target-accept 0.8 ^
    --mala-step-size 1e-3 ^
    --prox-sum0-lambda 1e-2 ^
    --prox-l1-lambda 1e-2 ^
    --mala-target-accept 0.564 ^
    --seed 1 ^
    --use-car true ^
    --car-distance bray ^
    --p0 truth ^
    --p0-alpha 0.05 ^
    --outdir "results\simulation\test_n%test_n%_p%test_p%_non12_weibull_%current_time%"

if errorlevel 1 (
    echo SIMULATION FAILED! 1>&2
    exit /b 1
)

REM MAIN LOOP
REM for %%n in (200 400) do (
REM     for %%p in (50 150) do (
REM         for %%b in (weibull exponential gompertz) do (
REM             for %%z in (6 12) do (
REM                 echo [SIMULATION] n=%%n, p=%%p, %%z nonzero coefs with %%b baseline.
REM
REM                 julia +1.11.7 -t 4 --project=. scripts\run_simulation.jl ^
REM                     --sampler nuts ^
REM                     --nuts-backend explicit ^
REM                     --samples 5000 ^
REM                     --warmup 4000 ^
REM                     --chains 4 ^
REM                     --reps 50 ^
REM                     --n %%n ^
REM                     --p %%p ^
REM                     --nonzero %%z ^
REM                     --baseline %%b ^
REM                     --constraint soft_center ^
REM                     --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso ^
REM                     --target-accept 0.9 ^
REM                     --seed 1 ^
REM                     --use-car true ^
REM                     --car-distance bray ^
REM                     --p0 truth ^
REM                     --p0-alpha 0.05 ^
REM                     --outdir results\simulation\softcenter\n%%n_p%%p_nonzero%%z_%%b
REM
REM                 if errorlevel 1 (
REM                     echo SIMULATION FAILED! 1>&2
REM                     exit /b 1
REM                 )
REM             )
REM         )
REM     )
REM )

echo ALL SIMULATION DONE!
