@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0\.."

REM set "GKSwstype=100"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "current_time=%%i"

REM TEST
if not defined test_sampler set "test_sampler=ula"
if not defined test_samples set "test_samples=5000"
if not defined test_warmup set "test_warmup=2000"
if not defined test_thin set "test_thin=1"
if not defined test_chains set "test_chains=4"
if not defined test_reps set "test_reps=1"
if not defined test_n set "test_n=400"
if not defined test_p set "test_p=150"
if not defined test_nonzero set "test_nonzero=12"
if not defined test_baseline set "test_baseline=weibull"
if not defined test_priors set "test_priors=prox-lasso"
if not defined test_constraint set "test_constraint=prox"
if not defined test_target_accept set "test_target_accept=0.8"
if not defined test_mala_step_size set "test_mala_step_size=1e-4"
if not defined test_prox_sum0_lambda set "test_prox_sum0_lambda=1e-2"
if not defined test_prox_l1_lambda set "test_prox_l1_lambda=1e-2"
if not defined test_mala_target_accept set "test_mala_target_accept=0.574"
if not defined test_seed set "test_seed=1"
if not defined test_use_car set "test_use_car=true"
if not defined test_car_distance set "test_car_distance=bray"
if not defined test_p0 set "test_p0=truth"
if not defined test_p0_alpha set "test_p0_alpha=0.05"
if not defined test_progress set "test_progress=true"
if not defined test_diagnose set "test_diagnose=true"

julia +1.11.7 -t 4 --project=. scripts\run_simulation.jl ^
    --sampler %test_sampler% ^
    --samples %test_samples% ^
    --warmup %test_warmup% ^
    --thin %test_thin% ^
    --chains %test_chains% ^
    --multichains ^
    --reps %test_reps% ^
    --n %test_n% ^
    --p %test_p% ^
    --nonzero %test_nonzero% ^
    --baseline %test_baseline% ^
    --priors %test_priors% ^
    --constraint %test_constraint% ^
    --target-accept %test_target_accept% ^
    --mala-step-size %test_mala_step_size% ^
    --prox-sum0-lambda %test_prox_sum0_lambda% ^
    --prox-l1-lambda %test_prox_l1_lambda% ^
    --mala-target-accept %test_mala_target_accept% ^
    --seed %test_seed% ^
    --use-car %test_use_car% ^
    --car-distance %test_car_distance% ^
    --p0 %test_p0% ^
    --p0-alpha %test_p0_alpha% ^
    --progress %test_progress% ^
    --diagnose %test_diagnose% ^
    --outdir "results\simulation\test_n%test_n%_p%test_p%_non%test_nonzero%_%test_baseline%_%current_time%"

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
