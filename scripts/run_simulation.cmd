@echo off
setlocal
cd /d "%~dp0\.."

if not defined GKSwstype set "GKSwstype=100"

julia --project=. scripts\run_simulation.jl ^
  --sampler mh ^
  --samples 50 ^
  --warmup 20 ^
  --reps 1 ^
  --n 200 ^
  --p 50 ^
  --nonzero 6 ^
  --baseline weibull ^
  --priors hs,cox-rhs,lasso ^
  --p0 truth ^
  %*

exit /b %ERRORLEVEL%

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
REM     --diagnose-format png ^
REM     --diagnose-top-k-beta 10 ^
REM     --diagnose-max-lag 50 ^
REM     --resume true ^
REM     --plot-first-rep-only true ^
REM     --sample-thin 10 ^
REM     --first-rep-sample-thin 1 ^
REM     --save-wide-csv true ^
REM     --save-jld2 false
