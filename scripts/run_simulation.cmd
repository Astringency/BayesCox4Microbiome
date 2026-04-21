@echo off
setlocal
cd /d "%~dp0\.."

julia --project=. scripts\run_simulation.jl ^
  --sampler mh ^
  --samples 50 ^
  --warmup 20 ^
  --reps 1 ^
  --n 200 ^
  --p 50 ^
  --nonzero 6 ^
  --baseline weibull ^
  --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso ^
  --use-car true ^
  --car-distance bray ^
  --p0 auto ^
  --p0-alpha 0.05 ^
  --outdir results\simulation
