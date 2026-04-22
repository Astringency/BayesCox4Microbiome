@echo off
setlocal
cd /d "%~dp0\.."

julia --project=. scripts\run_simulation.jl ^
  --sampler nuts ^
  --samples 5000 ^
  --warmup 2000 ^
  --reps 100 ^
  --n 200,400 ^
  --p 50,150,300 ^
  --nonzero 6,12 ^
  --baseline weibull,exponential,gompertz ^
  --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso ^
  --target-accept 0.9 ^
  --seed 1 ^
  --use-car true ^
  --car-distance bray ^
  --p0 truth ^
  --p0-alpha 0.05 ^
  --outdir results\simulation
