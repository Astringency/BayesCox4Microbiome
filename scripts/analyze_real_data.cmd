@echo off
setlocal
cd /d "%~dp0\.."

julia --project=. scripts\analyze_real_data.jl ^
  --input data\T1D.csv ^
  --sampler mh ^
  --samples 50 ^
  --warmup 20 ^
  --chains 1 ^
  --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso ^
  --use-car true ^
  --car-distance bray ^
  --p0 auto ^
  --p0-alpha 0.05 ^
  --outdir results\realdata\T1D
