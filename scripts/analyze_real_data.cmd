@echo off
setlocal
cd /d "%~dp0\.."

REM --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso ^

set priors_val=rhs,new-rhs,est-rhs
julia -t 4 --project=. scripts\analyze_real_data.jl ^
  --input data\T1D.csv ^
  --time-col time ^
  --status-col status ^
  --exclude-cols Sex,Antibiotics ^
  --sampler nuts ^
  --samples 5000 ^
  --warmup 2000 ^
  --chains 4 ^
  --priors %priors_val% ^
  --target-accept 0.8 ^
  --seed 1 ^
  --use-car true ^
  --car-distance bray ^
  --p0 auto ^
  --p0-alpha 0.05 ^
  --outdir results\realdata\T1D
