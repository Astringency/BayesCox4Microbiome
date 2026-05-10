@echo off
setlocal
cd /d "%~dp0\.."

if not defined GKSwstype set "GKSwstype=100"

REM julia +1.11.7 -t 4 --project=. scripts\analyze_real_data.jl ^
REM   --input data\T1D.csv ^
REM   --time-col time ^
REM   --status-col status ^
REM   --exclude-cols Sex,Antibiotics ^
REM   --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso ^
REM   --sampler nuts ^
REM   --nuts-backend explicit ^
REM   --target-accept 0.95 ^
REM   --samples 5000 ^
REM   --warmup 4000 ^
REM   --chains 4 ^
REM   --multichains true ^
REM   --p0 auto ^
REM   --constraint soft_center^
REM   --diagnose true ^
REM   %*

julia +1.11.7 -t 4 --project=. scripts\analyze_real_data.jl ^
  --input data\T1D.csv ^
  --time-col time ^
  --status-col status ^
  --exclude-cols Sex,Antibiotics ^
  --priors prox-lasso,prox-sslasso ^
  --sampler nuts ^
  --nuts-backend explicit ^
  --target-accept 0.95 ^
  --samples 5000 ^
  --warmup 4000 ^
  --chains 4 ^
  --multichains true ^
  --p0 auto ^
  --constraint prox ^
  --diagnose true ^
  %*

exit /b %ERRORLEVEL%
